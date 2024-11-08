import { Hono } from "hono";
import { Ai } from '@cloudflare/ai'
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createRetrievalChain } from "langchain/chains/retrieval"
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"
import { qaPrompt } from "./prompts";
import { contextualizeQPrompt } from "./prompts";
import { CustomRetriever } from "./custom-retriever";
import { cors } from "hono/cors";
import { historyToChatHistory } from "./utils";
import { StringOutputParser } from "@langchain/core/output_parsers";

type Env = {
  AI: Ai;
  DB: D1Database;
  VECTOR_INDEX: VectorizeIndex;
  D1_TABLE_NAME: string;
  OPENAI_API_KEY: string;
  OPENAI_MODEL_NAME: string;
  OPENAI_MODEL_TEMPERATURE: string;
  OPENAI_MODEL_MAX_TOKEN: string;
  MODEL_TOPK: string;
  HISTORY_MODEL_LIMIT: string;
  FINAL_OUTPUT_PARSER_NAME: string;
}

const app = new Hono<{ Bindings: Env }>();

app.use('*', cors({
  origin: '*',
  allowHeaders: ['*'],
  allowMethods: ['POST', 'GET', 'OPTIONS'],
  exposeHeaders: ['Content-Length'],
  maxAge: 600,
  credentials: true,
}))


app.post('/', async (c) => {
  const today = Date.now()
  const { question, history, filter } = await c.req.json()
  if (!question) {
    return c.text("Missing question", 400);
  }

  const chatHistory = historyToChatHistory(history || [], parseInt(c.env.HISTORY_MODEL_LIMIT))


  const embeddings = new OpenAIEmbeddings({
    apiKey: c.env.OPENAI_API_KEY,
  })

  const retriever = new CustomRetriever({
    embeddings,
    index: c.env.VECTOR_INDEX,
    db: c.env.DB,
    topK: parseInt(c.env.MODEL_TOPK),
    filter,
    tableName: c.env.D1_TABLE_NAME,
  })

  const llm = new ChatOpenAI({
    modelName: c.env.OPENAI_MODEL_NAME,
    temperature: parseFloat(c.env.OPENAI_MODEL_TEMPERATURE),
    apiKey: c.env.OPENAI_API_KEY,
    streaming: true,
    maxTokens: parseInt(c.env.OPENAI_MODEL_MAX_TOKEN),
  })

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever,
    rephrasePrompt: contextualizeQPrompt,
  })

  const finalOutputParser = new StringOutputParser()
  finalOutputParser.name = c.env.FINAL_OUTPUT_PARSER_NAME
  const questionAnswerChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt(today),
    outputParser: finalOutputParser,
  })

  const ragChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: questionAnswerChain,
  })


  const eventStream = ragChain.streamEvents({
    input: question,
    chat_history: chatHistory,
  }, {
    version: "v2",
    encoding: "text/event-stream",
  })

  return new Response(eventStream, {
    headers: {
      "Content-Type": "text/event-stream",
    },
  });
});

app.post("/vector", async (c) => {
  const { text, values, metadata } = await c.req.json();
  if (!text || !values || !metadata) {
    return c.text("Missing text, values, or metadata", 400);
  }
  const { instanceName, createdBy } = metadata
  if (!instanceName || !createdBy) {
    return c.text("Missing instanceName or createdBy", 400);
  }

  try {
    const { results } = await c.env.DB.prepare(
      `INSERT INTO ${c.env.D1_TABLE_NAME} (text, created_by, instance_name) VALUES (?, ?, ?) RETURNING *`,
    )
    .bind(text, createdBy, instanceName)
    .run<{ id: string }>();
    
    const recordId = results[0].id
    await c.env.VECTOR_INDEX.upsert([
      {
        id: recordId,
        values: values as VectorFloatArray,
        metadata: metadata as Record<string, any>,
      },
    ]);
    
    return c.json({ "message": "The text and vector data is created with id: " + recordId }, 200);

  } catch (e) {
    // handle unique constraint case in D1
    if (
      e instanceof Error &&
      e.message.includes("UNIQUE constraint failed") &&
      e.message.includes("SQLITE_CONSTRAINT")
    ) {
      return c.json({ "message": "The text and vector data already exists, no update is done"}, 200)
    }
    throw e
  }
});

app.delete("/vector/:id", async (c) => {
  const { id } = c.req.param();

  const query = `DELETE FROM ${c.env.D1_TABLE_NAME} WHERE id = ?`;
  await c.env.DB.prepare(query).bind(id).run();
  await c.env.VECTOR_INDEX.deleteByIds([id]);
  return c.json({ "message": "Deleted chatbot text data and vector data" }, 200);
});

app.get("/chatbot/:userId/list", async (c) => {
  const { userId } = c.req.param();
  const query = `SELECT instance_name FROM ${c.env.D1_TABLE_NAME} WHERE created_by = ?`;
  const { results } = await c.env.DB.prepare(query).bind(userId).run();
  const uniqueInstanceNames = [...new Set(results.map((result) => result.instance_name))];
  return c.json({ instanceNames: uniqueInstanceNames }, 200);
});

app.delete("/chatbot/:userId/:chatbotName", async (c) => {
  const { userId, chatbotName } = c.req.param();
  const selectQuery = `SELECT id FROM ${c.env.D1_TABLE_NAME} WHERE created_by = ? AND instance_name = ?`;
  const { results } = await c.env.DB.prepare(selectQuery).bind(userId, chatbotName).run();
  const ids = results.map((result) => result.id) as string[]

  const deleteQuery = `DELETE FROM ${c.env.D1_TABLE_NAME} WHERE id IN (${ids.map((id) => `?`).join(",")})`;
  await c.env.DB.prepare(deleteQuery).bind(...ids).run();
  await c.env.VECTOR_INDEX.deleteByIds(ids);
  return c.json({ "message": "Deleted chatbot text data and vector data" }, 200);
});

app.onError((err, c) => {
  return c.text(err.message, 500);
});

export default app;
