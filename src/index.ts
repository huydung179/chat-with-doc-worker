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
  CLOUDFLARE_ACCOUNT_ID: string;
  CLOUDFLARE_VECTORIZE_API_KEY: string;
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
  const { question, history, indexName, filter } = await c.req.json()
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

app.post("/vectors/upsert", async (c) => {
  const { text, values, metadata } = await c.req.json();
  if (!text || !values || !metadata) {
    return c.text("Missing text, values, or metadata", 400);
  }

  const { results } = await c.env.DB.prepare(
    "INSERT INTO ChatbotTextData (text) VALUES (?) RETURNING *",
  )
    .bind(text)
    .run();

  const record = results.length ? results[0] : null;

  if (!record) {
    return c.text("Failed to create chatbot text data", 500);
  }

  const recordId = record.id as string
  const inserted = await c.env.VECTOR_INDEX.upsert([
    {
      id: recordId,
      values: values as VectorFloatArray,
      metadata: metadata as Record<string, any>,
    },
  ]);

  return c.json({ id: recordId, inserted });
});

app.delete("/notes/:id", async (c) => {
  const { id } = c.req.param();

  const query = `DELETE FROM notes WHERE id = ?`;
  await c.env.DB.prepare(query).bind(id).run();

  await c.env.VECTOR_INDEX.deleteByIds([id]);

  return c.status(204);
});

app.onError((err, c) => {
  return c.text(err.message, 500);
});

export default app;
