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
  D1_DATA_TABLE_NAME: string;
  D1_UPDATE_HISTORY_TABLE_NAME: string;
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
    tableName: c.env.D1_DATA_TABLE_NAME,
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
  const { text, values, metadata, updateId, description } = await c.req.json();
  if (!text || !values || !metadata || !updateId || !description) {
    return c.json({
      "message": "Missing text, values, metadata, updateId, or description",
      "ok": false,
    }, 400);
  }
  const { instanceName, createdBy } = metadata
  if (!instanceName || !createdBy) {
    return c.json({
      "message": "Missing instanceName or createdBy",
      "ok": false,
    }, 400);
  }

  const { results: existingRecord } = await c.env.DB.prepare(
    `SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE text = ? AND created_by = ? AND instance_name = ?`,
  )
  .bind(text, createdBy, instanceName)
  .run<{ id: string }>()

  if (existingRecord.length > 0) {
    try {
      await c.env.DB.prepare(
        `INSERT INTO ${c.env.D1_UPDATE_HISTORY_TABLE_NAME} (id, knowledge_id, description) VALUES (?, ?, ?)`,
      )
      .bind(existingRecord[0].id, updateId, description)
      .run()
      return c.json({
        "message": "The knowledge is updated successfully",
        "ok": true,
      }, 200);
    } catch (e) {
      if (
        e instanceof Error &&
        e.message.includes("UNIQUE constraint failed") &&
        e.message.includes("SQLITE_CONSTRAINT")
      ) {
        return c.json({
          "message": "The knowledgeId already exists",
          "ok": false,
        }, 409)
      }
    }
  } else {
    const { results } = await c.env.DB.prepare(
      `INSERT INTO ${c.env.D1_DATA_TABLE_NAME} (text, created_by, instance_name) VALUES (?, ?, ?) RETURNING *`,
    )
    .bind(text, createdBy, instanceName)
    .run<{ id: string }>();

    await c.env.DB.prepare(
      `INSERT INTO ${c.env.D1_UPDATE_HISTORY_TABLE_NAME} (id, knowledge_id, description) VALUES (?, ?, ?)`,
    )
    .bind(results[0].id, updateId, description)
    .run()
    
    await c.env.VECTOR_INDEX.upsert([
    {
        id: results[0].id,
        values: values as VectorFloatArray,
        metadata: metadata as Record<string, any>,
      },
    ]);
    
    return c.json({
      "message": "The text and vector data is created successfully",
      "ok": true,
      "id": results[0].id,
    }, 200);
  }
});

app.delete("/vector/:id", async (c) => {
  const { id } = c.req.param();

  const query = `DELETE FROM ${c.env.D1_DATA_TABLE_NAME} WHERE id = ?`;
  await c.env.DB.prepare(query).bind(id).run();
  await c.env.VECTOR_INDEX.deleteByIds([id]);
  return c.json({
    "message": "Deleted chatbot text data and vector data",
    "ok": true,
  }, 200);
});

app.get("/chatbot/:userId/list", async (c) => {
  const { userId } = c.req.param();
  const query = `SELECT instance_name FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ?`;
  const { results } = await c.env.DB.prepare(query).bind(userId).run();
  const uniqueInstanceNames = [...new Set(results.map((result) => result.instance_name))];
  return c.json({
    "instanceNames": uniqueInstanceNames,
    "ok": true,
  }, 200);
});

app.delete("/chatbot/:userId/:chatbotName", async (c) => {
  const { userId, chatbotName } = c.req.param();
  const selectQuery = `SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ? AND instance_name = ?`;
  const { results } = await c.env.DB.prepare(selectQuery).bind(userId, chatbotName).run();
  const ids = results.map((result) => result.id) as string[]

  const deleteQuery = `DELETE FROM ${c.env.D1_DATA_TABLE_NAME} WHERE id IN (${ids.map((id) => `?`).join(",")})`;
  await c.env.DB.prepare(deleteQuery).bind(...ids).run();
  await c.env.VECTOR_INDEX.deleteByIds(ids);
  return c.json({
    "message": "Deleted chatbot text data and vector data",
    "ok": true,
  }, 200);
});

app.delete("/chatbot/:userId/:chatbotName/:knowledgeId", async (c) => {
  const { userId, chatbotName, knowledgeId } = c.req.param();

  // get all ids of the chatbot of the user
  const selectQuery = `SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ? AND instance_name = ?`;
  const { results: existingRecords } = await c.env.DB.prepare(selectQuery).bind(userId, chatbotName).run();
  const ids = existingRecords.map((record) => record.id) as string[]

  // delete the update history of the knowledge
  const deleteQuery = `DELETE FROM ${c.env.D1_UPDATE_HISTORY_TABLE_NAME} WHERE id IN (${ids.map((id) => `?`).join(",")}) AND knowledge_id = ?`;
  await c.env.DB.prepare(deleteQuery).bind(...ids, knowledgeId).run();
  return c.json({
    "message": "Deleted chatbot update history",
    "ok": true,
  }, 200);
});

app.onError((err, c) => {
  return c.json({
    "message": err.message,
    "ok": false,
  }, 500);
});

export default app;
