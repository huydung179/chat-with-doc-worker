import { Context, Hono, Next } from "hono";
import { Ai } from '@cloudflare/ai'
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createRetrievalChain } from "langchain/chains/retrieval"
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"
import { qaPrompt } from "./prompts";
import { contextualizeQPrompt } from "./prompts";
import { CustomRetriever } from "./custom-retriever";
import { cors } from "hono/cors";
import { defaultPrompt, historyToChatHistory } from "./utils";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HTTP_STATUS_RESPONSES } from "./error-constants";

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
  SECRET_BEARER_TOKEN: string;
  D1_PROMPT_TABLE_NAME: string;
}

const app = new Hono<{ Bindings: Env }>();

app.use('*', cors({
  origin: "*",
  allowHeaders: ["*"],
  allowMethods: ["*"],
  exposeHeaders: ["Content-Length"],
  maxAge: 600,
  credentials: true,
}))

const authMiddleware = async (c: Context, next: Next) => {
  const authHeader = c.req.header('Authorization');
  if (authHeader === `Bearer ${c.env.SECRET_BEARER_TOKEN}`) {
    await next();
  } else {
    return c.text('Unauthorized', 401);
  }
};

app.put("/chatbot/:userId/:chatbotName/insert-knowledge", authMiddleware, async (c) => {
  const { userId, chatbotName } = c.req.param();
  const { text, values, metadata, domainKnowledgeName, description } = await c.req.json();
  if (!text || !values || !metadata || !domainKnowledgeName || !description) {
    return c.json({
      ...HTTP_STATUS_RESPONSES.BAD_REQUEST,
      message: "Missing text, values, metadata, domainKnowledgeName, or description",
    }, {
      status: HTTP_STATUS_RESPONSES.BAD_REQUEST.status,
    });
  }

  const { results: existingRecord } = await c.env.DB.prepare(
    `SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE text = ? AND created_by = ? AND instance_name = ?`,
  )
  .bind(text, userId, chatbotName)
  .run<{ id: string }>()

  if (existingRecord.length > 0) {
    try {
      await c.env.DB.prepare(
        `INSERT INTO ${c.env.D1_UPDATE_HISTORY_TABLE_NAME} (id, domain_knowledge_name, description) VALUES (?, ?, ?)`,
      )
      .bind(existingRecord[0].id, domainKnowledgeName, description)
      .run()
      return c.json({
        ...HTTP_STATUS_RESPONSES.OK,
        message: "The knowledge is updated successfully",
      }, {
        status: HTTP_STATUS_RESPONSES.OK.status,
      });
    } catch (e) {
      if (
        e instanceof Error &&
        e.message.includes("UNIQUE constraint failed") &&
        e.message.includes("SQLITE_CONSTRAINT")
      ) {
        return c.json({
          ...HTTP_STATUS_RESPONSES.CONFLICT,
          message: "The knowledgeId already exists",
        }, {
          status: HTTP_STATUS_RESPONSES.CONFLICT.status,
        });
      }
      throw e;
    }
  } else {
    const { results } = await c.env.DB.prepare(
      `INSERT INTO ${c.env.D1_DATA_TABLE_NAME} (text, created_by, instance_name) VALUES (?, ?, ?) RETURNING *`,
    )
    .bind(text, userId, chatbotName)
    .run<{ id: string }>();

    await c.env.DB.prepare(
      `INSERT INTO ${c.env.D1_UPDATE_HISTORY_TABLE_NAME} (id, domain_knowledge_name, description) VALUES (?, ?, ?)`,
    )
    .bind(results[0].id, domainKnowledgeName, description)
    .run()
    
    await c.env.VECTOR_INDEX.upsert([
    {
        id: results[0].id,
        values: values as VectorFloatArray,
        metadata: {
          ...metadata as Record<string, any>,
          createdBy: userId,
          instanceName: chatbotName,
        },
      },
    ]);
    
    return c.json({
      ...HTTP_STATUS_RESPONSES.OK,
      message: "The text and vector data is created successfully",
      id: results[0].id,
    }, {
      status: HTTP_STATUS_RESPONSES.OK.status,
    });
  }
});

app.delete("/vector/:id", authMiddleware, async (c) => {
  const { id } = c.req.param();

  const query = `DELETE FROM ${c.env.D1_DATA_TABLE_NAME} WHERE id = ?`;
  await c.env.DB.prepare(query).bind(id).run();
  await c.env.VECTOR_INDEX.deleteByIds([id]);
  return c.json({
    ...HTTP_STATUS_RESPONSES.OK,
    message: "Deleted chatbot text data and vector data",
  }, {
    status: HTTP_STATUS_RESPONSES.OK.status,
  });
});

app.post("/chatbot/:userId/:chatbotName", authMiddleware, async (c) => {
  const { userId, chatbotName } = c.req.param();
  const query = `INSERT INTO ${c.env.D1_DATA_TABLE_NAME} (created_by, instance_name) VALUES (?, ?) RETURNING *`;
  const { results } = await c.env.DB.prepare(query).bind(userId, chatbotName).run();
  await c.env.DB.prepare(
    `INSERT INTO ${c.env.D1_PROMPT_TABLE_NAME} (id, prompt) VALUES (?, ?)`,
  )
  .bind(results[0].id, defaultPrompt)
  .run()
  return c.json({
    ...HTTP_STATUS_RESPONSES.OK,
    message: "Created chatbot",
  }, {
    status: HTTP_STATUS_RESPONSES.OK.status,
  });
});

app.get("/chatbot/:userId/list", authMiddleware, async (c) => {
  const { userId } = c.req.param();
  const query = `SELECT instance_name FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ?`;
  const { results } = await c.env.DB.prepare(query).bind(userId).run();
  const uniqueInstanceNames = [...new Set(results.map((result) => result.instance_name))];
  return c.json({
    ...HTTP_STATUS_RESPONSES.OK,
    instanceNames: uniqueInstanceNames,
  }, {
    status: HTTP_STATUS_RESPONSES.OK.status,
  });
});

app.get("/chatbot/:userId/:chatbotName/list", authMiddleware, async (c) => {
  const { userId, chatbotName } = c.req.param();
  const query = `SELECT domain_knowledge_name FROM ${c.env.D1_UPDATE_HISTORY_TABLE_NAME} WHERE id IN (SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ? AND instance_name = ?)`;
  const { results } = await c.env.DB.prepare(query).bind(userId, chatbotName).run();
  const uniqueDomainKnowledgeNames = [...new Set(results.map((result) => result.domain_knowledge_name))];
  return c.json({
    ...HTTP_STATUS_RESPONSES.OK,
    domainKnowledgeNames: uniqueDomainKnowledgeNames,
  }, {
    status: HTTP_STATUS_RESPONSES.OK.status,
  });
});

app.get("/chatbot/:userId/:chatbotName/prompt", authMiddleware, async (c) => {
  const { userId, chatbotName } = c.req.param();
  const selectQuery = `SELECT prompt FROM ${c.env.D1_PROMPT_TABLE_NAME} WHERE id = (SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ? AND instance_name = ?)`;
  const { results } = await c.env.DB.prepare(selectQuery).bind(userId, chatbotName).run();
  return c.json({
    ...HTTP_STATUS_RESPONSES.OK,
    prompt: results.length > 0 ? results[0].prompt : "",
  }, {
    status: HTTP_STATUS_RESPONSES.OK.status,
  });
});

app.post("/chatbot/:userId/:chatbotName/prompt", authMiddleware, async (c) => {
  const { userId, chatbotName } = c.req.param();
  const { prompt } = await c.req.json();

  const selectQuery = `SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ? AND instance_name = ?`;
  const { results: promptId } = await c.env.DB.prepare(selectQuery).bind(userId, chatbotName).run<{ id: string }>();

  if (promptId.length === 0) {
    return c.json({
      ...HTTP_STATUS_RESPONSES.NOT_FOUND,
      message: "The chatbot does not exist",
    }, {
      status: HTTP_STATUS_RESPONSES.NOT_FOUND.status,
    });
  }

  const existingPromptQuery = `SELECT prompt FROM ${c.env.D1_PROMPT_TABLE_NAME} WHERE id = ?`;
  const { results: existingPrompt } = await c.env.DB.prepare(existingPromptQuery).bind(promptId[0].id).run<{ prompt: string }>(); 

  if (existingPrompt.length > 0) {
    const updateQuery = `UPDATE ${c.env.D1_PROMPT_TABLE_NAME} SET prompt = ? WHERE id = ?`;
    await c.env.DB.prepare(updateQuery).bind(prompt, promptId[0].id).run();
  } else {
    const insertQuery = `INSERT INTO ${c.env.D1_PROMPT_TABLE_NAME} (id, prompt) VALUES (?, ?)`;
    await c.env.DB.prepare(insertQuery).bind(promptId[0].id, prompt).run();
  }

  return c.json({
    ...HTTP_STATUS_RESPONSES.OK,
    message: "Set chatbot prompt",
  }, {
    status: HTTP_STATUS_RESPONSES.OK.status,
  });
});

app.delete("/chatbot/:userId/:chatbotName", authMiddleware, async (c) => {
  const { userId, chatbotName } = c.req.param();
  const selectQuery = `SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ? AND instance_name = ?`;
  const { results } = await c.env.DB.prepare(selectQuery).bind(userId, chatbotName).run();
  const ids = results.map((result) => result.id) as string[]

  const deleteQuery = `DELETE FROM ${c.env.D1_DATA_TABLE_NAME} WHERE id IN (${ids.map((id) => `?`).join(",")})`;
  await c.env.DB.prepare(deleteQuery).bind(...ids).run();
  await c.env.VECTOR_INDEX.deleteByIds(ids);
  return c.json({
    ...HTTP_STATUS_RESPONSES.OK,
    message: "Deleted chatbot text data and vector data",
  }, {
    status: HTTP_STATUS_RESPONSES.OK.status,
  });
});

app.delete("/chatbot/:userId/:chatbotName/:domainKnowledgeName", authMiddleware, async (c) => {
  const { userId, chatbotName, domainKnowledgeName } = c.req.param();

  // get all ids of the chatbot of the user
  const selectQuery = `SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ? AND instance_name = ?`;
  const { results: existingRecords } = await c.env.DB.prepare(selectQuery).bind(userId, chatbotName).run();
  const ids = existingRecords.map((record) => record.id) as string[]

  // delete the update history of the knowledge
  const deleteQuery = `DELETE FROM ${c.env.D1_UPDATE_HISTORY_TABLE_NAME} WHERE id IN (${ids.map((id) => `?`).join(",")}) AND domain_knowledge_name = ?`;
  try {
    await c.env.DB.prepare(deleteQuery).bind(...ids, domainKnowledgeName).run();
    return c.json({
      ...HTTP_STATUS_RESPONSES.OK,
      message: "Deleted chatbot update history",
    }, {
      status: HTTP_STATUS_RESPONSES.OK.status,
    });
  } catch (e) {
    return c.json({
      ...HTTP_STATUS_RESPONSES.NOT_FOUND,
      message: "The knowledge name does not exist",
    }, {
      status: HTTP_STATUS_RESPONSES.NOT_FOUND.status,
    });
  }
});

app.post('/', async (c) => {
  const today = Date.now()
  const { question, history, filter } = await c.req.json()
  if (!question) {
    return c.text("Missing question", 400);
  }
  const { createdBy, instanceName } = filter
  if (!createdBy || !instanceName) {
    return c.json({
      ...HTTP_STATUS_RESPONSES.BAD_REQUEST,
      message: "Missing createdBy or instanceName in the filter",
    }, {
      status: HTTP_STATUS_RESPONSES.BAD_REQUEST.status,
    });
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

  const { results: promptId } = await c.env.DB.prepare(
    `SELECT id FROM ${c.env.D1_DATA_TABLE_NAME} WHERE created_by = ? AND instance_name = ?`,
  )
  .bind(createdBy, instanceName)
  .run<{ id: string }>()

  const { results: prompt } = await c.env.DB.prepare(
    `SELECT prompt FROM ${c.env.D1_PROMPT_TABLE_NAME} WHERE id = ?`,
  )
  .bind(promptId[0].id)
  .run<{ prompt: string }>()

  const finalOutputParser = new StringOutputParser()
  finalOutputParser.name = c.env.FINAL_OUTPUT_PARSER_NAME
  const questionAnswerChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt(today, prompt[0].prompt),
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

app.onError((err, c) => {
  return c.json({
    ...HTTP_STATUS_RESPONSES.INTERNAL_SERVER_ERROR,
    message: err.message,
  }, {
    status: HTTP_STATUS_RESPONSES.INTERNAL_SERVER_ERROR.status,
  });
});

export default app;
