import { Hono } from "hono";
import { Ai } from '@cloudflare/ai'
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createRetrievalChain } from "langchain/chains/retrieval"
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"
import { qaPrompt } from "./prompts";
import { contextualizeQPrompt } from "./prompts";
import { CustomRetriever } from "./custom-retriever";
import { stream } from "hono/streaming";

type Note = {
  id: number;
  text: string;
}

type Env = {
  AI: Ai;
  DB: D1Database;
  VECTOR_INDEX: VectorizeIndex;
  OPENAI_API_KEY: string;
  local_model_cache: KVNamespace;
}

const app = new Hono<{ Bindings: Env }>();

app.get('/', async (c) => {
  const question = c.req.query('question')
  if (!question) {
    return c.text("Missing question", 400);
  }

  const embeddings = new OpenAIEmbeddings({
    apiKey: c.env.OPENAI_API_KEY,
  })

  const retriever = new CustomRetriever({
    embeddings,
    index: c.env.VECTOR_INDEX,
    db: c.env.DB,
    topK: 2,
  })

  const llm = new ChatOpenAI({
    modelName: "gpt-4o-mini",
    temperature: 0.9,
    apiKey: c.env.OPENAI_API_KEY,
    streaming: true,
  })

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever,
    rephrasePrompt: contextualizeQPrompt,
  })

  const questionAnswerChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt,
  })

  const ragChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: questionAnswerChain,
  })


  const eventStream = await ragChain.streamEvents({
    input: question,
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

// app.post("/notes", async (c: Context<Env>) => {
//   const { text } = await c.req.json();
//   if (!text) {
//     return c.text("Missing text", 400);
//   }

//   const { results } = await (c.env as Env).DB.prepare(
//     "INSERT INTO notes (text) VALUES (?) RETURNING *",
//   )
//     .bind(text)
//     .run();

//   const record = results.length ? results[0] : null;

//   if (!record) {
//     return c.text("Failed to create note", 500);
//   }

//   const { data } = await (c.env as Env).AI.run("@cf/baai/bge-base-en-v1.5", {
//     text: [text],
//   });
//   const values = data[0];

//   if (!values) {
//     return c.text("Failed to generate vector embedding", 500);
//   }

//   const { id } = record as { id: number };
//   const inserted = await (c.env as Env).VECTOR_INDEX.upsert([
//     {
//       id: id.toString(),
//       values,
//     },
//   ]);

//   return c.json({ id, text, inserted });
// });

// app.delete("/notes/:id", async (c) => {
//   const { id } = c.req.param();

//   const query = `DELETE FROM notes WHERE id = ?`;
//   await (c.env as Env).DB.prepare(query).bind(id).run();

//   await (c.env as Env).VECTOR_INDEX.deleteByIds([id]);

//   return c.status(204);
// });

// app.onError((err, c) => {
//   return c.text(err.message, 500);
// });

export default app;
