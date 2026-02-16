import {
  BaseRetriever,
  type BaseRetrieverInput,
} from "@langchain/core/retrievers";
import type { CallbackManagerForRetrieverRun } from "@langchain/core/callbacks/manager";
import { Document } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";

export interface CustomRetrieverInput extends BaseRetrieverInput {
  embeddings: Embeddings,
  index: VectorizeIndex,
  db: D1Database,
  topK: number,
  tableName: string,
  filter?: VectorizeVectorMetadataFilter,
}

export class CustomRetriever extends BaseRetriever {
  lc_namespace = ["langchain", "retrievers"];
  index: VectorizeIndex
  embeddings: Embeddings
  db: D1Database
  topK: number
  tableName: string
  filter?: VectorizeVectorMetadataFilter

  constructor(
    { embeddings, index, db, topK, tableName, filter, ...fields }: CustomRetrieverInput,
  ) {
    super(fields);
    this.embeddings = embeddings
    this.index = index
    this.db = db
    this.topK = topK
    this.tableName = tableName
    this.filter = filter
  }

  async _getRelevantDocuments(
    query: string,
    runManager?: CallbackManagerForRetrieverRun,
  ): Promise<Document[]> {
    const start = Date.now();

    // 1. Embedding
    const tEmbed = Date.now();
    const embeddings = await this.embeddings.embedQuery(query)
    console.log(`[Perf] Embeddings generation: ${Date.now() - tEmbed}ms`);

    // 2. Vectorize Query
    const tVector = Date.now();
    const relevantVectors = await this.index.query(embeddings, { topK: this.topK, filter: this.filter })
    console.log(`[Perf] Vectorize query: ${Date.now() - tVector}ms`);

    // 3. D1 Query
    const ids = relevantVectors.matches.map(m => m.id)
    if (ids.length === 0) {
      console.log(`[Perf] CustomRetriever total: ${Date.now() - start}ms (0 matches)`);
      return [];
    }

    const tDB = Date.now();
    const placeholders = ids.map(() => "?").join(",")
    const dbQuery = `SELECT * FROM ${this.tableName} WHERE id IN (${placeholders})`
    const results = await this.db.prepare(dbQuery).bind(...ids).all<{ text: string, metadata: Record<string, any> }>()
    console.log(`[Perf] D1 query: ${Date.now() - tDB}ms`);

    console.log(`[Perf] CustomRetriever total: ${Date.now() - start}ms`);

    return results.results.map(r => {
      return new Document({
        pageContent: r.text,
        metadata: r.metadata,
      })
    })
  }
}