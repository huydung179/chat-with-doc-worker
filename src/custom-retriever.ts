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
}
  
export class CustomRetriever extends BaseRetriever {
  lc_namespace = ["langchain", "retrievers"];
  embeddings: Embeddings
  index: VectorizeIndex
  db: D1Database
  topK: number
  tableName: string

  constructor(
    { embeddings, index, db, topK, tableName, ...fields }: CustomRetrieverInput,
  ) {
    super(fields);
    this.embeddings = embeddings
    this.index = index
    this.db = db
    this.topK = topK
    this.tableName = tableName
  }

  async _getRelevantDocuments(
    query: string,
    runManager?: CallbackManagerForRetrieverRun,
  ): Promise<Document[]> {
    const embeddings = await this.embeddings.embedQuery(query)
    const relevantVectors = await this.index.query(embeddings, { topK: this.topK })
    const ids = relevantVectors.matches.map(match => match.id)
    const placeholders = ids.map(() => "?").join(",")
    const dbQuery = `SELECT * FROM ${this.tableName} WHERE id IN (${placeholders})`
    const results = await this.db.prepare(dbQuery).bind(...ids).all<{ text: string, metadata: Record<string, any> }>()
    return results.results.map(r => {
      return new Document({
        pageContent: r.text,
        metadata: r.metadata,
      })
    })
  }
}