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
}
  
export class CustomRetriever extends BaseRetriever {
  lc_namespace = ["langchain", "retrievers"];
  embeddings: Embeddings
  index: VectorizeIndex
  db: D1Database
  topK: number

  constructor(
    { embeddings, index, db, topK, ...fields }: CustomRetrieverInput,
  ) {
    super(fields);
    this.embeddings = embeddings
    this.index = index
    this.db = db
    this.topK = topK
  }

  async _getRelevantDocuments(
    query: string,
    runManager?: CallbackManagerForRetrieverRun,
  ): Promise<Document[]> {
    const embeddings = await this.embeddings.embedQuery(query)
    const relevantVectors = await this.index.query(embeddings, { topK: this.topK })
    const ids = relevantVectors.matches.map(match => match.id)
    const dbQuery = `SELECT * FROM rag_docs WHERE id IN (${ids.join(",")})`
    const results = await this.db.prepare(dbQuery).bind().all<{ content: string, metadata: Record<string, any> }>()

    return results.results.map(result => new Document({
      pageContent: result.content,
      metadata: result.metadata,
    }))
  }
}