import {
  BaseRetriever,
  type BaseRetrieverInput,
} from "@langchain/core/retrievers";
import type { CallbackManagerForRetrieverRun } from "@langchain/core/callbacks/manager";
import { Document } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";
  
interface RelevantVectorsResponse {
  result: {
    matches: { id: string }[];
  };
}

async function getRelevantDocuments(
  accountId: string,
  vectorizeApiKey: string,
  indexName: string,
  embeddings: number[],
  topK: number,
  filter?: object,
): Promise<RelevantVectorsResponse> {
  const response = await fetch(`https://api.cloudflare.com/client/v4/accounts/${accountId}/vectorize/v2/indexes/${indexName}/query`, {
    method: 'POST',
    body: JSON.stringify({ vector: embeddings, topK, filter }),
    headers: {'Content-Type': 'application/json', Authorization: `Bearer ${vectorizeApiKey}`},
  })
  if (!response.ok) {
    throw new Error(`Failed to get relevant documents`)
  }
  return response.json()
}

export interface CustomRetrieverInput extends BaseRetrieverInput {
  embeddings: Embeddings,
  accountId: string,
  vectorizeApiKey: string,
  indexName: string,
  db: D1Database,
  topK: number,
  tableName: string,
  filter?: object,
}

export class CustomRetriever extends BaseRetriever {
  lc_namespace = ["langchain", "retrievers"];
  accountId: string
  vectorizeApiKey: string
  indexName: string
  embeddings: Embeddings
  db: D1Database
  topK: number
  tableName: string
  filter?: object

  constructor(
    { accountId, vectorizeApiKey, embeddings, indexName, db, topK, tableName, filter, ...fields }: CustomRetrieverInput,
  ) {
    super(fields);
    this.embeddings = embeddings
    this.accountId = accountId
    this.vectorizeApiKey = vectorizeApiKey
    this.indexName = indexName
    this.db = db
    this.topK = topK
    this.tableName = tableName
    this.filter = filter
  }

  async _getRelevantDocuments(
    query: string,
    runManager?: CallbackManagerForRetrieverRun,
  ): Promise<Document[]> {
    const embeddings = await this.embeddings.embedQuery(query)
    const relevantVectors = await getRelevantDocuments(this.accountId, this.vectorizeApiKey, this.indexName, embeddings, this.topK, this.filter)
    const ids = relevantVectors.result.matches.map(m => m.id)
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