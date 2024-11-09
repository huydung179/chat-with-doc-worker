import { ChatPromptTemplate } from "@langchain/core/prompts"
import { MessagesPlaceholder } from "@langchain/core/prompts"




export const systemPrompt = (today: number, prompt: string) => {
  return `
    Today is ${new Date(today).toUTCString()}
    ${prompt}

    **Now, answer the following question as above:**
    \n\n
    {context}
  `
}

export const qaPrompt = (today: number, prompt: string) => ChatPromptTemplate.fromMessages([
  ["system", systemPrompt(today, prompt)],
  new MessagesPlaceholder("chat_history"),
  ["human", "\nQuestion: {input}"],
])


export const contextualizeQSystemPrompt =
  "Given a chat history and the latest user question " +
  "which might reference context in the chat history, " +
  "formulate a standalone question which can be understood " +
  "without the chat history. Do NOT answer the question, " +
  "just reformulate it if needed and otherwise return it as is."

export const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
  ["system", contextualizeQSystemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
])
