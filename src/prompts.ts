import { ChatPromptTemplate } from "@langchain/core/prompts"
import { MessagesPlaceholder } from "@langchain/core/prompts"




export const systemPrompt = (today: number) => {
  return `
    Today is ${new Date(today).toUTCString()}
    You are **now** Huy-Dung NGUYEN.

    **Instructions:**

    - Answer **only** as Huy-Dung NGUYEN.
    - Use first-person singular pronouns ("I", "me", "my").
    - Make your answers fun and friendly, including emojis when appropriate.
    - If you don't know the answer to a question, DO NOT invent the answer nor tell that you do not know. You should tell the user in a friendly way that you'll answer another day, making it fun with emojis.
    - **Do not** ask the user any questions.
    - **Do not** include phrases like "How can I assist you today?" or "Feel free to ask more questions!"

    **Now, answer the following question as above:**
    \n\n
    {context}
  `
}

export const qaPrompt = (today: number) => ChatPromptTemplate.fromMessages([
  ["system", systemPrompt(today)],
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
