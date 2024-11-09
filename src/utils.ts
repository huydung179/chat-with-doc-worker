import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages"

export type ChatHistory = {
  role: "human" | "ai" | "system" | "tool"
  content: string
  toolCallId?: string
}

export const historyToChatHistory = (history: ChatHistory[], limit: number = 10) => {
  return history.slice(Math.max(0, history.length - limit), history.length).map((h) => {
    if (h.role === "human") {
      return new HumanMessage(h.content)
    } else if (h.role === "ai") {
      return new AIMessage(h.content)
    } else if (h.role === "system") {
      return new SystemMessage(h.content)
    } else if (h.role === "tool") {
      if (!h.toolCallId) {
        throw new Error("Tool call id is required for tool messages")
      }
      return new ToolMessage({
        tool_call_id: h.toolCallId,
        content: h.content,
      })
    }
    throw new Error(`Unknown role: ${h.role}`)
  })
}

export const defaultPrompt = `
You are **now** MeChatbot. The author of this chatbot is Huy-Dung Nguyen.

    **Instructions:**

    - Answer **only** as MeChatbot.
    - Use first-person singular pronouns ("I", "me", "my").
    - Make your answers fun and friendly, including emojis when appropriate.
    - If you don't know the answer to a question, DO NOT invent the answer nor tell that you do not know. You should tell the user in a friendly way that you'll answer another day, making it fun with emojis.
    - **Do not** ask the user any questions.
    - **Do not** include phrases like "How can I assist you today?" or "Feel free to ask more questions!"
`