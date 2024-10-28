import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages"

export type ChatHistory = {
  role: "human" | "ai" | "system" | "tool"
  content: string
  toolCallId?: string
}

export const historyToChatHistory = (history: ChatHistory[], limit: number = 50) => {
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
