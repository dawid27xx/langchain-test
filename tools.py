"""
This file is responsible for defining the tools available for the agent.
These are functions that the agent can call and the agent 
will automatically pass the required parameters (if applicable)
"""

from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from document_db import get_document_by_id
from vector import vector_store  

# define tool
# the doc string is used as a description
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# tool error handlings
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
        
# document lookup
@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for relevant information and return document IDs and metadata."""
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant information found."
    return "\n\n".join([
        f"ID: {r.metadata['id']}\nTitle: {r.metadata.get('title', '')}\nSource: {r.metadata.get('source', '')}\nDate: {r.metadata.get('date', '')}\nContent: {r.page_content}"
        for r in results
    ])
    
    
@tool
def get_full_document(doc_id: str) -> str:
    """Retrieve the full document content and metadata by its ID."""
    doc = get_document_by_id(doc_id)
    if not doc:
        return f"No document found with ID: {doc_id}"
    return (
        f"ID: {doc['id']}\n"
        f"Title: {doc['title']}\n"
        f"Source: {doc['source']}\n"
        f"Date: {doc['date']}\n"
        f"Content:\n{doc['content']}"
    )
    
# runtime defines the context window for a function
# ToolRuntime is the entire context inluding: state, context, store, streaming, config, and tool call ID.
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"
