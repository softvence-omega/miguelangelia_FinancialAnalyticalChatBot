from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Define state
# -----------------------------
class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
    file_summary: Optional[str]  # Optional file context

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatOpenAI(model_name="gpt-4")

# -----------------------------
# File summarization function
# -----------------------------
def summarize_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    df = df.replace([np.inf, -np.inf], np.nan)
    summary = f"File Summary: columns={df.columns.tolist()}, shape={df.shape}, sample={df.head(5).to_dict(orient='records')}"
    return summary

# -----------------------------
# Chat node function
# -----------------------------
def chat_node(state: ChatState):
    messages = state['message']

    if state.get('file_summary'):
        messages_with_context = [HumanMessage(content=f"Financial data context:\n{state['file_summary']}")] + messages
    else:
        messages_with_context = messages

    response = llm(messages_with_context)
    state['message'].append(AIMessage(content=response.content))
    return {'message': state['message'], 'file_summary': state.get('file_summary')}

# -----------------------------
# Graph setup function
# -----------------------------
def create_chatbot():
    checkpointer = InMemorySaver()
    graph = StateGraph(ChatState)
    graph.add_node('chat_node', chat_node)
    graph.add_edge(START, 'chat_node')
    graph.add_edge('chat_node', END)
    chatbot = graph.compile(checkpointer=checkpointer)
    return chatbot

# -----------------------------
# Initialize chatbot
# -----------------------------
chatbot = create_chatbot()

# -----------------------------
# Initialize state
# -----------------------------
def create_initial_state():
    return {
        'message': [HumanMessage(content='Hello! I am your financial assistant.')],
        'file_summary': None
    }

# -----------------------------
# Function to process user input
# -----------------------------
def process_user_input(state: ChatState, user_input: str):
    if user_input.startswith("file:"):
        file_path = user_input[5:].strip()
        if os.path.exists(file_path):
            try:
                file_summary = summarize_file(file_path)
                state['file_summary'] = file_summary
                return f"✅ File context added: {file_path}", state
            except Exception as e:
                return f"❌ Error reading file: {e}", state
        else:
            return "❌ File not found. Please provide a valid path.", state
    else:
        # Append user message
        state['message'].append(HumanMessage(content=user_input))
        # Invoke chatbot
        response = chatbot.invoke(state)
        return response['message'][-1].content, state
