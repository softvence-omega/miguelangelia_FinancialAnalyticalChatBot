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
from fastapi import UploadFile
import uuid

load_dotenv()

# -----------------------------
# Define state
# -----------------------------
class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
    file_summary: Optional[str]  # Optional file context
    thread_id: str  # Add thread_id to state

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatOpenAI(model_name="gpt-4")

# Store active threads and their states
active_threads: dict = {}

# -----------------------------
# File summarization function
# -----------------------------
def summarize_file(file: UploadFile) -> str:
    try:
        # Read file into pandas DataFrame based on extension
        ext = os.path.splitext(file.filename)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(file.file)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file.file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

        # Clean and summarize
        df = df.replace([np.inf, -np.inf], np.nan)

        summary = {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "sample": df.head(5).to_dict(orient="records"),
        }

        return f"File Summary: {summary}"

    except Exception as e:
        raise ValueError(f"Failed to summarize file: {str(e)}")

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
    return {
        'message': state['message'], 
        'file_summary': state.get('file_summary'),
        'thread_id': state['thread_id']
    }

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
# Create a new thread
# -----------------------------
def create_thread() -> str:
    """Creates a new conversation thread and returns thread_id"""
    thread_id = str(uuid.uuid4())
    active_threads[thread_id] = create_initial_state(thread_id)
    return thread_id

# -----------------------------
# Initialize state with thread_id
# -----------------------------
def create_initial_state(thread_id: str):
    return {
        'message': [HumanMessage(content='Hello!')],
        'file_summary': None,
        'thread_id': thread_id
    }

# Get thread state
# -----------------------------
def get_thread_state(thread_id: str) -> Optional[dict]:
    """Retrieve state for a specific thread"""
    return active_threads.get(thread_id)

# Delete thread
# -----------------------------
def delete_thread(thread_id: str) -> bool:
    """Delete a conversation thread"""
    if thread_id in active_threads:
        del active_threads[thread_id]
        return True
    return False

# List all threads
# -----------------------------
def list_threads() -> list:
    """Get all active thread IDs"""
    return list(active_threads.keys())

# -----------------------------
# Function to process user input
# -----------------------------
def process_user_input(thread_id: str, user_input: str):
    """
    Handles user input intelligently for a specific thread:
    - If file_summary exists → analyze with file context.
    - Otherwise → act as general financial assistant.
    """
    # Get thread state
    state = get_thread_state(thread_id)
    # print("state------------", state)
    if not state:
        return None, f"Thread {thread_id} not found"

    state['message'].append(HumanMessage(content=user_input))

    # If file context exists → analytical mode
    if state.get("file_summary"):
        context_prompt = f"""
        You are a financial data analyst.
        Use the following dataset summary to answer the question.

        Dataset Summary:
        {state['file_summary']}

        Question:
        {user_input}
        """
        state['message'].append(HumanMessage(content=context_prompt))
    else:
        # Knowledge-based mode (no file)
        context_prompt = f"""
        You are a financial assistant with expert knowledge in markets, investments, and accounting.
        Answer the following question using your general financial knowledge.

        Question:
        {user_input}
        """
        state['message'].append(HumanMessage(content=context_prompt))

    # Call the LLM through the graph with thread_id
    config = {"configurable": {"thread_id": thread_id}}
    response = chatbot.invoke(state, config=config)

    # Update thread state
    active_threads[thread_id] = response

    return response['message'][-1].content, state

# Upload file to thread
# -----------------------------
def upload_file_to_thread(thread_id: str, file: UploadFile):
    """Upload and summarize a file for a specific thread"""
    state = get_thread_state(thread_id)
    if not state:
        return f"Thread {thread_id} not found"
    
    try:
        file_summary = summarize_file(file)
        state['file_summary'] = file_summary
        active_threads[thread_id] = state
        return f"File uploaded successfully to thread {thread_id}"
    except Exception as e:
        return f"Error uploading file: {str(e)}"