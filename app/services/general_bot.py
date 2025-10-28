from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from app.services.memory import MongoSaver
from app.services.title_generator import generate_thread_title
from app.core.config import history_collection
from app.core.config import settings
import pickle
import uuid
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Initialize MongoSaver
checkpointer = MongoSaver(history_collection)

# -----------------------------
# 1️⃣ create state
# -----------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -----------------------------
# 2️⃣ initialize LLM (chat-based model)
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini",api_key=settings.openai_api_key)

# -----------------------------
# 3️⃣ define chat node
# -----------------------------
def chat_node(state: ChatState):
    """
    state: ChatState
    - state['messages'] is the list of previous messages
    """
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# -----------------------------
# 4️⃣ build graph
# -----------------------------
# checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
chatbot = graph.compile(checkpointer=checkpointer)

# -----------------------------
# 5️⃣ function to ask questions
# -----------------------------
def ask_me(question: str, thread_id: str , user_id: str):
    
    # Create config with thread_id for checkpointing
    config = {
        "configurable": {
            "thread_id": thread_id,   
            "user_id": user_id      
        }
    }
    
    # Get current state to check if it's the first message
    current_state = chatbot.get_state(config)
    # print("current_state-------------", current_state)
    
    # Prepare messages list
    messages = []
    
    # If first question (no messages in state), add SystemMessage
    if not current_state.values.get('messages'):
        messages.append(
            SystemMessage(
                content=f"You are a helpful financial assistant."
                        "Provide concise and accurate answers based on the user's financial data and queries."
            )
        )
        thread_title = generate_thread_title(question)
        print(f"Generated thread title: {thread_title}")

        # Save initial document with title
        checkpoint_id = str(uuid.uuid4())  # new checkpoint id
        checkpoint_data = pickle.dumps({"channel_values": {"messages": messages}})
        metadata_data = pickle.dumps({"session_title": thread_title})
        
        history_collection.insert_one({
            "thread_id": thread_id,
            "user_id": user_id,
            "session_title": thread_title,
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint_data,
            "metadata": metadata_data,
            "created_at": datetime.utcnow()
        })

    
    # Add HumanMessage for user question
    messages.append(HumanMessage(content=question))
    
    # Invoke chatbot with config
    response = chatbot.invoke(
        {"messages": messages},
        config=config
    )
    
    # Extract latest AIMessage content
    answer = response['messages'][-1].content

    return answer

# -----------------------------
# 6️⃣ interactive loop example
# -----------------------------
if __name__ == "__main__":
    thread_id = "1"  # You can change this for different conversations
    user_id = "user_123"

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        answer = ask_me(user_input, thread_id, user_id)
        print(f"AI: {answer}\n")
