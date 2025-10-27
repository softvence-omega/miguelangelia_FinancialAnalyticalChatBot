from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.general_bot import ask_me
import uuid

class GeneralRequest(BaseModel):
    thread_id: Optional[str] = None
    user_id: str
    question: str = "Hi"

class GeneralResponse(BaseModel):
    thread_id: str  # Added this field
    response: str

router = APIRouter()

@router.post("/general-assistant", response_model=GeneralResponse)
async def bot_endpoint(request: GeneralRequest):
    # Generate or use existing thread_id
    if not request.thread_id: 
        thread_id = str(uuid.uuid4())
    else:
        thread_id = request.thread_id
    
    print("thread id ========", thread_id)
    
    # Get response from tutor service
    response = ask_me(
        question=request.question,
        thread_id=thread_id,
        user_id=request.user_id
    )
    
    print("response ------------", response)

    # Return both thread_id and response
    return {
        "thread_id": thread_id,
        "response": response
    }