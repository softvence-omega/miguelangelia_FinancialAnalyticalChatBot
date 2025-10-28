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
    try:
        if not request.thread_id: 
            thread_id = str(uuid.uuid4())
        else:
            thread_id = request.thread_id
        
        print("thread id ========", thread_id)
        
        # Get response from tutor service
        response, total_tokens, input_tokens, output_tokens, model = ask_me(
            question=request.question,
            thread_id=thread_id,
            user_id=request.user_id
        )
        # calculate cost
        # cost = (input_tokens * 0.00003) + (output_tokens * 0.00006)

        # Return both thread_id and response
        return {
            "thread_id": thread_id,
            "response": response,
            "used_tokens": total_tokens,
            "model": model,
            # "cost": cost
        }
    
    except Exception as e:
        return {
            "status": "error",
            "thread_id": request.thread_id if request.thread_id else "N/A",
            "response": f"Error: {str(e)}"
        }