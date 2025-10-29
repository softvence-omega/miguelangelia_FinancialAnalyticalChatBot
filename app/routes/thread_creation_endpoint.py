from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.financial_bot import create_thread
from pydantic import BaseModel

router = APIRouter()

class ThreadCreationRequest(BaseModel):
    file_summary: dict
    user_id: str

@router.post("/threads")
async def create_thread_endpoint(request: ThreadCreationRequest):
    thread_id = create_thread(request.user_id, request.file_summary)
    return {"thread_id": thread_id}
