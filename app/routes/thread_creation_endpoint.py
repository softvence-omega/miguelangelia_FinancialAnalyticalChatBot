from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.financial_bot import create_thread

router = APIRouter()

@router.post("/threads")
async def create_thread_endpoint(file_summary: str, user_id: str):
    thread_id = create_thread(user_id, file_summary)
    return {"thread_id": thread_id}
