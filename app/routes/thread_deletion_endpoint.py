from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.financial_bot import delete_thread

router = APIRouter()

@router.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    success = delete_thread(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"detail": "Thread deleted successfully"}