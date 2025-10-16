from fastapi import APIRouter, HTTPException
from app.services.bot import create_thread

router = APIRouter()

@router.post("/create-thread")
async def create_session_endpoint():
    try:
        thread_id = create_thread()
        return thread_id
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   