from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, Union
from app.services.financial_bot import summarize_file, process_user_input

router = APIRouter()

@router.post("/financial-assistant")
async def financial_assistant(thread_id: str, question: str):


    # Process user input (works for both with/without file)
    response_text, state = process_user_input(thread_id, question)
    
    if response_text is None:
        raise HTTPException(status_code=404, detail=state)  # state contains error message here
    return {"response": response_text}
