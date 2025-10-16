from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, Union
from app.services.bot import summarize_file, process_user_input

router = APIRouter()

@router.post("/financial-assistant")
async def financial_assistant(thread_id: str, question: str, file: Optional[Union[UploadFile, str]] = File(None)):

    # If file is uploaded, summarize and add to context
    if file:
        try:
            file_summary = summarize_file(file)
            state["file_summary"] = file_summary
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    # Process user input (works for both with/without file)
    response_text, state = process_user_input(thread_id, question)

    return {"response": response_text}
