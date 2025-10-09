from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, Union
from app.services.file_explore import explore_df
from app.services.bot import chatbot, create_initial_state, process_user_input

router = APIRouter()

@router.post("/financial-assistant")
async def financial_assistant(question: str, file: Optional[Union[UploadFile, str]] = File(None)):
    state = create_initial_state()

    # If file is uploaded, add summary to state
    if file:
        try:
            file_summary = explore_df(file)
            state['file_summary'] = file_summary
        except HTTPException as e:
            raise e

    # Append user question and get bot response
    response_text, state = process_user_input(state, question)
    
    return {"response": response_text}