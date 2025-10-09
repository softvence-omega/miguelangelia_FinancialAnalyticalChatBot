from fastapi import APIRouter, HTTPException
from app.services.llm_call import call_openai
from pydantic import BaseModel
from typing import Dict

router = APIRouter()

# Request schema
class DataSummaryRequest(BaseModel):
    data_summary: Dict  # Accepts the JSON/dict summary of the dataset

# Response schema
class InsightsResponse(BaseModel):
    insights: str

@router.post("/summary-and-report-generator", response_model=InsightsResponse)
async def summary_and_report_generator(request: DataSummaryRequest):
    """
    Receives a dataset summary and asks GPT-4 to generate insights and a report.
    """
    try:
        # Call async OpenAI function
        summary = await call_openai(request.data_summary)
        
        # Return as JSON
        return {"summary": summary}

    except HTTPException as he:
        # Re-raise FastAPI HTTP exceptions
        raise he

    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=str(e))
