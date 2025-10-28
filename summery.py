import io
import json
import asyncio
import pandas as pd
import httpx
from fastapi import FastAPI, Query, APIRouter
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from app.core.config import settings  # Make sure this is correct

# Initialize OpenAI client
client = AsyncOpenAI(api_key=settings.openai_api_key)

# Initialize FastAPI app
app = FastAPI(title="Dataset Analyzer API")

# --------------------------
# Async utility functions
# --------------------------
async def read_csv_async(content: bytes) -> pd.DataFrame:
    return await asyncio.to_thread(pd.read_csv, io.BytesIO(content))

async def read_excel_async(content: bytes) -> pd.DataFrame:
    return await asyncio.to_thread(pd.read_excel, io.BytesIO(content))

async def summarize_dataset(df: pd.DataFrame) -> str:
    """Send dataset summary request to OpenAI LLM."""
    # Clean and limit dataset
    df_clean = await asyncio.to_thread(lambda: df.fillna("").head(100).to_dict(orient="records"))
    
    prompt = (
        f"Analyze this dataset and provide a summary in 6-10 sentences. "
        f"Include insights, patterns, and potential issues. Do not exceed 10 sentences:\n"
        f"{json.dumps(df_clean)}"
    )
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# --------------------------
# API Router
# --------------------------
router = APIRouter()

@router.post("/summary")
async def analyze_file_from_url(file_url: str = Query(..., description="Cloudinary file URL")):
    try:
        async with httpx.AsyncClient() as client_http:
            resp = await client_http.get(file_url)
            if resp.status_code != 200:
                return JSONResponse(
                    {"status": "error", "message": f"Failed to fetch file: HTTP {resp.status_code}"},
                    status_code=resp.status_code
                )
            content = resp.content

        # Determine file type from URL
        if file_url.endswith(".csv"):
            df = await read_csv_async(content)
        elif file_url.endswith((".xlsx", ".xls")):
            df = await read_excel_async(content)
        else:
            return JSONResponse({"status": "error", "message": "Unsupported file type"}, status_code=400)

        summary = await summarize_dataset(df)
        return {"summary": summary}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Include router in the app
app.include_router(router)
