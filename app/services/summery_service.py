import os
import io
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from openai import AsyncOpenAI
import asyncio
from app.core.config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)

app = FastAPI(title="Dataset Analyzer API")


async def read_csv_async(content: bytes) -> pd.DataFrame:
    return await asyncio.to_thread(pd.read_csv, io.BytesIO(content))


async def read_excel_async(content: bytes) -> pd.DataFrame:
    return await asyncio.to_thread(pd.read_excel, io.BytesIO(content))


async def summarize_dataset(df: pd.DataFrame) -> str:
    """Send dataset summary request to OpenAI LLM."""
    # Offload all pandas operations together
    df_clean = await asyncio.to_thread(
        lambda: df.fillna("").head(100).to_dict(orient="records")
    )
    
    # This is fast enough to stay synchronous
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