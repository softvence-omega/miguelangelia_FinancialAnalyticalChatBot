# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import json

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


@app.post("/summary")
async def analyze_file(file: UploadFile = File(...)):
    try:
        content = await file.read()

        if file.filename.endswith(".csv"):
            df = await read_csv_async(content)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = await read_excel_async(content)
        else:
            return JSONResponse({"status": "error", "message": "Unsupported file type"}, status_code=400)

        summary = await summarize_dataset(df)
        return {"summary": summary}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
