import io
import json
import asyncio
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import aiohttp
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Data Summary and Report API")

# ---------- Pydantic Models ----------
class ColumnDescription(BaseModel):
    column: str
    description: str

class Shape(BaseModel):
    rows: int
    columns: int

class ColumnStatistics(BaseModel):
    count: int
    unique: Optional[int] = None
    top: Optional[str] = None
    freq: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    percentile_25: Optional[float] = Field(None, alias="25%")
    percentile_50: Optional[float] = Field(None, alias="50%")
    percentile_75: Optional[float] = Field(None, alias="75%")
    max: Optional[float] = None

    class Config:
        populate_by_name = True
        validate_by_name = True

class DataAnalysisResponse(BaseModel):
    column_descriptions: List[ColumnDescription]
    shape: Shape
    statistics: Dict[str, ColumnStatistics]

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    error_type: str

class LLMReport(BaseModel):
    column_descriptions: List[ColumnDescription]

# ---------- Helper: Read uploaded file ----------
async def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    contents = await file.read()
    filename = file.filename.lower()
    
    def _read_file():
        if filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(contents))
        else:
            raise ValueError("File must be .csv or .xlsx")
    
    return await asyncio.to_thread(_read_file)

# ---------- Helper: Generate statistics ----------
def generate_statistics(df: pd.DataFrame) -> Dict[str, ColumnStatistics]:
    stats = {}
    for column in df.columns:
        col_data = {
            "count": int(df[column].count()),
            "unique": None,
            "top": None,
            "freq": None,
            "mean": None,
            "std": None,
            "min": None,
            "25%": None,
            "50%": None,
            "75%": None,
            "max": None
        }
        if pd.api.types.is_numeric_dtype(df[column]):
            col_data["mean"] = float(df[column].mean()) if not pd.isna(df[column].mean()) else None
            col_data["std"] = float(df[column].std()) if not pd.isna(df[column].std()) else None
            col_data["min"] = float(df[column].min()) if not pd.isna(df[column].min()) else None
            col_data["25%"] = float(df[column].quantile(0.25)) if not pd.isna(df[column].quantile(0.25)) else None
            col_data["50%"] = float(df[column].median()) if not pd.isna(df[column].median()) else None
            col_data["75%"] = float(df[column].quantile(0.75)) if not pd.isna(df[column].quantile(0.75)) else None
            col_data["max"] = float(df[column].max()) if not pd.isna(df[column].max()) else None
        else:
            col_data["unique"] = int(df[column].nunique())
            if col_data["count"] > 0:
                value_counts = df[column].value_counts()
                if len(value_counts) > 0:
                    col_data["top"] = str(value_counts.index[0])
                    col_data["freq"] = int(value_counts.iloc[0])
        stats[column] = ColumnStatistics(**col_data)
    return stats

# ---------- Helper: Generate report via LLM ----------
async def get_llm_report(data_preview: str, columns: List[str]) -> LLMReport:
    prompt = f"""
You are a professional Data Analyst. Analyze the provided dataset and output a structured, domain-relevant JSON summary.

INPUTS:
- Total Columns: {len(columns)}
- Column Names: {', '.join(columns)}
- Dataset Preview (first 20 rows): {data_preview[:2500]}

EXPECTED OUTPUT FORMAT (STRICTLY JSON, NO TEXT OUTSIDE JSON):
{{
    "column_descriptions": [
        {{
            "column": "ColumnName",
            "description": "Short (5-15 words) explanation of what this column represents"
        }}
    ]
}}

STRICT RULES:
1. Return ONLY valid JSON. No markdown, no comments, no extra text.
2. Provide ONE description for EVERY column in the dataset.
3. Each description must:
   - Be clear, business-relevant, and 5–15 words.
   - Avoid technical jargon.
   - Reflect real-world meaning.
4. Use dataset context and column details when writing descriptions.
5. Adhere exactly to the JSON structure above.
"""
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    json_response = json.loads(response.choices[0].message.content)
    return LLMReport(**json_response)

# ---------- Route: File upload ----------
@app.post("/report-file", response_model=DataAnalysisResponse)
async def generate_report_file(file: UploadFile = File(...)):
    try:
        df = await read_uploaded_file(file)
        preview = await asyncio.to_thread(lambda: df.head(20).to_string(index=False))
        llm_report = await get_llm_report(preview, list(df.columns))
        shape = Shape(rows=int(df.shape[0]), columns=int(df.shape[1]))
        statistics = await asyncio.to_thread(generate_statistics, df)
        return DataAnalysisResponse(
            column_descriptions=llm_report.column_descriptions,
            shape=shape,
            statistics=statistics
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message=str(e),
                error_type=type(e).__name__
            ).dict()
        )

# ---------- Route: Cloudinary / URL ----------
@app.post("/report-url", response_model=DataAnalysisResponse)
async def generate_report_url(file_url: str = Query(..., description="Cloudinary file URL")):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                if resp.status != 200:
                    raise ValueError(f"Failed to download file: status {resp.status}")
                content = await resp.read()

        if file_url.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file_url.lower().endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("File must be .csv or .xlsx")

        preview = df.head(20).to_string(index=False)
        llm_report = await get_llm_report(preview, list(df.columns))
        shape = Shape(rows=int(df.shape[0]), columns=int(df.shape[1]))
        statistics = await asyncio.to_thread(generate_statistics, df)

        return DataAnalysisResponse(
            column_descriptions=llm_report.column_descriptions,
            shape=shape,
            statistics=statistics
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message=str(e),
                error_type=type(e).__name__
            ).dict()
        )
