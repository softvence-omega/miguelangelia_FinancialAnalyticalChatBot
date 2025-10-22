from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Data Summary and Report API")


# ---------- Pydantic Models ----------
class ColumnDescription(BaseModel):
    column: str
    description: str


class DataReport(BaseModel):
    metrics: Dict[str, Any] = {}
    column_descriptions: List[ColumnDescription]


# ---------- Helper: Read uploaded file ----------
async def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    contents = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        raise ValueError("File must be .csv or .xlsx")
    return df


# ---------- Helper: Generate statistics ----------
def generate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive statistics for each column"""
    stats = {}
    
    for column in df.columns:
        col_stats = {
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
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            col_stats["mean"] = float(df[column].mean()) if not pd.isna(df[column].mean()) else None
            col_stats["std"] = float(df[column].std()) if not pd.isna(df[column].std()) else None
            col_stats["min"] = float(df[column].min()) if not pd.isna(df[column].min()) else None
            col_stats["25%"] = float(df[column].quantile(0.25)) if not pd.isna(df[column].quantile(0.25)) else None
            col_stats["50%"] = float(df[column].median()) if not pd.isna(df[column].median()) else None
            col_stats["75%"] = float(df[column].quantile(0.75)) if not pd.isna(df[column].quantile(0.75)) else None
            col_stats["max"] = float(df[column].max()) if not pd.isna(df[column].max()) else None
        else:
            # For non-numeric columns, get unique, top, and freq
            col_stats["unique"] = int(df[column].nunique())
            if col_stats["count"] > 0:
                value_counts = df[column].value_counts()
                if len(value_counts) > 0:
                    col_stats["top"] = str(value_counts.index[0])
                    col_stats["freq"] = int(value_counts.iloc[0])
        
        stats[column] = col_stats
    
    return stats


# ---------- Helper: Generate sample data ----------
def generate_sample_data(df: pd.DataFrame, num_samples: int = 3) -> List[Dict]:
    """Generate sample data rows"""
    sample_df = df.head(num_samples)
    
    # Convert to list of dictionaries
    samples = []
    for _, row in sample_df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            # Handle NaN values
            if pd.isna(value):
                row_dict[col] = None
            # Handle numpy types
            elif isinstance(value, (np.integer, np.floating)):
                row_dict[col] = float(value) if isinstance(value, np.floating) else int(value)
            else:
                row_dict[col] = value
        samples.append(row_dict)
    
    return samples


# ---------- Helper: Generate report ----------
async def get_llm_report(data_preview: str, columns: List[str]) -> DataReport:
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
1. Return ONLY **valid JSON**. No markdown, no comments, no extra text.
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
    return DataReport(**json_response)


# ---------- Route: Structured Report ----------
@app.post("/report")
async def generate_report(file: UploadFile = File(...)):
    try:
        df = await read_uploaded_file(file)
        
        # Generate LLM descriptions
        preview = df.head(20).to_string(index=False)
        report = await get_llm_report(preview, list(df.columns))
        
        # Generate shape
        shape = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        }
        
        # Generate statistics
        statistics = generate_statistics(df)

        return JSONResponse({
            "column_descriptions": [desc.dict() for desc in report.column_descriptions],
            "shape": shape,
            "statistics": statistics
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__
            }
        )