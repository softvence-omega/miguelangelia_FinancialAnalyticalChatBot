import io
import json
import os
import pandas as pd
import numpy as np
import asyncio
from fastapi import UploadFile
from typing import List, Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.schemas.file_exploar_shcema import (
    ColumnStatistics,
    LLMReport
)
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# ---------- Helper: Read uploaded file ----------
# Wrap file reading in thread pool
async def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    contents = await file.read()
    filename = file.filename.lower()
    
    def _read_file():
        if filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx", ".xls"):
            return pd.read_excel(io.BytesIO(contents))
        else:
            raise ValueError("File must be .csv, , .xls or .xlsx")
    
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

# ---------- Helper: Generate sample data ----------
def generate_sample_data(df: pd.DataFrame, num_samples: int = 3) -> List[Dict[str, Any]]:
    sample_df = df.head(num_samples)
    samples = []
    for _, row in sample_df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                row_dict[col] = None
            elif isinstance(value, (np.integer, np.floating)):
                row_dict[col] = float(value) if isinstance(value, np.floating) else int(value)
            else:
                row_dict[col] = value
        samples.append(row_dict)
    return samples

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
    return LLMReport(**json_response)
