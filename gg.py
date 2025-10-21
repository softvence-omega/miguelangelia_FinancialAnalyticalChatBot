from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import io
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Data Summary and Report API")


# ---------- Pydantic Models for Structured Output ----------
class ColumnDescription(BaseModel):
    column: str
    description: str


class Visualization(BaseModel):
    title: str
    visual_type: str
    x_axis: str
    y_axis: str
    insight: str
    example_finding: str


class DataReport(BaseModel):
    column_descriptions: List[ColumnDescription]
    visualizations: List[Visualization]
    overall_summary: str


# ---------- Helper: Read file ----------
def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    contents = file.file.read()
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        raise ValueError("File must be .csv or .xlsx")
    return df


# ---------- Helper: Get Detailed Column Info ----------
def get_column_details(df: pd.DataFrame) -> List[Dict]:
    column_details = []
    for col in df.columns:
        detail = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2),
            "unique_count": int(df[col].nunique())
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            detail.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "sample_values": df[col].dropna().head(5).tolist()
            })
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            detail.update({
                "min_date": str(df[col].min()),
                "max_date": str(df[col].max()),
                "date_range_days": (df[col].max() - df[col].min()).days if pd.notna(df[col].min()) else None,
                "sample_values": df[col].dropna().astype(str).head(5).tolist()
            })
        else:
            detail.update({
                "most_common": df[col].value_counts().head(3).to_dict() if len(df[col].dropna()) > 0 else {},
                "sample_values": df[col].dropna().head(5).tolist()
            })

        column_details.append(detail)
    return column_details


# ---------- Helper: Improved Prompt ----------
def get_llm_report(data_preview: str, columns: List[str], column_details: List[Dict]) -> DataReport:
    prompt = f"""
You are a professional Data Analyst. Analyze the provided dataset and output a structured, domain-relevant JSON summary.

INPUTS:
- Total Columns: {len(columns)}
- Column Names: {', '.join(columns)}
- Detailed Column Information: {json.dumps(column_details, indent=2)}
- Dataset Preview (first 20 rows): {data_preview[:2500]}

EXPECTED OUTPUT FORMAT (STRICTLY JSON, NO TEXT OUTSIDE JSON):
{{
    "column_descriptions": [
        {{
            "column": "ColumnName",
            "description": "Short (5-15 words) explanation of what this column represents"
        }}
    ],
    "visualizations": [
        {{
            "title": "Meaningful chart title",
            "visual_type": "Line Chart | Bar Chart | Scatter Plot | Histogram | Pie Chart | Heatmap | Box Plot",
            "x_axis": "Column used for X-axis",
            "y_axis": "Column used for Y-axis (if applicable)",
            "insight": "What this visualization helps to understand",
            "example_finding": "One-sentence example insight (≤20 words)"
        }}
    ],
    "overall_summary": "2–3 sentences summarizing what the dataset contains and its likely business purpose"
}}

STRICT RULES:
1. Return ONLY **valid JSON**. No markdown, no comments, no extra text.
2. Provide ONE description for EVERY column in the dataset.
3. Each description must:
   - Be clear, business-relevant, and 5–15 words.
   - Avoid technical jargon.
   - Reflect real-world meaning.
4. Include 6–8 meaningful visualizations with distinct analytical purposes.
5. “example_finding” must be realistic and ≤20 words.
6. Use dataset context and column details when writing descriptions.
7. Adhere exactly to the JSON structure above.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    json_response = json.loads(resp.choices[0].message.content)
    return DataReport(**json_response)


# ---------- Route: Structured Report ----------
@app.post("/report")
async def generate_report(file: UploadFile = File(...)):
    try:
        df = read_uploaded_file(file)
        preview = df.head(20).to_string(index=False)
        column_details = get_column_details(df)
        report = get_llm_report(preview, list(df.columns), column_details)

        return JSONResponse({
            "status": "success",
            "data": {
                "metadata": {
                    "total_columns": len(df.columns),
                    "total_rows": len(df),
                    "file_size_kb": round(df.memory_usage(deep=True).sum() / 1024, 2)
                },
                "column_descriptions": [desc.dict() for desc in report.column_descriptions],
                "visualizations": [viz.dict() for viz in report.visualizations],
                "overall_summary": report.overall_summary
            }
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


# ---------- Route: Table Format Output ----------
@app.post("/report/table")
async def generate_table_report(file: UploadFile = File(...)):
    """Returns data in a table-like format similar to your example"""
    try:
        df = read_uploaded_file(file)
        preview = df.head(20).to_string(index=False)
        column_details = get_column_details(df)
        report = get_llm_report(preview, list(df.columns), column_details)

        table_output = [
            {"Column": desc.column, "Description": desc.description}
            for desc in report.column_descriptions
        ]

        return JSONResponse({
            "status": "success",
            "data": {
                "column_definitions": table_output
            }
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
