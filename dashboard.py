from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from app.core.config import settings
load_dotenv()
client = OpenAI(api_key=settings.openai_api_key)

app = FastAPI(title="Data Summary and Report API")


# ---------- Pydantic Models for Structured Output ----------
class ColumnDescription(BaseModel):
    column: str
    description: str


class Visualization(BaseModel):
    title: str
    visual_type: str
    chart_data: List[Dict[str, Any]]  # Dynamic chart data
    insight: str
    example_finding: str


class DataReport(BaseModel):
    metrics: Dict[str, Any]
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
You are a professional Data Analyst. Analyze the provided dataset and output a structured, domain-relevant JSON summary with embedded chart data.

INPUTS:
- Total Columns: {len(columns)}
- Column Names: {', '.join(columns)}
- Detailed Column Information: {json.dumps(column_details, indent=2)}
- Dataset Preview (first 20 rows): {data_preview[:2500]}

EXPECTED OUTPUT FORMAT (STRICTLY JSON, NO TEXT OUTSIDE JSON):
{{
    "metrics": {{
        "key_metric_1": 450000,
        "key_metric_2": 1000,
        "key_metric_3": 450
    }},
    "visualizations": [
        {{
            "title": "Meaningful chart title",
            "visual_type": "Line Chart | Bar Chart | Scatter Plot | Histogram | Pie Chart | Heatmap | Box Plot",
            "chart_data": [
                {{"label": "Category1", "value": 1234, "percentage": 45}},
                {{"label": "Category2", "value": 5678, "percentage": 55}}
            ],
            "insight": "What this visualization helps to understand",
            "example_finding": "One-sentence example insight (≤20 words)"
        }}
    ],
    "overall_summary": "2–3 sentences summarizing what the dataset contains and its likely business purpose"
}}

STRICT RULES:
1. Return ONLY **valid JSON**. No markdown, no comments, no extra text.
2. **metrics**: Generate 3-7 key performance indicators relevant to the dataset domain (e.g., total_sales, avg_order_value, total_profit, profit_margin_avg for sales data OR total_records, avg_price, price_range for stock data). Use realistic metric names based on available columns.
3. Provide ONE description for EVERY column in the dataset.
4. Each description must:
   - Be clear, business-relevant, and 5–15 words.
   - Avoid technical jargon.
   - Reflect real-world meaning.
5. Include 5–8 meaningful visualizations with distinct analytical purposes.
6. **chart_data**: For EACH visualization, generate realistic sample data arrays (3-6 items) based on the dataset columns:
   - For "Bar Chart" or "Pie Chart": Use format like [{{"category": "Electronics", "sales": 180000, "percentage": 40}}]
   - For "Line Chart": Use format like [{{"month": "2024-01", "sales": 35000, "orders": 78}}]
   - For "Scatter Plot": Use format like [{{"x_value": 100, "y_value": 250}}]
   - For "Heatmap": Use format like [{{"row": "Product A", "col": "Region 1", "value": 45}}]
   - Ensure data is contextually relevant to the column being visualized
7. "example_finding" must be realistic and ≤20 words.
8. Use dataset context and column details when writing descriptions.
9. Adhere exactly to the JSON structure above.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    json_response = json.loads(resp.choices[0].message.content)
    return DataReport(**json_response)


# ---------- Route: Structured Report ----------
@app.post("/generate-dashboard")
async def generate_report(file: UploadFile = File(...)):
    try:
        df = read_uploaded_file(file)
        preview = df.head(20).to_string(index=False)
        column_details = get_column_details(df)
        report = get_llm_report(preview, list(df.columns), column_details)

        return JSONResponse({
            "status": "success",
            "data": {
                "metrics": report.metrics,
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


