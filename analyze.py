from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
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
class Visualization(BaseModel):
    title: str
    visual_type: str
    x_axis: str
    y_axis: str
    insight: str
    example_finding: str


class DataReport(BaseModel):
    visualizations: List[Visualization]


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


# ---------- Helper: Structured Report ----------
def get_llm_report(data_preview: str, columns: List[str], stats: Dict) -> DataReport:
    prompt = f"""
    You are a financial data analyst. Analyze this dataset:
    
    Columns: {', '.join(columns)}
    
    Preview:
    {data_preview[:3000]}
    
    Statistics:
    {json.dumps(stats, indent=2)[:2000]}

    Provide a comprehensive analysis in the following EXACT JSON structure:

    {{
        "visualizations": [
            {{
                "title": "Price Trend Over Time",
                "visual_type": "Line Chart",
                "x_axis": "Date",
                "y_axis": "Close",
                "insight": "Analyze stock price movement throughout the period",
                "example_finding": "Upward trend observed from July to December"
            }}
        ]
    }}

    Provide 8-10 visualizations, 5-7 calculated metrics, and 6-8 KPIs.
    Return ONLY valid JSON, no markdown formatting.
    """
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        response_format={"type": "json_object"}
    )
    
    json_response = json.loads(resp.choices[0].message.content)
    return DataReport(**json_response)





# ---------- Route 2: Structured Report ----------
@app.post("/report")
async def generate_report(file: UploadFile = File(...)):
    try:
        df = read_uploaded_file(file)
        preview = df.head(30).to_string(index=False)
        
        # Get numeric statistics
        numeric_summary = df.describe(include='all').fillna("").to_dict()
        
        # Get LLM structured report
        report = get_llm_report(preview, list(df.columns), numeric_summary)

        return JSONResponse({
            "status": "success",
            "data": {
                "metadata": {
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "column_types": df.dtypes.astype(str).to_dict()
                },
                "visualizations": [viz.dict() for viz in report.visualizations]
            }
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )



