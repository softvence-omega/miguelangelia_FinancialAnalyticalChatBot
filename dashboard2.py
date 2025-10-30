from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import os
import json
import warnings
import requests
from openai import OpenAI
from dotenv import load_dotenv
from app.core.config import settings

load_dotenv()
client = OpenAI(api_key=settings.openai_api_key)

app = FastAPI(title="Data Summary and Report API")


# ---------- Updated Pydantic Models ----------
class TimeAggregation(BaseModel):
    period: str
    value: float
    trend: Optional[str] = None


class TimeAggregations(BaseModel):
    monthly: Optional[List[TimeAggregation]] = None
    quarterly: Optional[List[TimeAggregation]] = None
    yearly: Optional[List[TimeAggregation]] = None


class Visualization(BaseModel):
    title: str
    visual_type: str
    chart_data: List[Dict[str, Any]]
    time_aggregations: Optional[TimeAggregations] = None
    insight: str
    example_finding: str
    date_column_used: Optional[str] = None


class DataReport(BaseModel):
    metrics: Dict[str, Any]
    visualizations: List[Visualization]
    overall_summary: str


# ---------- Helper: Read file from URL ----------
def read_file_from_url(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {e}")

    if url.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif url.lower().endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(status_code=400, detail="File must be .csv or .xlsx")
    return df


# ---------- Helper: Detect Date Columns ----------
def detect_date_columns(df: pd.DataFrame) -> List[str]:
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pd.to_datetime(df[col], errors="raise")
                date_cols.append(col)
            except:
                pass
    return date_cols


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
            "unique_count": int(df[col].nunique()),
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
                "sample_values": df[col].dropna().astype(str).head(5).tolist(),
                "is_date_column": True
            })
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test_date = pd.to_datetime(df[col].dropna().head(10), errors="coerce")
                if test_date.notna().sum() > 5:
                    detail["is_date_column"] = True
            except:
                pass

            detail.update({
                "most_common": df[col].value_counts().head(3).to_dict() if len(df[col].dropna()) > 0 else {},
                "sample_values": df[col].dropna().head(5).tolist()
            })

        column_details.append(detail)
    return column_details


# ---------- Helper: Analyze Dataset ----------
def analyze_dataset_type(df: pd.DataFrame, column_details: List[Dict]) -> Dict[str, Any]:
    characteristics = {
        "has_time_series": False,
        "has_categories": False,
        "has_numeric": False,
        "has_geographic": False,
        "dataset_type": "general",
        "date_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "geographic_columns": []
    }

    for detail in column_details:
        col_name = detail["name"].lower()
        if detail.get("is_date_column") or any(k in col_name for k in ["date", "time", "year"]):
            characteristics["date_columns"].append(detail["name"])
            characteristics["has_time_series"] = True
        if "min" in detail and "max" in detail:
            characteristics["numeric_columns"].append(detail["name"])
            characteristics["has_numeric"] = True
        if "most_common" in detail and detail.get("unique_count", 0) < len(df) * 0.5:
            characteristics["categorical_columns"].append(detail["name"])
            characteristics["has_categories"] = True
        if any(geo in col_name for geo in ["country", "city", "state", "region", "location", "address", "zip", "postal"]):
            characteristics["geographic_columns"].append(detail["name"])
            characteristics["has_geographic"] = True

    if characteristics["has_time_series"] and characteristics["has_numeric"]:
        characteristics["dataset_type"] = "time_series"
    elif characteristics["has_geographic"]:
        characteristics["dataset_type"] = "geographic"
    elif characteristics["has_categories"] and characteristics["has_numeric"]:
        characteristics["dataset_type"] = "categorical_analysis"
    elif characteristics["has_numeric"]:
        characteristics["dataset_type"] = "numeric_analysis"

    return characteristics


# ---------- Helper: Universal LLM Report Generator ----------
def get_llm_report(data_preview: str, columns: List[str], column_details: List[Dict], date_columns: List[str]) -> DataReport:
    # Analyze dataset characteristics
    df_temp = pd.DataFrame()  # We'll use column_details instead
    characteristics = analyze_dataset_type(df_temp, column_details)
    
    has_date_columns = len(date_columns) > 0 or characteristics["has_time_series"]
    date_info = f"Date Columns: {', '.join(date_columns if date_columns else characteristics['date_columns'])}" if has_date_columns else "No date columns"
    
    # Build dynamic examples based on dataset type
    visualization_examples = """
        EXAMPLE VISUALIZATIONS BY TYPE:
        
        TIME-SERIES DATA (if date columns exist):
        {{
            "title": "Daily Closing Prices Over Time",
            "visual_type": "Line Chart",
            "chart_data": [
                {{"label": "2024-01-01", "value": 45.5}},
                {{"label": "2024-01-02", "value": 46.2}}
            ],
            "time_aggregations": {{
                "monthly": [
                    {{"period": "January 2024", "value": 45.8, "trend": "up"}},
                    {{"period": "February 2024", "value": 47.2, "trend": "up"}}
                ],
                "quarterly": [
                    {{"period": "2024Q1", "value": 46.5, "trend": "up"}}
                ],
                "yearly": [
                    {{"period": "2023", "value": 44.2}},
                    {{"period": "2024", "value": 46.5}}
                ]
            }},
            "date_column_used": "Date",
            "insight": "Shows price trends and volatility",
            "example_finding": "Prices increased 15% from January to March 2024."
        }}
        
        CATEGORICAL DATA:
        {{
            "title": "Sales Distribution by Category",
            "visual_type": "Bar Chart",
            "chart_data": [
                {{"label": "Electronics", "value": 125000, "percentage": 35}},
                {{"label": "Clothing", "value": 98000, "percentage": 28}}
            ],
            "insight": "Compares performance across categories",
            "example_finding": "Electronics leads with 35% of total sales."
        }}
        
        NUMERIC CORRELATION:
        {{
            "title": "Price vs Sales Volume",
            "visual_type": "Scatter Plot",
            "chart_data": [
                {{"x_value": 29.99, "y_value": 1250}},
                {{"x_value": 49.99, "y_value": 890}}
            ],
            "insight": "Reveals relationship between two variables",
            "example_finding": "Higher prices correlate with lower sales volume."
        }}
        
        DISTRIBUTION:
        {{
            "title": "Order Value Distribution",
            "visual_type": "Histogram",
            "chart_data": [
                {{"range": "0-50", "count": 450}},
                {{"range": "51-100", "count": 320}}
            ],
            "insight": "Shows frequency distribution of values",
            "example_finding": "Most orders fall between $0-50 range."
        }}
    """
    
    # Conditional time aggregation instructions
    time_agg_instructions = ""
    if has_date_columns:
        time_agg_instructions = """
   - **time_aggregations** (REQUIRED for time-series visualizations):
     * Include "monthly", "quarterly", and "yearly" nested objects
     * Use period formats: "January 2024", "2024Q1", "2024"
     * Add "trend" field: "up", "down", or "stable" (compare consecutive periods)
     * Generate 3-5 monthly, 2-4 quarterly, 2-3 yearly data points
     * Set "date_column_used" to the actual date column from dataset
     * Only include for Line Charts, Area Charts showing time-based data"""
    else:
        time_agg_instructions = """
   - **time_aggregations**: Omit this field if no date/time columns exist"""
    
    prompt = f"""
You are an expert Data Analyst. Analyze this dataset and generate a comprehensive JSON report with domain-specific insights and visualizations.

DATASET INFORMATION:
- Total Columns: {len(columns)}
- Column Names: {', '.join(columns)}
- {date_info}
- Dataset Characteristics: {characteristics['dataset_type']}
- Numeric Columns: {len(characteristics['numeric_columns'])}
- Categorical Columns: {len(characteristics['categorical_columns'])}

DETAILED COLUMN METADATA:
{json.dumps(column_details, indent=2)}

DATASET PREVIEW (first 20 rows):
{data_preview[:3000]}

{visualization_examples}

OUTPUT STRUCTURE (VALID JSON ONLY):
{{
    "metrics": {{
        "metric_1": <number>,
        "metric_2": <number>,
        "metric_3": <number>
    }},
    "visualizations": [
        {{
            "title": "...",
            "visual_type": "Line Chart | Bar Chart | Area Chart | Pie Chart | Scatter Plot | Histogram | Heatmap",
            "chart_data": [...],
            "time_aggregations": {{ ... }},  // Only if time-series data
            "date_column_used": "...",  // Only if time-series
            "insight": "...",
            "example_finding": "..."
        }}
    ],
    "overall_summary": "..."
}}

GENERATION RULES:
1. **STRICTLY VALID JSON** - No markdown, comments, or extra text
2. **metrics**: 5-8 domain-relevant KPIs based on available data
   - Sales data: total_sales, avg_order_value, total_orders, revenue_growth, etc.
   - Stock data: avg_price, price_volatility, total_volume, price_range, etc.
   - User data: total_users, avg_age, active_users, conversion_rate, etc.
3. **visualizations**: 6-10 diverse, meaningful charts
   - Choose appropriate visual_type based on data characteristics
   - Ensure each visualization serves a distinct analytical purpose
4. **chart_data**: Generate realistic sample data (4-6 items per chart)
   - Bar/Pie: [{{"label": "X", "value": 123, "percentage": 45}}]
   - Line/Area: [{{"label": "Period", "value": 123}}]
   - Scatter: [{{"x_value": 12, "y_value": 34}}]
   - Histogram: [{{"range": "0-10", "count": 45}}]
{time_agg_instructions}
5. **insight**: Clear explanation of what the chart reveals (10-20 words)
6. **example_finding**: Specific, data-driven insight (≤20 words)
7. **overall_summary**: 2-3 sentences about dataset purpose and key characteristics
8. Use actual column names from the dataset in your analysis
9. Tailor metrics and visualizations to the domain (finance, retail, analytics, etc.)
10. Ensure all numeric values are realistic based on the data preview

CRITICAL: Output ONLY the JSON object. No preamble, no explanation, no markdown.
"""
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    json_response = json.loads(resp.choices[0].message.content)
    return DataReport(**json_response)


# ---------- Route: Structured Report from Cloudinary Link ----------
@app.post("/generate-dashboard")
async def generate_report(file_url: str):
    try:
        df = read_file_from_url(file_url)
        preview = df.head(20).to_string(index=False)
        column_details = get_column_details(df)
        date_columns = detect_date_columns(df)
        report = get_llm_report(preview, list(df.columns), column_details, date_columns)

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
            content={"status": "error", "message": str(e), "error_type": type(e).__name__}
        )
