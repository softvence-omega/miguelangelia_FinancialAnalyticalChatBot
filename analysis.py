from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io, os, json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI(title="Data Visualization Report API")

# ---------- Helper: Read File ----------
def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    contents = file.file.read()
    if file.filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(contents))
    elif file.filename.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(contents))
    raise ValueError("File must be .csv or .xlsx")

# ---------- Helper: Detect Date Columns ----------
def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that contain date/datetime data"""
    date_columns = []
    for col in df.columns:
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
        
        # Try to parse as date
        try:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    # Try parsing with utc=True to avoid timezone warning
                    test_parse = pd.to_datetime(sample, errors='raise', utc=True)
                    # Only add if successfully parsed
                    if test_parse is not None:
                        date_columns.append(col)
        except:
            continue
    
    return date_columns

# ---------- Helper: Parse Date Columns ----------
def parse_date_columns(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """Convert date columns to datetime and add period columns"""
    df = df.copy()
    
    for col in date_columns:
        try:
            # Parse with utc=True, then convert to timezone-naive
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            
            # Convert to timezone-naive datetime
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
            
            # Only add derived columns if we have valid datetime data
            if df[col].notna().any():
                # Add derived time period columns
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.to_period('M').astype(str)
                df[f'{col}_quarter'] = df[col].dt.to_period('Q').astype(str)
                df[f'{col}_month_name'] = df[col].dt.strftime('%B %Y')
            
        except Exception as e:
            print(f"Error parsing date column {col}: {e}")
            # Remove from date_columns list if parsing failed
            if col in date_columns:
                date_columns.remove(col)
            continue
    
    return df

# ---------- Helper: Aggregate Data by Time Period ----------
def aggregate_by_time_period(df: pd.DataFrame, date_col: str, value_col: str, 
                             period: str = 'month') -> pd.DataFrame:
    """Aggregate numeric data by time period (month, quarter, year)"""
    
    period_map = {
        'month': f'{date_col}_month_name',
        'quarter': f'{date_col}_quarter',
        'year': f'{date_col}_year'
    }
    
    period_col = period_map.get(period)
    if period_col not in df.columns:
        return None
    
    # Aggregate data
    if pd.api.types.is_numeric_dtype(df[value_col]):
        agg_df = df.groupby(period_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()
        agg_df.columns = ['period', 'total', 'average', 'count']
        
        # Convert all values to JSON-serializable types
        agg_df['period'] = agg_df['period'].astype(str)
        agg_df['total'] = agg_df['total'].astype(float)
        agg_df['average'] = agg_df['average'].astype(float)
        agg_df['count'] = agg_df['count'].astype(int)
        
        return agg_df.sort_values('period')
    
    return None

# ---------- Helper: Get Time-based Stats ----------
def get_time_based_stats(df: pd.DataFrame, date_columns: List[str]) -> Dict:
    """Generate statistics about date ranges and periods"""
    stats = {}
    
    for col in date_columns:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            valid_dates = df[col].dropna()
            if len(valid_dates) > 0:
                stats[col] = {
                    'min_date': str(valid_dates.min().strftime('%Y-%m-%d')),
                    'max_date': str(valid_dates.max().strftime('%Y-%m-%d')),
                    'total_records': int(len(valid_dates)),
                    'unique_months': int(df[f'{col}_month'].nunique()),
                    'unique_quarters': int(df[f'{col}_quarter'].nunique()),
                    'unique_years': int(df[f'{col}_year'].nunique())
                }
    
    return stats

# ---------- Helper: Column Details ----------
def get_column_details(df: pd.DataFrame):
    column_details = []
    for col in df.columns:
        info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "unique_count": int(df[col].nunique()),
        }
        
        # Add sample values for better context
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            sample_vals = df[col].dropna().unique()[:5].tolist()
            # Convert any non-serializable values to strings
            info['sample_values'] = [str(v) for v in sample_vals]
        elif pd.api.types.is_numeric_dtype(df[col]):
            info['min'] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            info['max'] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            info['mean'] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Handle datetime columns
            valid_dates = df[col].dropna()
            if len(valid_dates) > 0:
                info['min'] = str(valid_dates.min())
                info['max'] = str(valid_dates.max())
        
        column_details.append(info)
    return column_details

# ---------- LLM Visualization Generator ----------
def get_visualizations(data_preview: str, columns: list, column_details: list, 
                       date_columns: list, time_stats: dict):
    
    date_info = ""
    if date_columns:
        date_info = f"""
DATE COLUMNS DETECTED: {', '.join(date_columns)}
Time Period Information: {json.dumps(time_stats, indent=2)}

IMPORTANT: For EVERY visualization, you MUST provide three time-period variations:
1. Monthly breakdown (using columns ending with '_month_name')
2. Quarterly breakdown (using columns ending with '_quarter')
3. Yearly breakdown (using columns ending with '_year')

Each visualization should have a 'time_aggregations' field containing monthly, quarterly, and yearly data.
"""
    
    prompt = f"""
You are a professional Data Visualization Analyst specializing in time-series analysis.
Analyze the dataset preview and suggest 5–8 meaningful visualizations.

INPUT:
- Columns: {', '.join(columns)}
- Column Details: {json.dumps(column_details, indent=2)}
- Data Preview: {data_preview[:2500]}
{date_info}

OUTPUT FORMAT (STRICT JSON):
{{
  "visualizations": [
    {{
      "title": "Descriptive chart title",
      "visual_type": "Bar Chart | Line Chart | Pie Chart | Scatter Plot | Heatmap | Box Plot | Area Chart",
      "chart_data": [
        {{"label": "Category A", "value": 1200, "percentage": 30}},
        {{"label": "Category B", "value": 2800, "percentage": 70}}
      ],
      "time_aggregations": {{
        "monthly": [
          {{"period": "January 2024", "value": 400, "trend": "up"}},
          {{"period": "February 2024", "value": 450, "trend": "up"}},
          {{"period": "March 2024", "value": 350, "trend": "down"}}
        ],
        "quarterly": [
          {{"period": "2024Q1", "value": 1200, "trend": "stable"}},
          {{"period": "2024Q2", "value": 1600, "trend": "up"}}
        ],
        "yearly": [
          {{"period": "2023", "value": 4500}},
          {{"period": "2024", "value": 5200}}
        ]
      }},
      "insight": "What this chart helps interpret, including time-based patterns",
      "example_finding": "Concise example finding with time context (≤25 words)",
      "date_column_used": "name of date column if applicable"
    }}
  ]
}}

Rules:
- If date columns exist, EVERY chart MUST include time_aggregations (monthly, quarterly, yearly)
- Generate realistic chart_data arrays (3–8 entries) based on actual data patterns
- For time-series data, prefer Line Charts or Area Charts
- Include trend indicators (up/down/stable) in time aggregations
- Use dataset context when naming chart axes or labels
- Monthly data should show month names with year (e.g., "January 2024")
- Quarterly data should use format "2024Q1", "2024Q2", etc.
- Yearly data should show just the year
- No extra text outside JSON
"""
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)["visualizations"]

# ---------- Route: Enhanced Visualization with Time Analysis ----------
@app.post("/report/visualizations")
async def generate_visualizations(file: UploadFile = File(...)):
    try:
        # Read file
        df = read_uploaded_file(file)
        
        # Detect and parse date columns
        date_columns = detect_date_columns(df)
        has_date_data = len(date_columns) > 0
        
        if has_date_data:
            df = parse_date_columns(df, date_columns)
            # Re-check date_columns after parsing (some might have failed)
            date_columns = [col for col in date_columns if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])]
            has_date_data = len(date_columns) > 0
            
        if has_date_data:
            time_stats = get_time_based_stats(df, date_columns)
        else:
            time_stats = {}
        
        # Get column details and preview
        preview = df.head(20).to_string(index=False)
        column_details = get_column_details(df)
        
        # Generate visualizations
        visualizations = get_visualizations(
            preview, 
            list(df.columns), 
            column_details,
            date_columns,
            time_stats
        )
        
        return JSONResponse({
            "status": "success",
            "data": {
                "visualizations": visualizations,
                "has_date_data": has_date_data,
                "date_columns": date_columns,
                "time_statistics": time_stats,
                "total_records": len(df)
            }
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# ---------- Optional: Route for Specific Time Aggregation ----------
@app.post("/report/time-aggregation")
async def get_time_aggregation(
    file: UploadFile = File(...),
    date_column: str = None,
    value_column: str = None,
    period: str = 'month'
):
    """
    Get specific time-based aggregation for a date and value column
    period: 'month', 'quarter', or 'year'
    """
    try:
        df = read_uploaded_file(file)
        date_columns = detect_date_columns(df)
        
        if not date_columns:
            return JSONResponse({
                "status": "error",
                "message": "No date columns found in the dataset"
            })
        
        df = parse_date_columns(df, date_columns)
        
        # Use first date column if not specified
        date_col = date_column if date_column else date_columns[0]
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        value_col = value_column if value_column else numeric_cols[0] if numeric_cols else None
        
        if not value_col:
            return JSONResponse({
                "status": "error",
                "message": "No numeric column found for aggregation"
            })
        
        # Aggregate data
        agg_data = aggregate_by_time_period(df, date_col, value_col, period)
        
        if agg_data is None:
            return JSONResponse({
                "status": "error",
                "message": f"Could not aggregate data by {period}"
            })
        
        return JSONResponse({
            "status": "success",
            "data": {
                "period": period,
                "date_column": date_col,
                "value_column": value_col,
                "aggregated_data": agg_data.to_dict(orient='records')
            }
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)