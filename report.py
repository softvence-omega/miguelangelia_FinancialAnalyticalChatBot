from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import matplotlib.pyplot as plt
import base64
from pandas_profiling import ProfileReport
from openai import OpenAI
import os
from typing import Dict, List, Any
import numpy as np

app = FastAPI(title="Dataset Analyzer with OpenAI Insights")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Set your key in env

def encode_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def analyze_numeric(col: pd.Series, name: str) -> Dict[str, Any]:
    if col.nunique() == 0:
        return {"type": "empty", "plots": []}
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(col.dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title(f'{name} - Histogram')
    
    # Boxplot
    axes[1].boxplot(col.dropna(), vert=False)
    axes[1].set_title(f'{name} - Boxplot')
    
    # QQ Plot
    from scipy import stats
    stats.probplot(col.dropna(), dist="norm", plot=axes[2])
    axes[2].set_title(f'{name} - QQ Plot')
    
    img = encode_plot(fig)
    return {
        "type": "numeric",
        "stats": {
            "skewness": float(col.skew()) if not col.empty else None,
            "kurtosis": float(col.kurtosis()) if not col.empty else None,
            "original": col.describe().to_dict()
        },
        "plots": [f"data:image/png;base64,{img}"]
    }

def generate_openai_summary(df: pd.DataFrame, profile: ProfileReport) -> str:
    profile_text = profile.to_json()
    # Truncate if too long
    if len(profile_text) > 10000:
        profile_text = profile_text[:10000] + "..."

    prompt = f"""
You are a senior data analyst. Analyze this dataset summary and write a concise, professional 'About Report' paragraph (3-5 sentences) like lorem ipsum but meaningful.

Dataset has {len(df)} rows and {len(df.columns)} columns.
Variables: {', '.join(df.columns)}

Summary stats (missing values, types, etc.):
{profile_text}

Focus on: data quality, key patterns, potential issues, and suitability for analysis.
Use formal language. Do NOT mention technical tools.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    if file.content_type not in [
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ]:
        raise HTTPException(400, detail="Invalid file type. Upload CSV or Excel.")

    content = await file.read()
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    else:
        df = pd.read_excel(io.BytesIO(content))

    if df.empty:
        raise HTTPException(400, detail="Empty dataset.")

    # Generate profile
    profile = ProfileReport(df, minimal=True, title="Dataset Report", explorative=True)
    
    # OpenAI Summary
    about_report = generate_openai_summary(df, profile)

    # Variable Info
    var_info = []
    for col in df.columns:
        col_data = df[col]
        missing_count = col_data.isna().sum()
        missing_percent = round(missing_count / len(df) * 100, 2)
        unique_count = col_data.nunique()
        unique_percent = round(unique_count / len(df) * 100, 2) if len(df) > 0 else 0

        var_type = "unknown"
        if pd.api.types.is_numeric_dtype(col_data):
            var_type = "numeric"
        elif pd.api.types.is_bool_dtype(col_data):
            var_type = "logical"
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'category':
            var_type = "factor"
        elif pd.api.types.is_string_dtype(col_data):
            var_type = "character"

        var_info.append({
            "variable": col,
            "type": var_type,
            "missing_count": int(missing_count),
            "missing_percent": missing_percent,
            "unique_count": int(unique_count),
            "unique_percent": unique_percent
        })

    # Distribution Stats for Numeric Columns
    dist_stats = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        dist_stats.append({
            "variable": col,
            "skewness": round(skewness, 4) if not pd.isna(skewness) else None,
            "kurtosis": round(kurtosis, 4) if not pd.isna(kurtosis) else None
        })

    # Generate Visuals (only for numeric, limit to 3)
    visuals = []
    for i, col in enumerate(numeric_cols[:3]):
        result = analyze_numeric(df[col], col)
        if result["type"] == "numeric":
            visuals.extend(result["plots"])

    # Final Report
    report = {
        "about_report": about_report,
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "size_mb": round(len(content) / (1024*1024), 2)
        },
        "information_of_variables": var_info,
        "distribution_stats": dist_stats,
        "visuals": visuals[:3]  # Limit to 3 images
    }

    return JSONResponse(content=report)