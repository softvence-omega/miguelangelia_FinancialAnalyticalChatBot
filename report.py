from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import io
import json
import requests
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from app.core.config import settings
# -------------------- Setup -------------------- #
load_dotenv()
app = FastAPI(title="Dataset Analysis API")
router = APIRouter(prefix="/api", tags=["Report"])

client = AsyncOpenAI(api_key=settings.openai_api_key)


# -------------------- Schemas -------------------- #
class FileInput(BaseModel):
    file_url: str

class VariableInfo(BaseModel):
    variable: str
    types: str
    missing_count: int
    missing_percent: float
    unique_count: int
    unique_percent: float

class SkewnessInfo(BaseModel):
    type: str
    skewness: float
    kurtosis: float

class ChartDataPoint(BaseModel):
    label: str
    value: float

class VisualizationData(BaseModel):
    title: str
    visual_type: str
    chart_data: List[ChartDataPoint]

class AnalysisReport(BaseModel):
    about_report: str
    dataset_info: str
    variables: List[VariableInfo]
    skewness_info: List[SkewnessInfo]
    visualizations: List[VisualizationData]


# -------------------- File Loader -------------------- #
async def load_from_url(file_url: str) -> pd.DataFrame:
    loop = asyncio.get_event_loop()

    def _load():
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            content = io.BytesIO(response.content)

            if file_url.endswith(".csv"):
                return pd.read_csv(content)
            elif file_url.endswith((".xls", ".xlsx")):
                return pd.read_excel(content)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")

    return await loop.run_in_executor(None, _load)


# -------------------- Dataset Analysis -------------------- #
async def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()

    def _analyze():
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing': df.isnull().sum().to_dict(),
            'missing_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
            'unique_counts': df.nunique().to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['skewness'] = df[numeric_cols].skew().to_dict()
            analysis['kurtosis'] = df[numeric_cols].kurtosis().to_dict()

        return analysis

    return await loop.run_in_executor(None, _analyze)


# -------------------- Visualization Generation -------------------- #
async def generate_visualizations(df: pd.DataFrame) -> List[VisualizationData]:
    loop = asyncio.get_event_loop()

    def _generate():
        visualizations = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        for col in df.select_dtypes(include=['object']).columns:
            try:
                result = pd.to_datetime(df[col].head(10), errors='coerce', format='mixed')
                if result.notna().sum() >= 5:
                    date_cols.append(col)
            except Exception:
                pass

        for idx, col in enumerate(numeric_cols[:3]):
            data = df[col].dropna()
            if len(data) == 0:
                continue

            unique_count = data.nunique()
            data_length = len(data)

            if date_cols and idx == 0:
                date_col = date_cols[0]
                temp_df = df[[date_col, col]].dropna().sort_values(date_col)
                if len(temp_df) > 50:
                    step = len(temp_df) // 50
                    temp_df = temp_df.iloc[::step]

                chart_data = [
                    ChartDataPoint(label=str(row[date_col]).split("T")[0], value=float(row[col]))
                    for _, row in temp_df.iterrows()
                ]
                visualizations.append(VisualizationData(title=f"{col} Over Time", visual_type="Line Chart", chart_data=chart_data))

            elif unique_count <= 10 and unique_count < data_length * 0.1:
                value_counts = data.value_counts().sort_index().head(10)
                chart_data = [ChartDataPoint(label=str(l), value=float(v)) for l, v in value_counts.items()]
                visualizations.append(VisualizationData(title=f"Distribution of {col}", visual_type="Bar Chart", chart_data=chart_data))
            else:
                hist, bins = np.histogram(data, bins=15)
                chart_data = [ChartDataPoint(label=f"{bins[i]:.2f}-{bins[i+1]:.2f}", value=float(hist[i])) for i in range(len(hist))]
                visualizations.append(VisualizationData(title=f"Distribution of {col}", visual_type="Histogram", chart_data=chart_data))

        if len(visualizations) < 3:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if len(visualizations) >= 3:
                    break
                value_counts = df[col].value_counts().head(10)
                if len(value_counts) > 1:
                    chart_data = [ChartDataPoint(label=str(l), value=float(v)) for l, v in value_counts.items()]
                    visualizations.append(VisualizationData(title=f"Top Categories in {col}", visual_type="Bar Chart", chart_data=chart_data))

        return visualizations

    return await loop.run_in_executor(None, _generate)


# -------------------- AI Insights -------------------- #
async def get_ai_insights(analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, str]:
    context = f"""
    Dataset Analysis Summary:
    - Shape: {analysis['shape'][0]} rows, {analysis['shape'][1]} columns
    - Columns: {', '.join(analysis['columns'])}
    - Data Types: {json.dumps(analysis['dtypes'], indent=2)}
    - Missing Values: {json.dumps(analysis['missing'], indent=2)}
    Sample Data (first 5 rows):
    {df.head().to_string()}
    Numeric Summary Statistics:
    {json.dumps(analysis.get('numeric_summary', {}), indent=2)}
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst expert. Provide concise dataset insights."},
                {"role": "user", "content": f"{context}\n\nProvide 1) about_report, 2) dataset_info in JSON format."}
            ],
            temperature=0.7,
            max_tokens=500
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception:
        return {
            "about_report": "Comprehensive dataset analysis including structure, quality, and statistics.",
            "dataset_info": f"Dataset has {analysis['shape'][0]} rows and {analysis['shape'][1]} columns."
        }


# -------------------- Helpers -------------------- #
async def prepare_variables(df: pd.DataFrame, analysis: Dict[str, Any]) -> List[VariableInfo]:
    loop = asyncio.get_event_loop()

    def _prepare():
        vars_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype.startswith('int') or dtype.startswith('float'):
                var_type = 'numeric'
            elif dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                var_type = 'factor' if unique_ratio < 0.5 else 'character'
            elif dtype == 'bool':
                var_type = 'logical'
            else:
                var_type = dtype

            miss = int(analysis['missing'][col])
            miss_pct = float(analysis['missing_percent'][col])
            uniq = int(analysis['unique_counts'][col])
            uniq_pct = float(uniq / len(df) * 100) if len(df) > 0 else 0

            vars_info.append(VariableInfo(
                variable=col,
                types=var_type,
                missing_count=miss,
                missing_percent=round(miss_pct, 1),
                unique_count=uniq,
                unique_percent=round(uniq_pct, 2)
            ))
        return vars_info

    return await loop.run_in_executor(None, _prepare)


async def prepare_skewness(df: pd.DataFrame, analysis: Dict[str, Any]) -> List[SkewnessInfo]:
    loop = asyncio.get_event_loop()

    def _prepare():
        skews = []
        if 'skewness' not in analysis:
            return skews
        skews.append(SkewnessInfo(
            type="original",
            skewness=round(float(np.mean(list(analysis['skewness'].values()))), 4),
            kurtosis=round(float(np.mean(list(analysis['kurtosis'].values()))), 4)
        ))

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return skews

        positive_data = df[numeric_cols].apply(lambda x: x[x > 0])
        if not positive_data.empty:
            log_skew = positive_data.apply(lambda x: np.log(x).skew()).mean()
            log_kurt = positive_data.apply(lambda x: np.log(x).kurtosis()).mean()
            skews.append(SkewnessInfo(type="log transform", skewness=round(float(log_skew), 4), kurtosis=round(float(log_kurt), 4)))

        sqrt_data = df[numeric_cols].apply(lambda x: np.sqrt(x[x >= 0]))
        if not sqrt_data.empty:
            sqrt_skew = sqrt_data.apply(lambda x: x.skew()).mean()
            sqrt_kurt = sqrt_data.apply(lambda x: x.kurtosis()).mean()
            skews.append(SkewnessInfo(type="sqrt transform", skewness=round(float(sqrt_skew), 4), kurtosis=round(float(sqrt_kurt), 4)))

        return skews

    return await loop.run_in_executor(None, _prepare)


# -------------------- API Endpoint -------------------- #
@router.post("/generate-report", response_model=AnalysisReport)
async def analyze_file(input_data: FileInput):
    try:
        df = await load_from_url(input_data.file_url)
        analysis = await analyze_dataset(df)

        insights_task = get_ai_insights(analysis, df)
        variables_task = prepare_variables(df, analysis)
        skew_task = prepare_skewness(df, analysis)
        visuals_task = generate_visualizations(df)

        insights, variables, skew_info, visuals = await asyncio.gather(
            insights_task, variables_task, skew_task, visuals_task
        )

        return AnalysisReport(
            about_report=insights['about_report'],
            dataset_info=insights['dataset_info'],
            variables=variables,
            skewness_info=skew_info,
            visualizations=visuals
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")


# -------------------- Mount Router -------------------- #
app.include_router(router)
