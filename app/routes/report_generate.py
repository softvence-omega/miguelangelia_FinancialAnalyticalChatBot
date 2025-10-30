from fastapi import APIRouter, HTTPException, Query
from app.services.report_service import load_from_url, analyze_dataset, get_ai_insights, prepare_variables, prepare_skewness, generate_visualizations, sanitize_json
from app.schemas.report_schema import AnalysisReport, FileInput
import asyncio


router = APIRouter()

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

        result = AnalysisReport(
            about_report=insights['about_report'],
            dataset_info=insights['dataset_info'],
            variables=variables,
            skewness_info=skew_info,
            visualizations=visuals
        )
        return sanitize_json(result.dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")