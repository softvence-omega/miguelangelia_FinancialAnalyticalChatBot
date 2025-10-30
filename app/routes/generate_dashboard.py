from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.dashboard_service import read_file_from_url, get_column_details, detect_date_columns, get_llm_report
router = APIRouter()

@router.post("/generate-dashboard")
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
