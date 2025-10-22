from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import asyncio
from app.services.file_explore import read_uploaded_file, generate_statistics, get_llm_report
from app.schemas.file_exploar_shcema import DataAnalysisResponse, ErrorResponse, Shape  

router = APIRouter()

@router.post("/file-explore")
async def generate_report(file: UploadFile = File(...)):
    try:
        # In the route, wrap the preview generation too:
        df = await read_uploaded_file(file)
        preview = await asyncio.to_thread(lambda: df.head(20).to_string(index=False))
        llm_report = await get_llm_report(preview, list(df.columns))
        shape = Shape(rows=int(df.shape[0]), columns=int(df.shape[1]))
        statistics = await asyncio.to_thread(generate_statistics, df)
        response = DataAnalysisResponse(
            column_descriptions=llm_report.column_descriptions,
            shape=shape,
            statistics=statistics
        )
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message=str(e),
                error_type=type(e).__name__
            ).dict()
        )

   