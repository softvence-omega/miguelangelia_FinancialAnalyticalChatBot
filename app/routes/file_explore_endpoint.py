import io
import asyncio
import pandas as pd
import aiohttp
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.services.file_explore import  generate_statistics, get_llm_report
from app.schemas.file_exploar_shcema import DataAnalysisResponse, ErrorResponse, Shape  

router = APIRouter()

import aiohttp
from fastapi import Query

# ---------- Route: Structured Report via Cloudinary URL ----------
@router.post("/file-explore", response_model=DataAnalysisResponse)
async def generate_report_url(file_url: str = Query(..., description="Cloudinary file URL")):
    try:
        # Download file from URL asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                if resp.status != 200:
                    raise ValueError(f"Failed to download file: status {resp.status}")
                content = await resp.read()

        # Determine file type
        if file_url.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file_url.lower().endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("File must be .csv or .xlsx")

        # Generate preview
        preview = df.head(20).to_string(index=False)

        # LLM report
        llm_report = await get_llm_report(preview, list(df.columns))

        # Shape & statistics
        shape = Shape(rows=int(df.shape[0]), columns=int(df.shape[1]))
        statistics = await asyncio.to_thread(generate_statistics, df)

        return DataAnalysisResponse(
            column_descriptions=llm_report.column_descriptions,
            shape=shape,
            statistics=statistics
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                message=str(e),
                error_type=type(e).__name__
            ).dict()
        )
