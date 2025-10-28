from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse 
from app.services.summery_service import read_csv_async, read_excel_async, summarize_dataset
import httpx

router = APIRouter()


@router.post("/summary")
async def analyze_file_from_url(file_url: str = Query(..., description="Cloudinary file URL")):
    try:
        async with httpx.AsyncClient() as client_http:
            resp = await client_http.get(file_url)
            if resp.status_code != 200:
                return JSONResponse(
                    {"status": "error", "message": f"Failed to fetch file: HTTP {resp.status_code}"},
                    status_code=resp.status_code
                )
            content = resp.content

        # Determine file type from URL
        if file_url.endswith(".csv"):
            df = await read_csv_async(content)
        elif file_url.endswith((".xlsx", ".xls")):
            df = await read_excel_async(content)
        else:
            return JSONResponse({"status": "error", "message": "Unsupported file type"}, status_code=400)

        summary = await summarize_dataset(df)
        return {"summary": summary}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)