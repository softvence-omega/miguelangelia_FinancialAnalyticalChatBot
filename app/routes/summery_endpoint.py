from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse 
from app.services.summery_service import read_csv_async, read_excel_async, summarize_dataset

router = APIRouter()


@router.post("/summary")
async def analyze_file(file: UploadFile = File(...)):
    try:
        content = await file.read()

        if file.filename.endswith(".csv"):
            df = await read_csv_async(content)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = await read_excel_async(content)
        else:
            return JSONResponse({"status": "error", "message": "Unsupported file type"}, status_code=400)

        summary = await summarize_dataset(df)
        return {"summary": summary}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)