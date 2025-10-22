from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_explore import explore_df
import tempfile
import json
import os

router = APIRouter()

@router.post("/file-explore")
async def explore_df_endpoint(file: UploadFile = File(...)):
    try:
        response_json = await explore_df(file)

        # Convert JSON string to dict before returning
        print(response_json)
        return json.loads(response_json)

    except ValueError as ve:
        # Specific error for invalid file format or bad data
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   