import pandas as pd
import json
import numpy as np
import os
from fastapi import UploadFile, HTTPException

async def explore_df(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower()
    
    try:
        if ext == ".csv":
            df = pd.read_csv(file.file)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file.file)
        else:
            raise ValueError("Unsupported file format. Only CSV/XLSX allowed.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Prepare data summary
    df_info = {
        "columns": df.columns.tolist(),
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "sample_data": df.head(3).to_dict(orient="records"),
        "statistics": df.describe(include="all").replace({np.nan: None}).to_dict()
    }

    # Convert to valid JSON (allow_nan=False ensures strict compliance)
    return json.dumps(df_info, indent=4, allow_nan=False)


