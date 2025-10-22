from typing import Dict, List, Optional
from pydantic import BaseModel, Field



# ---------- Pydantic Models ----------
class ColumnDescription(BaseModel):
    column: str
    description: str

class Shape(BaseModel):
    rows: int
    columns: int

class ColumnStatistics(BaseModel):
    count: int
    unique: Optional[int] = None
    top: Optional[str] = None
    freq: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    percentile_25: Optional[float] = Field(None, alias="25%")
    percentile_50: Optional[float] = Field(None, alias="50%")
    percentile_75: Optional[float] = Field(None, alias="75%")
    max: Optional[float] = None

    class Config:
        populate_by_name = True
        validate_by_name = True

class DataAnalysisResponse(BaseModel):
    column_descriptions: List[ColumnDescription]
    shape: Shape
    statistics: Dict[str, ColumnStatistics]

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    error_type: str

class LLMReport(BaseModel):
    column_descriptions: List[ColumnDescription]
