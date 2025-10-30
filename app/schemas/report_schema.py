from typing import Any, Dict, List
from pydantic import BaseModel


# ---------- Response Models ----------
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
