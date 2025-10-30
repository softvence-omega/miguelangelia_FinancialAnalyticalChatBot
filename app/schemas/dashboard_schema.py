from typing import List, Dict, Any, Optional
from pydantic import BaseModel



# ---------- Updated Pydantic Models ----------
class TimeAggregation(BaseModel):
    period: str
    value: float
    trend: Optional[str] = None


class TimeAggregations(BaseModel):
    monthly: Optional[List[TimeAggregation]] = None
    quarterly: Optional[List[TimeAggregation]] = None
    yearly: Optional[List[TimeAggregation]] = None


class Visualization(BaseModel):
    title: str
    visual_type: str
    chart_data: List[Dict[str, Any]]
    time_aggregations: Optional[TimeAggregations] = None
    insight: str
    example_finding: str
    date_column_used: Optional[str] = None


class DataReport(BaseModel):
    metrics: Dict[str, Any]
    visualizations: List[Visualization]
    overall_summary: str

