from pydantic import BaseModel
from typing import Literal, Optional

class TrainModel(BaseModel):
    model_name: Optional[str] = None
    epochs: int
    model : Optional[Literal["small", "medium" ,"large"]] = "small"
