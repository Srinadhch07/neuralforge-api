from pydantic import BaseModel
from typing import Literal
class TrainModel(BaseModel):
    epochs: int
    model : Literal["small", "medium" ,"large"] = "small"
