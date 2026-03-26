from pydantic import BaseModel, EmailStr, Field, constr, validator
from datetime import datetime, time
from typing import Literal,Optional

class ModelType(BaseModel):
    name: Optional[Literal["small", "medium", "large"]] = None
