from pydantic import BaseModel, EmailStr, Field, constr, validator
from datetime import datetime, time
from typing import Optional, List, Literal

class TaskPredict(BaseModel):
    path: str
    model: str