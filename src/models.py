from pydantic import BaseModel
from enums import ModelType

class Type(BaseModel):
    model_type: ModelType