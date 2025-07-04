from pydantic import BaseModel

class TextsRequest(BaseModel):
    texts: list[str]