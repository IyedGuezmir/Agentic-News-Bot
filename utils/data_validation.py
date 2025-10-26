from pydantic import BaseModel, Field

from pydantic import BaseModel, Field

class NewsItem(BaseModel):
    title: str
    text: str
    subject: str
    date: str
    label: int = Field(default=0)  # dummy for testing with ML model
    
class VerificationResult(BaseModel):
    verdict: int  # 1 for True, 0 for False
    url: str = ""  # optional supporting link