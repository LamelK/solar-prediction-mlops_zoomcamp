from pydantic import BaseModel, Field
from typing import Optional

class RawInputData(BaseModel):
    UNIXTime: int = Field(..., example=1481298952)
    Data: str = Field(..., example="12/9/2016 12:00:00 AM")
    Time: str = Field(..., example="05:55:52")
    Temperature: float = Field(..., example=48)
    Pressure: float = Field(..., example=30.38)
    Humidity: int = Field(..., example=89)
    WindDirection_Degrees: float = Field(..., example=247.97)
    Speed: float = Field(..., example=7.87)
    TimeSunRise: str = Field(..., example="06:46:00")
    TimeSunSet: str = Field(..., example="17:44:00")
    datetime: str = Field(..., example="2016-12-09 15:55:52")

    class Config:
        validate_by_name = True
