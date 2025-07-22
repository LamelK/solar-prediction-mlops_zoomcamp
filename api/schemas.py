from pydantic import BaseModel, Field
from pydantic import ConfigDict


class RawInputData(BaseModel):
    """
    Pydantic model for raw input data expected by the prediction API.
    Each field includes an example for documentation and validation.
    """

    UNIXTime: int = Field(..., json_schema_extra={"example": 1481298952})
    Data: str = Field(..., json_schema_extra={"example": "12/9/2016 12:00:00 AM"})
    Time: str = Field(..., json_schema_extra={"example": "05:55:52"})
    Temperature: int = Field(..., json_schema_extra={"example": 48})
    Pressure: float = Field(..., json_schema_extra={"example": 30.38})
    Humidity: int = Field(..., json_schema_extra={"example": 89})
    WindDirection_Degrees: float = Field(..., json_schema_extra={"example": 247.97})
    Speed: float = Field(..., json_schema_extra={"example": 7.87})
    TimeSunRise: str = Field(..., json_schema_extra={"example": "06:46:00"})
    TimeSunSet: str = Field(..., json_schema_extra={"example": "17:44:00"})
    datetime: str = Field(..., json_schema_extra={"example": "2016-12-09 15:55:52"})

    model_config = ConfigDict(validate_assignment=True)
