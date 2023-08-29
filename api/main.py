from pydantic import BaseModel, Field
from ms import app
from ms.functions import get_model_response

# Input for data validation
class Input(BaseModel):
    to_user_distance: float = Field(..., gt=0)
    to_user_elevation: float
    total_earning: float = Field(..., gt=0)
    hour: float = Field(..., gt=0)
    taken_percentage: float = Field(..., gt=0)

    class Config:
        schema_extra = {
            "to_user_distance": 2.478101,
            "to_user_elevation": -72.719360,
            "total_earning": 4200,
            "hour": 20,
            "taken_percentage": 90.714286,
        }

# Ouput for data validation
class Output(BaseModel):
    label: str
    prediction: int

@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response
