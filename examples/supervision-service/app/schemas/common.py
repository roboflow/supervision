from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(examples=["ok"])


class ApiInfo(BaseModel):
    """Root endpoint metadata."""

    message: str
    docs: str
    health: str
