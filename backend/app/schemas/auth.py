from pydantic import BaseModel, EmailStr, Field
from uuid import UUID

class UserRegisterRequest(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=8)

class UserLoginRequest(BaseModel):
    email:    EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      UUID
    email:        str
    role:         str

class UserResponse(BaseModel):
    id:        UUID
    email:     str
    role:      str
    is_active: bool
    class Config: from_attributes = True