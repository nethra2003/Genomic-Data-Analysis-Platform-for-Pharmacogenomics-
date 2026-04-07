from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Any, Dict

class SignupIn(BaseModel):
    full_name: str
    user_code: str
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class PatientIn(BaseModel):
    patient_code: str
    first_name: str
    last_name: str
    sex: Optional[str] = "unknown"
    email: Optional[EmailStr] = None

class SampleIn(BaseModel):
    patient_code: str
    sample_id: str
    genome_build: Optional[str] = "GRCh37"
    notes: Optional[str] = None

class RunAnalysisIn(BaseModel):
    sample_id: str
    pipeline_name: str = "pgx_pipeline"
    pipeline_version: str = "1.0.0"
    parameters_json: Dict[str, Any] = Field(default_factory=dict)

class ReportOut(BaseModel):
    report_id: int
    pdf_path: Optional[str] = None
