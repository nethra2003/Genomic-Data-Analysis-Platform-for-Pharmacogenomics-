from sqlalchemy import Column, BigInteger, String, DateTime, Text, ForeignKey, JSON, Date, Boolean, TIMESTAMP
from sqlalchemy.orm import relationship
from db_config import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(BigInteger, primary_key=True)
    user_code = Column(String(64), unique=True, nullable=False)
    full_name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    role = Column(String(20), nullable=False, default="patient")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class Patient(Base):
    __tablename__ = "patients"

    id = Column(BigInteger, primary_key=True)
    patient_code = Column(String(64), unique=True, nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    full_name = Column(String(200))  # ✅ New field for convenience
    date_of_birth = Column(Date)
    sex = Column(String(12))
    email = Column(String(200), unique=True)
    clinician_name = Column(String(150))
    consent_given = Column(Boolean, default=False)
    password_hash = Column(String(200))  # ✅ For authentication
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow)
    reports = relationship("Report", back_populates="patient")

    
class Report(Base):
    __tablename__ = "reports"
    id = Column(BigInteger, primary_key=True)
    patient_id = Column(BigInteger, ForeignKey("patients.id"))
    report_title = Column(String(200))
    pdf_path = Column(Text)
    report_json = Column(JSON)
    generated_at = Column(TIMESTAMP)
    patient = relationship("Patient", back_populates="reports")
