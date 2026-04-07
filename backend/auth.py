# from sqlalchemy.orm import Session
# from pydantic import BaseModel, EmailStr
# from passlib.context import CryptContext
# from db_config import SessionLocal
# from models import Patient
# from jose import jwt
# from datetime import datetime, timedelta
# from fastapi import Depends, HTTPException, status, APIRouter
# from jose import JWTError, jwt
# import os
# from fastapi.security import OAuth2PasswordBearer
# from db_config import SessionLocal  # ✅ ensure this import exists

# # Dependency to get DB session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# router = APIRouter()
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# class SignupRequest(BaseModel):
#     fullName: str
#     userId: str
#     email: EmailStr
#     password: str



# @router.post("/signup")
# def signup(data: dict, db: Session = Depends(get_db)):
#     try:
#         full_name = data.get("full_name")
#         user_id = data.get("user_id")
#         email = data.get("email")
#         password = data.get("password")

#         if not all([full_name, user_id, email, password]):
#             raise HTTPException(status_code=400, detail="All fields are required")

#         # Check if email or ID already exists
#         existing_user = (
#             db.query(Patient)
#             .filter((Patient.email == email) | (Patient.patient_code == user_id))
#             .first()
#         )
#         if existing_user:
#             raise HTTPException(status_code=400, detail="Email or ID already exists")

#         hashed_pw = pwd_context.hash(password)

#         new_user = Patient(
#             full_name=full_name,
#             patient_code=user_id,
#             email=email,
#             password_hash=hashed_pw,
#         )

#         db.add(new_user)
#         db.commit()
#         db.refresh(new_user)

#         return {
#             "message": "Signup successful",
#             "user": {
#                 "full_name": new_user.full_name,
#                 "patient_code": new_user.patient_code,
#                 "email": new_user.email,
#             },
#         }

#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))




# # Constants
# SECRET_KEY = "mysecretkeyforjwt123"  # You can later move this to .env
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 60

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# router = APIRouter()

# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)

# def create_access_token(data: dict, expires_delta: timedelta | None = None):
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# router = APIRouter()

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# SECRET_KEY = "your_secret_key_here"
# ALGORITHM = "HS256"


# @router.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     try:
#         email = data.get("email")
#         password = data.get("password")

#         if not email or not password:
#             raise HTTPException(status_code=400, detail="Email and password are required")

#         user = db.query(Patient).filter(Patient.email == email).first()
#         if not user:
#             raise HTTPException(status_code=400, detail="Invalid email")

#         if not pwd_context.verify(password, user.password_hash):
#             raise HTTPException(status_code=400, detail="Invalid password")

#         # ✅ Create JWT token
#         token_data = {"sub": user.email}
#         access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

#         # ✅ Clean return object — frontend uses this for localStorage
#         return {
#             "access_token": access_token,
#             "token_type": "bearer",
#             "user": {
#                 "patient_code": user.patient_code,
#                 "full_name": user.full_name or "User",
#                 "email": user.email,
#             },
#         }

#     except Exception as e:
#         print("Login Error:", e)
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         db.close()



# # 🧩 FETCH CURRENT USER (FOR AUTO-FILL IN FRONTEND)
# @router.get("/me")
# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(SessionLocal)):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         user_email = payload.get("sub")
#         if user_email is None:
#             raise HTTPException(status_code=401, detail="Invalid authentication token")

#         user = db.query(Patient).filter(Patient.email == user_email).first()
#         if not user:
#             raise HTTPException(status_code=404, detail="User not found")

#         return {
#     "patient_code": user.patient_code,
#     "full_name": user.full_name if hasattr(user, "full_name") else None,
#     "email": user.email,
# }

#     except JWTError:
#         raise HTTPException(status_code=401, detail="Could not validate token")

from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, APIRouter, status
from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer

from db_config import SessionLocal
from models import Patient

# ===================================================
# CONFIG
# ===================================================
SECRET_KEY = "mysecretkeyforjwt123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

router = APIRouter()

# ===================================================
# DB Dependency
# ===================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===================================================
# SIGNUP
# ===================================================
@router.post("/signup")
def signup(data: dict, db: Session = Depends(get_db)):
    """
    Signup endpoint — matches frontend fields exactly
    """
    full_name = data.get("fullName")   # ✅ matches frontend
    user_id   = data.get("userId")     # ✅ matches frontend
    email     = data.get("email")
    password  = data.get("password")

    if not all([full_name, user_id, email, password]):
        raise HTTPException(status_code=400, detail="All fields are required")

    # Check if email or user ID already exists
    existing_user = (
        db.query(Patient)
        .filter(
            (Patient.email == email) |
            (Patient.patient_code == user_id)
        )
        .first()
    )
    if existing_user:
        raise HTTPException(status_code=400, detail="Email or User ID already exists")

    hashed_pw = pwd_context.hash(password)

    new_user = Patient(
        full_name=full_name,
        patient_code=user_id,
        email=email,
        password_hash=hashed_pw,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "Signup successful",
        "user": {
            "full_name": new_user.full_name,
            "patient_code": new_user.patient_code,
            "email": new_user.email,
        },
    }

# ===================================================
# LOGIN
# ===================================================
@router.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    user = db.query(Patient).filter(Patient.email == email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email")

    if not pwd_context.verify(password, user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid password")

    token_data = {"sub": user.email}
    access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "patient_code": user.patient_code,
            "full_name": user.full_name,
            "email": user.email,
        },
    }

# ===================================================
# GET CURRENT USER
# ===================================================
@router.get("/me")
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = db.query(Patient).filter(Patient.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "patient_code": user.patient_code,
            "full_name": user.full_name,
            "email": user.email,
        }

    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate token")
