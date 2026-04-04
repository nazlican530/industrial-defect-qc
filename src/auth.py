from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    status,
    UploadFile,
    File,
    Form,
)
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from src.database import users_collection


SECRET_KEY = "change-this-secret-key-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 gün

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

router = APIRouter(prefix="/auth", tags=["auth"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def to_public_upload_path(path_value):
    if not path_value:
        return None

    path_str = str(path_value).replace("\\", "/")

    if "/uploads/" in path_str:
        return path_str[path_str.index("/uploads/"):]

    if path_str.startswith("uploads/"):
        return f"/{path_str}"

    if path_str.startswith("/uploads/"):
        return path_str

    filename = Path(path_str).name
    return f"/uploads/{filename}"


def sanitize_user(user: dict):
    if not user:
        return None

    user["_id"] = str(user["_id"])
    user.pop("hashed_password", None)

    # frontend için düzgün dönsün
    user["profile_image"] = to_public_upload_path(user.get("profile_image"))
    return user


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = users_collection.find_one({"email": email})
    if not user:
        raise credentials_exception

    return sanitize_user(user)


@router.post("/register")
def register(data: RegisterRequest):
    existing_user = users_collection.find_one({"email": data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = {
        "name": data.name.strip(),
        "email": data.email.strip().lower(),
        "hashed_password": hash_password(data.password),
        "profile_image": None,
        "created_at": datetime.utcnow(),
    }

    users_collection.insert_one(new_user)

    return {"message": "User registered successfully"}


@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest):
    user = users_collection.find_one({"email": data.email.strip().lower()})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token({"sub": user["email"]})

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.put("/me")
async def update_me(
    name: str = Form(...),
    profile_image: UploadFile | None = File(None),
    current_user: dict = Depends(get_current_user),
):
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    update_fields = {
        "name": clean_name,
        "updated_at": datetime.utcnow(),
    }

    if profile_image is not None:
        content = await profile_image.read()

        if not content:
            raise HTTPException(status_code=400, detail="Empty image file")

        original_name = profile_image.filename or "profile.jpg"
        ext = Path(original_name).suffix.lower()

        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(
                status_code=400,
                detail="Only .jpg, .jpeg, .png, .webp files are allowed",
            )

        filename = f"profile_{uuid.uuid4().hex}{ext}"
        save_path = UPLOAD_DIR / filename
        save_path.write_bytes(content)

        update_fields["profile_image"] = f"/uploads/{filename}"

    users_collection.update_one(
        {"email": current_user["email"]},
        {"$set": update_fields},
    )

    updated_user = users_collection.find_one({"email": current_user["email"]})
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    return sanitize_user(updated_user)