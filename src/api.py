import io
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from bson import ObjectId
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms

from src.auth import router as auth_router, get_current_user
from src.database import predictions_collection, users_collection
from src.gradcam_utils import generate_gradcam
from src.llm_agent import generate_report as llm_generate_report


MODEL_PATH = Path("outputs/models/best_model.pt")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# MODEL LOAD
# =========================
def load_model_and_tf():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    arch = ckpt.get("arch", "efficientnet_b0")

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = ckpt["model_state"]
    fixed_state_dict = {}

    for k, v in state_dict.items():
        new_key = k.replace("classifier.1.1", "classifier.1")
        fixed_state_dict[new_key] = v

    model.load_state_dict(fixed_state_dict, strict=True)
    model.to(device)
    model.eval()

    mean = tuple(ckpt.get("imagenet_mean", (0.485, 0.456, 0.406)))
    std = tuple(ckpt.get("imagenet_std", (0.229, 0.224, 0.225)))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return model, tf, idx_to_class, num_classes, arch


model, tf, idx_to_class, num_classes, ARCH = load_model_and_tf()


# =========================
# HELPERS
# =========================
def severity_score(conf: float) -> str:
    """
    Genel kullanım için daha dengeli threshold:
    0.85+  -> High
    0.55+  -> Medium
    altı   -> Low
    """
    if conf >= 0.85:
        return "High"
    elif conf >= 0.55:
        return "Medium"
    return "Low"


def quality_decision(defect_class: str, severity: str, confidence: float) -> str:
    """
    Genel karar mantığı:
    - Kritik defectlerde daha temkinli davran
    - Düşük güvenli tahminleri doğrudan ACCEPT etme
    - Belirsiz durumları REWORK'e yönlendir
    """

    critical_rework_classes = {"scratches", "pitted_surface"}
    reject_prone_classes = {"scratches", "pitted_surface", "rolled-in_scale"}

    # Çok güvenli ve kritik kusur -> REJECT
    if defect_class in reject_prone_classes and confidence >= 0.85:
        return "REJECT"

    # Kritik kusurlar -> en az REWORK
    if defect_class in critical_rework_classes:
        return "REWORK"

    # Genel olarak düşük güvenli prediction'ı ACCEPT etme
    if confidence < 0.65:
        return "REWORK"

    # Orta güven seviyesinde tekrar kontrol mantığı
    if severity == "Medium":
        return "REWORK"

    # Çok yüksek confidence varsa ve kritik değilse reject yerine kabul edilebilir
    if severity == "High":
        return "ACCEPT"

    return "ACCEPT"


def to_public_upload_path(path_value):
    """
    uploads/abc.jpg  -> /uploads/abc.jpg
    /uploads/abc.jpg -> /uploads/abc.jpg
    full path gelirse yine /uploads/... olarak normalize etmeye çalışır
    """
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


def format_prediction(doc):
    return {
        "_id": str(doc["_id"]),
        "defect": doc.get("defect"),
        "confidence": doc.get("confidence"),
        "severity": doc.get("severity"),
        "decision": doc.get("decision"),
        "image_path": to_public_upload_path(doc.get("image")),
        "gradcam_path": to_public_upload_path(doc.get("gradcam")),
        "report": doc.get("report"),
        "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None,
        "owner_name": doc.get("owner_name"),
        "owner_photo": to_public_upload_path(doc.get("owner_photo")),
        "user_email": doc.get("user_email"),
    }


def find_last_conv_layer(module: nn.Module):
    """
    Model içindeki son Conv2d katmanını bulur.
    """
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def get_target_layer(model: nn.Module):
    """
    Grad-CAM için uygun son convolution layer'ı bul.
    """
    if hasattr(model, "features"):
        layer = find_last_conv_layer(model.features)
        if layer is not None:
            return layer

    if hasattr(model, "layer4"):
        layer = find_last_conv_layer(model.layer4)
        if layer is not None:
            return layer

    layer = find_last_conv_layer(model)
    if layer is None:
        raise ValueError("Grad-CAM target layer bulunamadı.")
    return layer


# =========================
# FASTAPI
# =========================
app = FastAPI(title="AI Quality Control API")

app.include_router(auth_router)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "classes": list(idx_to_class.values()),
        "arch": ARCH,
    }


@app.get("/metrics")
def metrics():
    return {
        "arch": ARCH,
        "device": device,
        "classes": list(idx_to_class.values()),
    }


# =========================
# PREDICT
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    content = await file.read()

    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.jpg"
    save_path = UPLOAD_DIR / fname
    save_path.write_bytes(content)

    img = Image.open(io.BytesIO(content)).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    # Prediction kısmında gradient gerekmiyor
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    conf = float(conf.item())
    pred_idx = int(pred_idx.item())
    defect_class = idx_to_class[pred_idx]

    severity = severity_score(conf)
    decision = quality_decision(defect_class, severity, conf)

    try:
        report = llm_generate_report(defect_class, conf, severity, decision)
    except Exception:
        report = "LLM error"

    image_public_path = f"/uploads/{fname}"

    # Grad-CAM için gradient gerekir
    try:
        target_layer = get_target_layer(model)
        overlay = generate_gradcam(model, x, target_layer, save_path)

        gradcam_fname = f"gradcam_{fname}"
        gradcam_path = UPLOAD_DIR / gradcam_fname
        cv2.imwrite(str(gradcam_path), overlay)

        gradcam_public_path = f"/uploads/{gradcam_fname}"
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        gradcam_public_path = None

    owner_name = current_user.get("name")
    owner_photo = current_user.get("photo")

    try:
        user_doc = users_collection.find_one({"email": current_user["email"]})
        if user_doc:
            owner_name = user_doc.get("name") or owner_name or current_user["email"]
            owner_photo = user_doc.get("photo") or owner_photo
    except Exception:
        pass

    if not owner_name:
        owner_name = current_user["email"]

    result = {
        "user_email": current_user["email"],
        "owner_name": owner_name,
        "owner_photo": to_public_upload_path(owner_photo),
        "defect": defect_class,
        "confidence": round(conf, 4),
        "severity": severity,
        "decision": decision,
        "image": image_public_path,
        "gradcam": gradcam_public_path,
        "report": report,
        "timestamp": datetime.utcnow(),
    }

    predictions_collection.insert_one(result)

    return {
        "defect": defect_class,
        "confidence": round(conf, 4),
        "severity": severity,
        "decision": decision,
        "image_path": image_public_path,
        "gradcam_path": gradcam_public_path,
        "report": report,
        "owner_name": owner_name,
        "owner_photo": to_public_upload_path(owner_photo),
    }


# =========================
# PRIVATE HISTORY
# =========================
@app.get("/history")
def history(current_user: dict = Depends(get_current_user)):
    data = list(
        predictions_collection.find(
            {"user_email": current_user["email"]}
        ).sort("timestamp", -1).limit(50)
    )

    return [format_prediction(d) for d in data]


@app.delete("/history/{item_id}")
def delete_history_item(
    item_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        object_id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid history item id")

    result = predictions_collection.delete_one({
        "_id": object_id,
        "user_email": current_user["email"],
    })

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="History item not found")

    return {"message": "Deleted successfully"}


# =========================
# PUBLIC HISTORY
# =========================
@app.get("/history/public")
def public_history():
    data = list(
        predictions_collection.find()
        .sort("timestamp", -1)
        .limit(100)
    )

    return [format_prediction(d) for d in data]


# =========================
# ANALYTICS
# =========================
@app.get("/analytics")
def analytics(current_user: dict = Depends(get_current_user)):
    pipeline = [
        {
            "$match": {
                "user_email": current_user["email"]
            }
        },
        {
            "$group": {
                "_id": "$defect",
                "count": {"$sum": 1}
            }
        }
    ]

    stats = list(predictions_collection.aggregate(pipeline))

    return {
        "defect_statistics": stats
    }