import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import torch
import torch.nn as nn
from bson import ObjectId
from PIL import Image
from pydantic import BaseModel
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


# MODEL LOAD
def load_model_and_tf():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    # class bilgisi checkpoint içinde olmalı, yoksa hata veririz
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

    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported architecture: {arch}. "
            f"Supported architectures are: efficientnet_b0, resnet50"
        )

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


# RISK / DECISION RULES kendimiz belirliyoruz
DEFECT_SEVERITY_MAP = {
    "crazing": "Medium",
    "inclusion": "Medium",
    "patches": "High",
    "pitted_surface": "Medium",
    "rolled-in_scale": "High",
    "scratches": "Low",
}

CALIBRATION_THRESHOLD = 0.90   # otomatik karar sınırı
REWORK_THRESHOLD = 0.60        # %60 altındaki tahminler güvenilmez olduğu için reddedilir
TOP2_UNCERTAINTY_MARGIN = 0.10 # kararsızlığa bakıyor


def severity_score(defect_class: str) -> str:
    return DEFECT_SEVERITY_MAP.get(defect_class, "Medium")


def confidence_interpretation(confidence: float) -> str:
    if confidence >= 0.90:
        return "Very confident"
    elif confidence >= 0.75:
        return "High confidence"
    elif confidence >= 0.60:
        return "Moderate confidence"
    elif confidence >= 0.40:
        return "Low confidence"
    return "Very low confidence"


def quality_decision(defect_class: str, confidence: float, uncertainty_flag: bool = False) -> tuple[str, str]:
    """
    Final industrial decision logic with only 3 classes:
    ACCEPT / REWORK / REJECT

    Rules:
    1. If model is uncertain (top-2 gap too small), reject for safety.
    2. If confidence < 0.60, reject.
    3. If 0.60 <= confidence < 0.90, rework.
    4. If confidence >= 0.90:
       - Low severity    -> ACCEPT
       - Medium severity -> REWORK
       - High severity   -> REJECT
    """
    severity = severity_score(defect_class)

    if uncertainty_flag:
        return "REJECT", severity

    if confidence < REWORK_THRESHOLD:
        return "REJECT", severity

    if confidence < CALIBRATION_THRESHOLD:
        return "REWORK", severity

    if severity == "Low":
        return "ACCEPT", severity

    if severity == "Medium":
        return "REWORK", severity

    if severity == "High":
        return "REJECT", severity

    return "REJECT", severity


def production_impact(decision: str, defect_class: str, severity: str) -> str:
    if decision == "REJECT":
        return "High"

    if decision == "REWORK":
        if severity == "High":
            return "Moderate to High"
        if severity == "Medium":
            return "Moderate"
        return "Low to Moderate"

    if decision == "ACCEPT":
        if defect_class == "scratches":
            return "Minimal"
        return "Low"

    return "Unknown"


def build_fallback_report( # LLM raporu tutarsız veya üretilemezse bu fallback raporu kullanacağız.
    defect_class: str,
    confidence: float,
    severity: str,
    decision: str,
    uncertainty_flag: bool = False, # Model belirsiz olduğunda raporda bunu belirtmek için kullanacağız.
    top_predictions: list[dict] | None = None
) -> str:
    pretty_name = defect_class.replace("_", " ").replace("-", " ").capitalize() # "pitted_surface" -> "Pitted surface"
    impact = production_impact(decision, defect_class, severity) # "High", "Moderate", "Low" gibi ifadelerle üretim hattına etkisini açıklıyoruz.
    conf_text = confidence_interpretation(confidence) # "High confidence", "Low confidence" gibi ifadelerle güven seviyesini açıklıyoruz.

    if uncertainty_flag: # Model belirsiz olduğunda daha temkinli bir dil kullanarak ürünü reddettiğimizi ve manuel inceleme gerektiğini belirtiyoruz.
        action_text = (
            "The model is uncertain because the top class probabilities are close. "
            "For safety, the product is marked as REJECT and should be manually inspected."
        )
    elif decision == "REWORK":
        action_text = "The product should be rechecked or reworked before approval."
    elif decision == "REJECT":
        action_text = "The product should be rejected due to low confidence or defect severity."
    else:
        action_text = "The product can be accepted."

    top_pred_text = ""
    if top_predictions:
        formatted = ", ".join(
            [f"{p['class']} ({p['probability']:.4f})" for p in top_predictions] # "Top Predictions: scratches (0.8500), pitted_surface (0.1000), inclusion (0.0500)"
        )
        top_pred_text = f"\nTop Predictions: {formatted}"

    return ( # Raporun genel formatı
        f"Defect Description: {pretty_name} detected\n"
        f"Risk Level: {severity}\n"
        f"Recommended Action: {decision}\n"
        f"Action Detail: {action_text}\n"
        f"Production Impact: {impact}\n"
        f"Confidence Interpretation: {conf_text}"
        f"{top_pred_text}"
    )


# LLM VALIDATION LAYER
# Bu katman LLM'in model çıktısıyla çelişip çelişmediğini kontrol eder.
# Amaç: LLM halüsinasyonunu azaltmak ve raporu model kararlarıyla tutarlı hale getirmek.
def validate_llm_report(report, severity: str, decision: str) -> list[str]:
    # Burada report bazen string, bazen dict gelebilir.
    # Eski hata: report dict geldiğinde report.lower() patlıyordu.
    # Bu yüzden önce tip kontrolü yapıyoruz.
    errors = []

    if not report:
        errors.append("Empty report")
        return errors

    severity_lower = severity.lower()
    decision_lower = decision.lower()

    if isinstance(report, dict):
        # LLM JSON/dict döndürürse alanları ayrı ayrı kontrol ediyoruz.
        risk_level = str(report.get("risk_level", "")).lower()
        recommended_action = str(report.get("recommended_action", "")).lower()
        confidence_text = str(report.get("confidence_interpretation", "")).lower()
        full_text = " ".join(str(v).lower() for v in report.values())
    else:
        # LLM string döndürürse eski sistem gibi tüm metinden kontrol ediyoruz.
        full_text = str(report).lower()
        risk_level = full_text
        recommended_action = full_text
        confidence_text = full_text

    # Risk seviyesi modelin severity değeriyle aynı mı diye kontrol ediyoruz.
    # Örnek: Model Low derken LLM Medium yazarsa fallback devreye girer.
    if severity_lower not in risk_level:
        errors.append("Severity mismatch")

    # Önerilen aksiyon modelin decision değeriyle aynı mı diye kontrol ediyoruz.
    # Örnek: Model ACCEPT derken LLM REJECT derse fallback devreye girer.
    # Burada "Reject the product" içinde "reject" geçtiği için doğru kabul edilir.
    if decision_lower not in recommended_action:
        errors.append("Decision mismatch")

    # Low severity olan bir durumda LLM çok ağır ifadeler kullanırsa bunu da riskli kabul ediyoruz.
    if severity == "Low" and ("high risk" in full_text or "severe" in full_text):
        errors.append("Low severity but high-risk language detected")

    return errors


def to_public_upload_path(path_value): # Verilen bir dosya yolu değerini, eğer mümkünse, "/uploads/filename.jpg" formatında halka açık bir URL'ye dönüştürür. Bu, kullanıcıların kendi yükledikleri dosyalara veya diğer dosyalara erişebilmesi için gereklidir.
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


def format_prediction(doc): # Veritabanından alınan bir tahmin belgesini, API yanıtında döndürmek için uygun bir formata dönüştürür. Bu fonksiyon, tarih formatlama, dosya yollarını halka açık URL'lere dönüştürme ve eksik alanları güvenli bir şekilde ele alma gibi işlemleri yapar.
    return {
        "_id": str(doc["_id"]),
        "defect": doc.get("defect"),
        "confidence": doc.get("confidence"),
        "severity": doc.get("severity"),
        "decision": doc.get("decision"),
        "image_path": to_public_upload_path(doc.get("image")),
        "gradcam_path": to_public_upload_path(doc.get("gradcam")),
        "report": doc.get("report"),
        "validation": doc.get("validation", {}),
        "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None,
        "owner_name": doc.get("owner_name"),
        "owner_photo": to_public_upload_path(doc.get("owner_photo")),
        "user_email": doc.get("user_email"),
        "uncertainty_flag": doc.get("uncertainty_flag", False),
        "top_predictions": doc.get("top_predictions", []),
    }


def find_last_conv_layer(module: nn.Module): # Verilen bir PyTorch modülünde (örneğin, modelin tamamı veya belirli bir alt modül) bulunan son Conv2d katmanını bulur. Bu, Grad-CAM için hedef katmanı belirlemek amacıyla kullanılır. Eğer Conv2d katmanı bulunamazsa None döner.
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def get_target_layer(model: nn.Module):
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


# FASTAPI
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


# Bu endpointler API'nin sağlıklı çalıştığını ve modelin yüklendiğini doğrulamak için kullanılabilir.
# Ayrıca model ve karar kuralları hakkında temel bilgileri döner.
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "classes": list(idx_to_class.values()),
        "arch": ARCH,
        "model_path": str(MODEL_PATH),
        "calibration_threshold": CALIBRATION_THRESHOLD,
        "rework_threshold": REWORK_THRESHOLD,
        "top2_uncertainty_margin": TOP2_UNCERTAINTY_MARGIN,
    }


# Bu endpoint, modelin mimarisi, sınıf isimleri, karar kuralları ve diğer önemli metrikler hakkında bilgi sağlar.
@app.get("/metrics")
def metrics():
    return {
        "arch": ARCH,
        "device": device,
        "classes": list(idx_to_class.values()),
        "supported_architectures": ["efficientnet_b0", "resnet50"],
        "thresholds": {
            "rework_threshold": REWORK_THRESHOLD,
            "accept_reject_threshold": CALIBRATION_THRESHOLD,
            "top2_uncertainty_margin": TOP2_UNCERTAINTY_MARGIN,
        },
        "severity_mapping": DEFECT_SEVERITY_MAP,
        "decision_classes": ["ACCEPT", "REWORK", "REJECT"],
    }


# PREDICT
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

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        top3_probs, top3_idxs = torch.topk(probs, k=min(3, num_classes), dim=1)
        top2_probs, _ = torch.topk(probs, k=min(2, num_classes), dim=1)

    confidence = float(conf.item())
    pred_idx = int(pred_idx.item())
    defect_class = idx_to_class[pred_idx]

    top_predictions = []
    for i in range(top3_probs.shape[1]):
        cls_idx = int(top3_idxs[0][i].item())
        cls_prob = float(top3_probs[0][i].item())
        top_predictions.append({
            "class": idx_to_class[cls_idx],
            "probability": round(cls_prob, 4),
        })

    if top2_probs.shape[1] >= 2:
        top1 = float(top2_probs[0][0].item())
        top2 = float(top2_probs[0][1].item())
        uncertainty_gap = abs(top1 - top2)
    else:
        top1 = confidence
        top2 = 0.0
        uncertainty_gap = confidence

    uncertainty_flag = uncertainty_gap < TOP2_UNCERTAINTY_MARGIN

    decision, severity = quality_decision(
        defect_class=defect_class,
        confidence=confidence,
        uncertainty_flag=uncertainty_flag
    )

    # Debug çıktıları
    print("=== DEBUG PREDICT ===")
    print("Predicted class:", defect_class)
    print("Confidence:", confidence)
    print("Top predictions:", top_predictions)
    print("Top-2 gap:", round(uncertainty_gap, 4))
    print("Uncertainty flag:", uncertainty_flag)
    print("Final decision:", decision)
    print("Severity:", severity)
    print("=====================")

    # LLM + VALIDATION + FALLBACK
    # Önce LLM raporu üretmeye çalışıyoruz.
    # Sonra raporun model çıktılarıyla uyumlu olup olmadığını kontrol ediyoruz.
    # Eğer LLM severity veya decision açısından çelişirse fallback rapor kullanıyoruz.
    try:
        llm_report = llm_generate_report(defect_class, confidence, severity, decision)

        validation_errors = validate_llm_report(
            report=llm_report,
            severity=severity,
            decision=decision
        )

        if validation_errors:
            print("⚠️ LLM VALIDATION ERROR:", validation_errors)

            report = build_fallback_report(
                defect_class=defect_class,
                confidence=confidence,
                severity=severity,
                decision=decision,
                uncertainty_flag=uncertainty_flag,
                top_predictions=top_predictions
            )

            validation_info = {
                "status": "fallback_used",
                "errors": validation_errors,
                "message": "LLM report was inconsistent with model output. Fallback report was used."
            }

        else:
            report = llm_report

            validation_info = {
                "status": "ok",
                "errors": [],
                "message": "LLM report is consistent with model output."
            }

    except Exception as e:
        print("❌ LLM ERROR:", e)

        report = build_fallback_report(
            defect_class=defect_class,
            confidence=confidence,
            severity=severity,
            decision=decision,
            uncertainty_flag=uncertainty_flag,
            top_predictions=top_predictions
        )

        validation_info = {
            "status": "llm_failed",
            "errors": [str(e)],
            "message": "LLM report generation failed. Fallback report was used."
        }

    image_public_path = f"/uploads/{fname}"

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
        "confidence": round(confidence, 4),
        "severity": severity,
        "decision": decision,
        "image": image_public_path,
        "gradcam": gradcam_public_path,
        "report": report,
        "validation": validation_info,
        "timestamp": datetime.utcnow(),
        "uncertainty_flag": uncertainty_flag,
        "top_predictions": top_predictions,
        "top2_gap": round(uncertainty_gap, 4),
    }

    predictions_collection.insert_one(result)

    return {
        "defect": defect_class,
        "confidence": round(confidence, 4),
        "severity": severity,
        "decision": decision,
        "image_path": image_public_path,
        "gradcam_path": gradcam_public_path,
        "report": report,
        "validation": validation_info,
        "owner_name": owner_name,
        "owner_photo": to_public_upload_path(owner_photo),
        "uncertainty_flag": uncertainty_flag,
        "top_predictions": top_predictions,
        "top2_gap": round(uncertainty_gap, 4),
    }


# PRIVATE HISTORY
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


# PROFILE
class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None


@app.get("/profile")
def get_profile(current_user: dict = Depends(get_current_user)):
    user_doc = users_collection.find_one({"email": current_user["email"]})

    if not user_doc:
        return {
            "name": current_user.get("name", ""),
            "email": current_user["email"],
            "photo": to_public_upload_path(current_user.get("photo")),
        }

    return {
        "name": user_doc.get("name", ""),
        "email": user_doc.get("email", current_user["email"]),
        "photo": to_public_upload_path(user_doc.get("photo")),
    }


@app.put("/profile")
def update_profile(
    payload: ProfileUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    update_data = {}

    if payload.name is not None:
        update_data["name"] = payload.name.strip()

    if update_data:
        users_collection.update_one(
            {"email": current_user["email"]},
            {
                "$set": update_data,
                "$setOnInsert": {"email": current_user["email"]}
            },
            upsert=True
        )

    user_doc = users_collection.find_one({"email": current_user["email"]})

    return {
        "message": "Profile updated successfully",
        "profile": {
            "name": user_doc.get("name", "") if user_doc else payload.name,
            "email": current_user["email"],
            "photo": to_public_upload_path(user_doc.get("photo")) if user_doc else None,
        }
    }


@app.post("/profile/photo")
async def upload_profile_photo(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    content = await file.read()

    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    fname = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / fname
    save_path.write_bytes(content)

    photo_path = f"/uploads/{fname}"

    users_collection.update_one(
        {"email": current_user["email"]},
        {
            "$set": {"photo": photo_path},
            "$setOnInsert": {"email": current_user["email"]}
        },
        upsert=True
    )

    user_doc = users_collection.find_one({"email": current_user["email"]})

    return {
        "message": "Profile photo uploaded successfully",
        "photo": to_public_upload_path(photo_path),
        "profile": {
            "name": user_doc.get("name", "") if user_doc else "",
            "email": current_user["email"],
            "photo": to_public_upload_path(photo_path),
        }
    }


# PUBLIC HISTORY
@app.get("/history/public")
def public_history():
    data = list(
        predictions_collection.find()
        .sort("timestamp", -1) 
        .limit(100)
    )

    return [format_prediction(d) for d in data]


# ANALYTICS
@app.get("/analytics")
def analytics(current_user: dict = Depends(get_current_user)):
    pipeline = [
        {
            "$match": {
                "user_email": current_user["email"]
            }
        },
        {
            "$sort": {
                "timestamp": -1
            }
        },
        {
            "$group": {
                "_id": "$defect",
                "count": {"$sum": 1},
                "images": {"$push": "$image"},
                "avg_confidence": {"$avg": "$confidence"},
                "decisions": {"$push": "$decision"}
            }
        },
        {
            "$project": {
                "_id": 1,
                "count": 1,
                "images": {
                    "$slice": [
                        {
                            "$filter": {
                                "input": "$images",
                                "as": "img",
                                "cond": {
                                    "$and": [
                                        {"$ne": ["$$img", None]},
                                        {"$ne": ["$$img", ""]}
                                    ]
                                }
                            }
                        },
                        5
                    ]
                },
                "avg_confidence": {
                    "$round": ["$avg_confidence", 4]
                },
                "accept_count": {
                    "$size": {
                        "$filter": {
                            "input": "$decisions",
                            "as": "d",
                            "cond": {"$eq": ["$$d", "ACCEPT"]}
                        }
                    }
                },
                "rework_count": {
                    "$size": {
                        "$filter": {
                            "input": "$decisions",
                            "as": "d",
                            "cond": {"$eq": ["$$d", "REWORK"]}
                        }
                    }
                },
                "reject_count": {
                    "$size": {
                        "$filter": {
                            "input": "$decisions",
                            "as": "d",
                            "cond": {"$eq": ["$$d", "REJECT"]}
                        }
                    }
                }
            }
        },
        {
            "$sort": {
                "count": -1
            }
        }
    ]

    stats = list(predictions_collection.aggregate(pipeline))

    total_inspections = sum(item.get("count", 0) for item in stats)
    total_accept = sum(item.get("accept_count", 0) for item in stats)
    total_rework = sum(item.get("rework_count", 0) for item in stats)
    total_reject = sum(item.get("reject_count", 0) for item in stats)

    return {
        "total_inspections": total_inspections,
        "decision_overview": {
            "accept": total_accept,
            "rework": total_rework,
            "reject": total_reject
        },
        "defect_statistics": stats
    }