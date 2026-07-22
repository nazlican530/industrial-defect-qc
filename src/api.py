import io
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import torch
import torch.nn as nn
from bson import ObjectId
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

from src.auth import get_current_user, router as auth_router
from src.database import predictions_collection, users_collection
from src.gradcam_utils import generate_gradcam
from src.llm_agent import generate_report as llm_generate_report


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("outputs/models/best_model_mobilenetv3.pt")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_THRESHOLD = 0.90
TOP2_UNCERTAINTY_MARGIN = 0.10

DEFECT_SEVERITY_MAP = {
    "crazing": "Medium",
    "inclusion": "Medium",
    "patches": "High",
    "pitted_surface": "Medium",
    "rolled-in_scale": "High",
    "scratches": "Low",
}

SUPPORTED_ARCHITECTURES = [
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "mobilenet_v3_large",
]

DECISION_CLASSES = [
    "ACCEPT",
    "REWORK",
    "REJECT",
    "HUMAN REVIEW",
]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


device = get_device()


# ============================================================
# MODEL LOADING
# ============================================================

def normalize_arch_name(arch: str) -> str:
    normalized = str(arch).strip().lower().replace("-", "_")

    aliases = {
        "mobilenetv3_large": "mobilenet_v3_large",
        "mobilenet_v3": "mobilenet_v3_large",
        "mobilenetv3": "mobilenet_v3_large",
        "efficientnetb0": "efficientnet_b0",
        "densenet_121": "densenet121",
        "resnet_50": "resnet50",
    }

    return aliases.get(normalized, normalized)


def create_model(arch: str, num_classes: int) -> nn.Module:
    arch = normalize_arch_name(arch)

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes,
        )
        return model

    if arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported architecture: {arch}. "
        f"Supported architectures: {SUPPORTED_ARCHITECTURES}"
    )


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    raise KeyError(
        "Checkpoint does not contain model_state, "
        "model_state_dict, or state_dict."
    )


def clean_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # Compatibility with an older EfficientNet checkpoint.
        new_key = new_key.replace(
            "classifier.1.1",
            "classifier.1",
        )

        cleaned[new_key] = value

    return cleaned


def load_model_and_transform():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    class_to_idx = checkpoint.get("class_to_idx")
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise KeyError(
            "Checkpoint must contain a non-empty class_to_idx mapping."
        )

    idx_to_class = {
        int(index): class_name
        for class_name, index in class_to_idx.items()
    }

    num_classes = len(class_to_idx)

    arch = normalize_arch_name(
        checkpoint.get("arch", "mobilenet_v3_large")
    )

    model = create_model(
        arch=arch,
        num_classes=num_classes,
    )

    state_dict = clean_state_dict(
        extract_state_dict(checkpoint)
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint is incompatible with architecture '{arch}': {exc}"
        ) from exc

    model.to(device)
    model.eval()

    mean = tuple(
        checkpoint.get(
            "imagenet_mean",
            (0.485, 0.456, 0.406),
        )
    )
    std = tuple(
        checkpoint.get(
            "imagenet_std",
            (0.229, 0.224, 0.225),
        )
    )

    image_size = int(
        checkpoint.get("image_size", 224)
    )

    inference_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return (
        model,
        inference_transform,
        idx_to_class,
        num_classes,
        arch,
        image_size,
    )


(
    model,
    tf,
    idx_to_class,
    num_classes,
    ARCH,
    IMAGE_SIZE,
) = load_model_and_transform()


# ============================================================
# DECISION ENGINE
# ============================================================

def severity_score(defect_class: str) -> str:
    return DEFECT_SEVERITY_MAP.get(
        defect_class,
        "Medium",
    )


def confidence_interpretation(confidence: float) -> str:
    if confidence >= 0.90:
        return "Very confident"

    if confidence >= 0.75:
        return "High confidence"

    if confidence >= 0.60:
        return "Moderate confidence"

    if confidence >= 0.40:
        return "Low confidence"

    return "Very low confidence"


def quality_decision(
    defect_class: str,
    confidence: float,
    uncertainty_flag: bool = False,
) -> tuple[str, str]:
    """
    Decision logic aligned with Algorithm 1 in the manuscript.

    1. Top-2 gap < 0.10 -> HUMAN REVIEW
    2. Confidence < 0.90 -> HUMAN REVIEW
    3. Confidence >= 0.90 -> severity-based decision
       - Low severity    -> ACCEPT
       - Medium severity -> REWORK
       - High severity   -> REJECT
    """
    severity = severity_score(defect_class)

    if uncertainty_flag:
        return "HUMAN REVIEW", severity

    if confidence < CALIBRATION_THRESHOLD:
        return "HUMAN REVIEW", severity

    if severity == "Low":
        return "ACCEPT", severity

    if severity == "Medium":
        return "REWORK", severity

    if severity == "High":
        return "REJECT", severity

    return "HUMAN REVIEW", severity


def production_impact(
    decision: str,
    defect_class: str,
    severity: str,
) -> str:
    if decision == "HUMAN REVIEW":
        return "Pending operator verification"

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


# ============================================================
# FALLBACK REPORT
# ============================================================

def build_fallback_report(
    defect_class: str,
    confidence: float,
    severity: str,
    decision: str,
    uncertainty_flag: bool = False,
    top_predictions: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    pretty_name = (
        defect_class
        .replace("_", " ")
        .replace("-", " ")
        .capitalize()
    )

    impact = production_impact(
        decision=decision,
        defect_class=defect_class,
        severity=severity,
    )

    confidence_text = confidence_interpretation(
        confidence
    )

    if decision == "HUMAN REVIEW":
        if uncertainty_flag:
            action_detail = (
                "The Top-2 class probabilities are too close. "
                "Operator verification is required before an "
                "ACCEPT, REWORK, or REJECT action is assigned."
            )
        else:
            action_detail = (
                "The prediction confidence is below the automation "
                "threshold. Operator verification is required before "
                "a final quality-control action is assigned."
            )

    elif decision == "REWORK":
        action_detail = (
            "The product should be rechecked or reworked "
            "before approval."
        )

    elif decision == "REJECT":
        action_detail = (
            "The product should be rejected according to "
            "the configured defect-severity rules."
        )

    else:
        action_detail = (
            "The product can be accepted under the "
            "configured quality-control rules."
        )

    top_prediction_text = ""

    if top_predictions:
        formatted = ", ".join(
            f"{item['class']} ({item['probability']:.4f})"
            for item in top_predictions
        )
        top_prediction_text = formatted

    return {
        "defect_description": f"{pretty_name} detected",
        "risk_level": severity,
        "recommended_action": decision,
        "action_detail": action_detail,
        "production_impact": impact,
        "confidence_interpretation": confidence_text,
        "top_predictions": top_prediction_text,
    }


# ============================================================
# LLM VALIDATION
# ============================================================

REQUIRED_REPORT_FIELDS = {
    "defect_description",
    "risk_level",
    "recommended_action",
    "production_impact",
    "confidence_interpretation",
}


def normalize_text(value: Any) -> str:
    text = str(value).lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_llm_report(
    report: dict[str, Any],
    defect_class: str,
    confidence: float,
    severity: str,
    decision: str,
) -> list[str]:
    errors: list[str] = []

    if not isinstance(report, dict):
        return ["Invalid report type"]

    missing_fields = sorted(
        field
        for field in REQUIRED_REPORT_FIELDS
        if field not in report
        or not str(report[field]).strip()
    )

    if missing_fields:
        errors.append(
            "Missing fields: "
            + ", ".join(missing_fields)
        )

    expected_defect = normalize_text(defect_class)
    actual_defect = normalize_text(
        report.get("defect_description", "")
    )

    if expected_defect not in actual_defect:
        errors.append("Defect-class mismatch")

    expected_severity = normalize_text(severity)
    actual_severity = normalize_text(
        report.get("risk_level", "")
    )

    if actual_severity != expected_severity:
        errors.append("Severity mismatch")

    expected_decision = normalize_text(decision)
    actual_decision = normalize_text(
        report.get("recommended_action", "")
    )

    if expected_decision not in actual_decision:
        errors.append("Decision mismatch")

    expected_confidence = normalize_text(
        confidence_interpretation(confidence)
    )
    actual_confidence = normalize_text(
        report.get("confidence_interpretation", "")
    )

    if expected_confidence not in actual_confidence:
        errors.append(
            "Confidence-interpretation mismatch"
        )

    full_text = normalize_text(
        " ".join(
            str(value)
            for value in report.values()
        )
    )

    unsupported_high_risk_phrases = (
        "critical risk",
        "catastrophic",
        "severe defect",
        "immediate shutdown",
    )

    if severity == "Low" and any(
        phrase in full_text
        for phrase in unsupported_high_risk_phrases
    ):
        errors.append(
            "Unsupported high-risk language"
        )

    return errors


# ============================================================
# FILE AND RESPONSE UTILITIES
# ============================================================

def to_public_upload_path(
    path_value: Any,
) -> Optional[str]:
    if not path_value:
        return None

    path_str = str(path_value).replace("\\", "/")

    if "/uploads/" in path_str:
        return path_str[
            path_str.index("/uploads/"):
        ]

    if path_str.startswith("uploads/"):
        return f"/{path_str}"

    if path_str.startswith("/uploads/"):
        return path_str

    filename = Path(path_str).name
    return f"/uploads/{filename}"


def format_prediction(
    document: dict[str, Any],
) -> dict[str, Any]:
    timestamp = document.get("timestamp")

    return {
        "_id": str(document["_id"]),
        "defect": document.get("defect"),
        "confidence": document.get("confidence"),
        "severity": document.get("severity"),
        "decision": document.get("decision"),
        "image_path": to_public_upload_path(
            document.get("image")
        ),
        "gradcam_path": to_public_upload_path(
            document.get("gradcam")
        ),
        "report": document.get("report"),
        "validation": document.get(
            "validation",
            {},
        ),
        "timestamp": (
            timestamp.isoformat()
            if timestamp
            else None
        ),
        "owner_name": document.get("owner_name"),
        "owner_photo": to_public_upload_path(
            document.get("owner_photo")
        ),
        "user_email": document.get("user_email"),
        "uncertainty_flag": document.get(
            "uncertainty_flag",
            False,
        ),
        "top_predictions": document.get(
            "top_predictions",
            [],
        ),
        "top2_gap": document.get("top2_gap"),
    }


def find_last_conv_layer(
    module: nn.Module,
) -> Optional[nn.Conv2d]:
    last_conv = None

    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv = layer

    return last_conv


def get_target_layer(
    current_model: nn.Module,
) -> nn.Conv2d:
    if hasattr(current_model, "features"):
        layer = find_last_conv_layer(
            current_model.features
        )

        if layer is not None:
            return layer

    if hasattr(current_model, "layer4"):
        layer = find_last_conv_layer(
            current_model.layer4
        )

        if layer is not None:
            return layer

    layer = find_last_conv_layer(current_model)

    if layer is None:
        raise ValueError(
            "Grad-CAM target layer could not be found."
        )

    return layer


def validate_uploaded_image(
    file: UploadFile,
    content: bytes,
) -> None:
    allowed_content_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPEG, PNG, and WEBP images "
                "are supported."
            ),
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    max_size_bytes = 10 * 1024 * 1024

    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB.",
        )


# ============================================================
# FASTAPI INITIALIZATION
# ============================================================

app = FastAPI(
    title="RobustDefect-LLM Quality Control API",
    version="1.0.0",
)

app.include_router(auth_router)

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH AND METRICS
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": device,
        "classes": list(idx_to_class.values()),
        "arch": ARCH,
        "model_path": str(MODEL_PATH),
        "image_size": IMAGE_SIZE,
        "automation_threshold": CALIBRATION_THRESHOLD,
        "top2_uncertainty_margin": TOP2_UNCERTAINTY_MARGIN,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {
        "arch": ARCH,
        "device": device,
        "classes": list(idx_to_class.values()),
        "supported_architectures": SUPPORTED_ARCHITECTURES,
        "thresholds": {
            "automation_threshold": CALIBRATION_THRESHOLD,
            "top2_uncertainty_margin": (
                TOP2_UNCERTAINTY_MARGIN
            ),
        },
        "severity_mapping": DEFECT_SEVERITY_MAP,
        "decision_classes": DECISION_CLASSES,
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(
        get_current_user
    ),
) -> dict[str, Any]:
    content = await file.read()

    validate_uploaded_image(
        file=file,
        content=content,
    )

    try:
        image = Image.open(
            io.BytesIO(content)
        ).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    filename = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex}.jpg"
    )

    save_path = UPLOAD_DIR / filename
    image.save(
        save_path,
        format="JPEG",
        quality=95,
    )

    input_tensor = (
        tf(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence_tensor, predicted_index_tensor = (
            torch.max(
                probabilities,
                dim=1,
            )
        )

        top_k = min(3, num_classes)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=1,
        )

        top2_probabilities, _ = torch.topk(
            probabilities,
            k=min(2, num_classes),
            dim=1,
        )

    confidence = float(
        confidence_tensor.item()
    )

    predicted_index = int(
        predicted_index_tensor.item()
    )

    defect_class = idx_to_class[
        predicted_index
    ]

    top_predictions: list[dict[str, Any]] = []

    for position in range(
        top_probabilities.shape[1]
    ):
        class_index = int(
            top_indices[0][position].item()
        )

        class_probability = float(
            top_probabilities[0][position].item()
        )

        top_predictions.append({
            "class": idx_to_class[class_index],
            "probability": round(
                class_probability,
                4,
            ),
        })

    if top2_probabilities.shape[1] >= 2:
        top1 = float(
            top2_probabilities[0][0].item()
        )
        top2 = float(
            top2_probabilities[0][1].item()
        )
        uncertainty_gap = abs(top1 - top2)

    else:
        uncertainty_gap = confidence

    uncertainty_flag = (
        uncertainty_gap
        < TOP2_UNCERTAINTY_MARGIN
    )

    decision, severity = quality_decision(
        defect_class=defect_class,
        confidence=confidence,
        uncertainty_flag=uncertainty_flag,
    )

    print("=== DEBUG PREDICT ===")
    print("Predicted class:", defect_class)
    print("Confidence:", confidence)
    print("Top predictions:", top_predictions)
    print(
        "Top-2 gap:",
        round(uncertainty_gap, 4),
    )
    print(
        "Uncertainty flag:",
        uncertainty_flag,
    )
    print("Final decision:", decision)
    print("Severity:", severity)
    print("=====================")

    try:
        llm_report = llm_generate_report(
            defect_class,
            confidence,
            severity,
            decision,
        )

        validation_errors = validate_llm_report(
            report=llm_report,
            defect_class=defect_class,
            confidence=confidence,
            severity=severity,
            decision=decision,
        )

        if validation_errors:
            report = build_fallback_report(
                defect_class=defect_class,
                confidence=confidence,
                severity=severity,
                decision=decision,
                uncertainty_flag=uncertainty_flag,
                top_predictions=top_predictions,
            )

            validation_info = {
                "status": "fallback_used",
                "errors": validation_errors,
                "message": (
                    "The LLM report was inconsistent with "
                    "the structured model output. "
                    "The fallback report was used."
                ),
            }

        else:
            report = llm_report

            validation_info = {
                "status": "ok",
                "errors": [],
                "message": (
                    "The LLM report is consistent with "
                    "the structured model output."
                ),
            }

    except Exception as exc:
        print("LLM ERROR:", exc)

        report = build_fallback_report(
            defect_class=defect_class,
            confidence=confidence,
            severity=severity,
            decision=decision,
            uncertainty_flag=uncertainty_flag,
            top_predictions=top_predictions,
        )

        validation_info = {
            "status": "llm_failed",
            "errors": [str(exc)],
            "message": (
                "LLM report generation failed. "
                "The fallback report was used."
            ),
        }

    image_public_path = (
        f"/uploads/{filename}"
    )

    try:
        target_layer = get_target_layer(
            model
        )

        overlay = generate_gradcam(
            model,
            input_tensor,
            target_layer,
            save_path,
        )

        gradcam_filename = (
            f"gradcam_{filename}"
        )

        gradcam_path = (
            UPLOAD_DIR / gradcam_filename
        )

        saved = cv2.imwrite(
            str(gradcam_path),
            overlay,
        )

        if not saved:
            raise RuntimeError(
                "Grad-CAM image could not be saved."
            )

        gradcam_public_path = (
            f"/uploads/{gradcam_filename}"
        )

    except Exception as exc:
        print("Grad-CAM error:", exc)
        gradcam_public_path = None

    owner_name = current_user.get("name")
    owner_photo = current_user.get("photo")

    try:
        user_document = users_collection.find_one({
            "email": current_user["email"]
        })

        if user_document:
            owner_name = (
                user_document.get("name")
                or owner_name
                or current_user["email"]
            )

            owner_photo = (
                user_document.get("photo")
                or owner_photo
            )

    except Exception as exc:
        print("Profile lookup error:", exc)

    if not owner_name:
        owner_name = current_user["email"]

    result = {
        "user_email": current_user["email"],
        "owner_name": owner_name,
        "owner_photo": to_public_upload_path(
            owner_photo
        ),
        "defect": defect_class,
        "confidence": round(
            confidence,
            4,
        ),
        "severity": severity,
        "decision": decision,
        "image": image_public_path,
        "gradcam": gradcam_public_path,
        "report": report,
        "validation": validation_info,
        "timestamp": datetime.now(
            timezone.utc
        ),
        "uncertainty_flag": uncertainty_flag,
        "top_predictions": top_predictions,
        "top2_gap": round(
            uncertainty_gap,
            4,
        ),
    }

    insert_result = (
        predictions_collection
        .insert_one(result)
    )

    return {
        "_id": str(insert_result.inserted_id),
        "defect": defect_class,
        "confidence": round(
            confidence,
            4,
        ),
        "severity": severity,
        "decision": decision,
        "image_path": image_public_path,
        "gradcam_path": gradcam_public_path,
        "report": report,
        "validation": validation_info,
        "owner_name": owner_name,
        "owner_photo": to_public_upload_path(
            owner_photo
        ),
        "uncertainty_flag": uncertainty_flag,
        "top_predictions": top_predictions,
        "top2_gap": round(
            uncertainty_gap,
            4,
        ),
    }


# ============================================================
# PRIVATE HISTORY
# ============================================================

@app.get("/history")
def history(
    current_user: dict = Depends(
        get_current_user
    ),
) -> list[dict[str, Any]]:
    data = list(
        predictions_collection.find({
            "user_email": current_user["email"]
        })
        .sort("timestamp", -1)
        .limit(50)
    )

    return [
        format_prediction(document)
        for document in data
    ]


@app.delete("/history/{item_id}")
def delete_history_item(
    item_id: str,
    current_user: dict = Depends(
        get_current_user
    ),
) -> dict[str, str]:
    try:
        object_id = ObjectId(item_id)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid history item ID.",
        ) from exc

    result = predictions_collection.delete_one({
        "_id": object_id,
        "user_email": current_user["email"],
    })

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="History item not found.",
        )

    return {
        "message": "Deleted successfully."
    }


# ============================================================
# PROFILE
# ============================================================

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None


@app.get("/profile")
def get_profile(
    current_user: dict = Depends(
        get_current_user
    ),
) -> dict[str, Any]:
    user_document = users_collection.find_one({
        "email": current_user["email"]
    })

    if not user_document:
        return {
            "name": current_user.get(
                "name",
                "",
            ),
            "email": current_user["email"],
            "photo": to_public_upload_path(
                current_user.get("photo")
            ),
        }

    return {
        "name": user_document.get(
            "name",
            "",
        ),
        "email": user_document.get(
            "email",
            current_user["email"],
        ),
        "photo": to_public_upload_path(
            user_document.get("photo")
        ),
    }


@app.put("/profile")
def update_profile(
    payload: ProfileUpdateRequest,
    current_user: dict = Depends(
        get_current_user
    ),
) -> dict[str, Any]:
    update_data: dict[str, Any] = {}

    if payload.name is not None:
        update_data["name"] = (
            payload.name.strip()
        )

    if update_data:
        users_collection.update_one(
            {
                "email": current_user["email"]
            },
            {
                "$set": update_data,
                "$setOnInsert": {
                    "email": current_user["email"]
                },
            },
            upsert=True,
        )

    user_document = users_collection.find_one({
        "email": current_user["email"]
    })

    return {
        "message": (
            "Profile updated successfully."
        ),
        "profile": {
            "name": (
                user_document.get("name", "")
                if user_document
                else payload.name
            ),
            "email": current_user["email"],
            "photo": (
                to_public_upload_path(
                    user_document.get("photo")
                )
                if user_document
                else None
            ),
        },
    }


@app.post("/profile/photo")
async def upload_profile_photo(
    file: UploadFile = File(...),
    current_user: dict = Depends(
        get_current_user
    ),
) -> dict[str, Any]:
    content = await file.read()

    validate_uploaded_image(
        file=file,
        content=content,
    )

    try:
        image = Image.open(
            io.BytesIO(content)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid profile image.",
        ) from exc

    filename = (
        f"profile_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex}.jpg"
    )

    save_path = UPLOAD_DIR / filename

    image.save(
        save_path,
        format="JPEG",
        quality=95,
    )

    photo_path = (
        f"/uploads/{filename}"
    )

    users_collection.update_one(
        {
            "email": current_user["email"]
        },
        {
            "$set": {
                "photo": photo_path
            },
            "$setOnInsert": {
                "email": current_user["email"]
            },
        },
        upsert=True,
    )

    user_document = users_collection.find_one({
        "email": current_user["email"]
    })

    return {
        "message": (
            "Profile photo uploaded successfully."
        ),
        "photo": photo_path,
        "profile": {
            "name": (
                user_document.get("name", "")
                if user_document
                else ""
            ),
            "email": current_user["email"],
            "photo": photo_path,
        },
    }


# ============================================================
# USER MANAGEMENT
# NOTE: Protect these endpoints with an admin dependency in production.
# ============================================================

@app.get("/users")
def get_users() -> list[dict[str, Any]]:
    users = list(
        users_collection.find(
            {},
            {"hashed_password": 0},
        )
    )

    for user in users:
        user["_id"] = str(user["_id"])

    return users


@app.put("/users/{user_id}")
def update_user(
    user_id: str,
    payload: ProfileUpdateRequest,
) -> dict[str, str]:
    try:
        object_id = ObjectId(user_id)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid user ID.",
        ) from exc

    update_data: dict[str, Any] = {}

    if payload.name is not None:
        update_data["name"] = (
            payload.name.strip()
        )

    result = users_collection.update_one(
        {"_id": object_id},
        {"$set": update_data},
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="User not found.",
        )

    return {
        "message": "User updated successfully."
    }


@app.delete("/users/{user_id}")
def delete_user(
    user_id: str,
) -> dict[str, str]:
    try:
        object_id = ObjectId(user_id)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid user ID.",
        ) from exc

    result = users_collection.delete_one({
        "_id": object_id
    })

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="User not found.",
        )

    return {
        "message": "User deleted successfully."
    }


# ============================================================
# PUBLIC HISTORY
# ============================================================

@app.get("/history/public")
def public_history() -> list[dict[str, Any]]:
    data = list(
        predictions_collection.find()
        .sort("timestamp", -1)
        .limit(100)
    )

    return [
        format_prediction(document)
        for document in data
    ]


# ============================================================
# ANALYTICS
# ============================================================

@app.get("/analytics")
def analytics(
    current_user: dict = Depends(
        get_current_user
    ),
) -> dict[str, Any]:
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
                "avg_confidence": {
                    "$avg": "$confidence"
                },
                "decisions": {
                    "$push": "$decision"
                },
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
                                "as": "image",
                                "cond": {
                                    "$and": [
                                        {
                                            "$ne": [
                                                "$$image",
                                                None,
                                            ]
                                        },
                                        {
                                            "$ne": [
                                                "$$image",
                                                "",
                                            ]
                                        },
                                    ]
                                },
                            }
                        },
                        5,
                    ]
                },
                "avg_confidence": {
                    "$round": [
                        "$avg_confidence",
                        4,
                    ]
                },
                "accept_count": {
                    "$size": {
                        "$filter": {
                            "input": "$decisions",
                            "as": "decision",
                            "cond": {
                                "$eq": [
                                    "$$decision",
                                    "ACCEPT",
                                ]
                            },
                        }
                    }
                },
                "rework_count": {
                    "$size": {
                        "$filter": {
                            "input": "$decisions",
                            "as": "decision",
                            "cond": {
                                "$eq": [
                                    "$$decision",
                                    "REWORK",
                                ]
                            },
                        }
                    }
                },
                "reject_count": {
                    "$size": {
                        "$filter": {
                            "input": "$decisions",
                            "as": "decision",
                            "cond": {
                                "$eq": [
                                    "$$decision",
                                    "REJECT",
                                ]
                            },
                        }
                    }
                },
                "human_review_count": {
                    "$size": {
                        "$filter": {
                            "input": "$decisions",
                            "as": "decision",
                            "cond": {
                                "$eq": [
                                    "$$decision",
                                    "HUMAN REVIEW",
                                ]
                            },
                        }
                    }
                },
            }
        },
        {
            "$sort": {
                "count": -1
            }
        },
    ]

    statistics = list(
        predictions_collection.aggregate(
            pipeline
        )
    )

    total_inspections = sum(
        item.get("count", 0)
        for item in statistics
    )

    total_accept = sum(
        item.get("accept_count", 0)
        for item in statistics
    )

    total_rework = sum(
        item.get("rework_count", 0)
        for item in statistics
    )

    total_reject = sum(
        item.get("reject_count", 0)
        for item in statistics
    )

    total_human_review = sum(
        item.get("human_review_count", 0)
        for item in statistics
    )

    return {
        "total_inspections": total_inspections,
        "decision_overview": {
            "accept": total_accept,
            "rework": total_rework,
            "reject": total_reject,
            "human_review": total_human_review,
        },
        "defect_statistics": statistics,
    }
