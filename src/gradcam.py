import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# ---------- Grad-CAM helper ----------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # [B, C, H, W]

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] shape: [B, C, H, W]
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        # activations: [1, C, H, W], gradients: [1, C, H, W]
        A = self.activations.detach()
        G = self.gradients.detach()

        # Global average pool gradients over H,W -> weights [1, C, 1, 1]
        weights = G.mean(dim=(2, 3), keepdim=True)

        # Weighted sum over channels -> [1, 1, H, W]
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize to [0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, logits.detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to an image file")
    parser.add_argument("--outdir", type=str, default="outputs/figures", help="Output directory")
    parser.add_argument("--class_idx", type=int, default=None, help="Optional: force target class index")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load("outputs/models/best_model.pt", map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Build model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Target layer for ResNet18: last conv block
    target_layer = model.layer4

    # Same preprocess as train/eval (no normalize)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_path = Path(args.image)
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    cam_engine = GradCAM(model, target_layer)

    # If class_idx not given, use predicted class
    with torch.no_grad():
        logits0 = model(x)
        pred_idx = int(logits0.argmax(dim=1).item())

    target_idx = pred_idx if args.class_idx is None else args.class_idx

    # Grad-CAM needs gradients, so no torch.no_grad here
    cam, logits = cam_engine(x, target_idx)
    cam_engine.close()

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred_conf = float(probs[pred_idx].item())

    target_name = idx_to_class.get(target_idx, str(target_idx))
    pred_name = idx_to_class.get(pred_idx, str(pred_idx))

    # Prepare visualization
    cam_2d = cam.squeeze(0).squeeze(0).cpu()  # [H,W]

    # Convert original image to display (resized)
    disp_img = img.resize((224, 224))
    disp_img_t = transforms.ToTensor()(disp_img).permute(1, 2, 0).cpu().numpy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_name = f"gradcam_pred-{pred_name}_conf-{pred_conf:.3f}_target-{target_name}_{img_path.stem}.png"
    out_path = outdir / out_name

    plt.figure()
    plt.imshow(disp_img_t)
    plt.imshow(cam_2d, alpha=0.45)  # heatmap overlay (default colormap)
    plt.title(f"Pred: {pred_name} ({pred_conf:.3f}) | Target CAM: {target_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print("✅ Saved:", out_path)

if __name__ == "__main__":
    main()