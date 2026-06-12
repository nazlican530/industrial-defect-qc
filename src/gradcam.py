import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


#  Grad-CAM yardımıyla modelin hangi bölgelere dikkat ettiğini görselleştiren bir sınıf
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model 
        self.target_layer = target_layer 
        self.activations = None
        self.gradients = None

        self.h1 = target_layer.register_forward_hook(self._forward_hook) 
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    # B = batch size, C = channels, H = height, W = width
    def _forward_hook(self, module, inp, out):  # hook fonksiyonu, hedef katmanın çıktısını kaydeder
        self.activations = out  # [B, C, H, W]

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] shape: [B, C, H, W]
        self.gradients = grad_output[0]
   # Burada model resme bakıyor ve en yüksek skorlu sınıfı buluyor. Ardından, bu sınıfın skoruna göre geri yayılım yaparak hedef katmandaki aktivasyonların ve gradyanların değerlerini kaydediyor. Sonra, bu değerleri kullanarak sınıfın hangi bölgelere dikkat ettiğini gösteren bir ısı haritası (heatmap) oluşturuyor.
    def __call__(self, x: torch.Tensor, class_idx: int): 
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()  #  geri yayılım yapar, sadece o sınıfa ait skoru kullanırız, böylece o sınıfa göre gradyanlar hesaplanır
        score.backward(retain_graph=False) # geri yayılım sırasında ara sonuçları tutmaz, böylece bellek kullanımı azalır

        # activations: [1, C, H, W], gradients: [1, C, H, W]
        A = self.activations.detach()
        G = self.gradients.detach()

        # Global average pool gradients over H,W -> weights [1, C, 1, 1]
        weights = G.mean(dim=(2, 3), keepdim=True)

        # Weighted sum over channels -> [1, 1, H, W]
        cam = (weights * A).sum(dim=1, keepdim=True) # heatmap oluştur
        cam = torch.relu(cam)  # negatif değerleri sıfırla , böylece sadece pozitif etkiler kalır

        # Normalize to [0,1]
        cam = cam - cam.min()  # resmin üstüne çizmek için normalize ediyoruz
        cam = cam / (cam.max() + 1e-8)  # küçük bir sayı ekleyerek sıfıra bölmeyi önler

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
    arch = ckpt.get("arch", "efficientnet_b0")

    # Build model
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
        raise ValueError(f"Unsupported architecture: {arch}")

    state_dict = ckpt["model_state"]
    fixed_state_dict = {}

    for k, v in state_dict.items():
        new_key = k.replace("classifier.1.1", "classifier.1")
        fixed_state_dict[new_key] = v

    model.load_state_dict(fixed_state_dict, strict=True)
    model.to(device)
    model.eval()

    # Grad-CAM için hedef katman olarak son convolutional bloğu seçiyoruz
    if hasattr(model, "features"):
        target_layer = model.features[-1] # gradcam katmanı efficientnet'te features bloğunun son katmanı olur
    elif hasattr(model, "layer4"): # resnet50'de layer4 bloğunun son katmanı olur
        target_layer = model.layer4
    else:
        raise ValueError("Grad-CAM target layer bulunamadı.")

    # Same preprocess as train/eval (no normalize)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_path = Path(args.image)
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    cam_engine = GradCAM(model, target_layer)

    # Modelin tahminini alıyoruz, ancak burada gradyanlara ihtiyacımız olmadığı için torch.no_grad kullanıyoruz. Tahmin edilen sınıf indeksini buluyoruz.
    with torch.no_grad():
        logits0 = model(x) # model resme bakıyor 
        pred_idx = int(logits0.argmax(dim=1).item()) # modelin tahmin ettiği sınıfın indeksini alıyoruz

    target_idx = pred_idx if args.class_idx is None else args.class_idx # model ne dediyse onu açıklıyor

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
    plt.imshow(cam_2d, alpha=0.45)  # heatmap'i orijinal resmin üstüne çiziyoruz, alpha ile saydamlık veriyoruz
    plt.title(f"Pred: {pred_name} ({pred_conf:.3f}) | Target CAM: {target_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print(" Saved:", out_path)

if __name__ == "__main__":
    main()