import torch
import numpy as np
import cv2


def generate_gradcam(model, input_tensor, target_layer, original_image_path):

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    pred = output.argmax()

    model.zero_grad()
    output[0, pred].backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(str(original_image_path))
    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    handle_f.remove()
    handle_b.remove()

    return overlay