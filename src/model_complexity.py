import os
import torch
import torch.nn as nn
from thop import profile
from torchvision import models


NUM_CLASSES = 6
INPUT_SIZE = 224


def create_efficientnet_b0():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )
    return model


def create_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )
    return model


def create_densenet121():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        NUM_CLASSES
    )
    return model


def create_mobilenet_v3_large():
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        NUM_CLASSES
    )
    return model


def count_parameters(model):
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def calculate_model_size(model):
    temporary_path = "temporary_model_size.pt"

    torch.save(model.state_dict(), temporary_path)
    size_mb = os.path.getsize(temporary_path) / (1024 ** 2)

    os.remove(temporary_path)

    return size_mb


def analyze_model(model_name, model):
    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        INPUT_SIZE,
        INPUT_SIZE
    )

    with torch.no_grad():
        macs, _ = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

    # Many papers report one MAC as approximately two FLOPs.
    flops = 2 * macs

    total_parameters, trainable_parameters = count_parameters(model)
    model_size_mb = calculate_model_size(model)

    return {
        "model": model_name,
        "parameters_m": total_parameters / 1e6,
        "trainable_parameters_m": trainable_parameters / 1e6,
        "macs_g": macs / 1e9,
        "flops_g": flops / 1e9,
        "model_size_mb": model_size_mb,
    }


def main():
    models_to_analyze = {
        "EfficientNet-B0": create_efficientnet_b0(),
        "ResNet50": create_resnet50(),
        "DenseNet121": create_densenet121(),
        "MobileNetV3-Large": create_mobilenet_v3_large(),
    }

    results = []

    for model_name, model in models_to_analyze.items():
        print(f"Analyzing {model_name}...")
        result = analyze_model(model_name, model)
        results.append(result)

    print("\n" + "=" * 105)
    print(
        f"{'Model':<22}"
        f"{'Parameters (M)':>16}"
        f"{'Trainable (M)':>16}"
        f"{'MACs (G)':>13}"
        f"{'FLOPs (G)':>13}"
        f"{'Size (MB)':>13}"
    )
    print("=" * 105)

    for result in results:
        print(
            f"{result['model']:<22}"
            f"{result['parameters_m']:>16.3f}"
            f"{result['trainable_parameters_m']:>16.3f}"
            f"{result['macs_g']:>13.3f}"
            f"{result['flops_g']:>13.3f}"
            f"{result['model_size_mb']:>13.2f}"
        )

    print("=" * 105)


if __name__ == "__main__":
    main()