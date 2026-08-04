# RobustDefect-LLM

RobustDefect-LLM is an explainable and robustness-aware industrial surface defect inspection framework developed using the NEU Surface Defect Database (NEU-DET). The framework combines deep learning-based defect classification, explainable AI (Grad-CAM), confidence-aware decision support, controlled LLM-assisted reporting, and a mobile inspection workflow.

This repository accompanies the paper:

**RobustDefect-LLM: Explainable and Robustness-Aware Industrial Surface Defect Classification with Decision Support and AI-Assisted Reporting**

---

# Main Components

- Transfer learning using multiple CNN architectures
- ResNet50
- EfficientNet-B0
- DenseNet121
- MobileNetV3-Large
- Grad-CAM visual explanations
- Confidence-aware decision support
- Top-2 confidence margin analysis
- HUMAN REVIEW, ACCEPT, REWORK, and REJECT decision pathways
- Controlled LLM-assisted inspection reporting
- FastAPI backend
- MongoDB database
- React Native / Expo mobile application

---

# Reproducibility Information

This repository corresponds to the experiments reported in the accompanying manuscript.

Repository version

**Commit**

```
315b5ad51781e46388508f866d3ea4e6f5828999
```

Experimental environment

- Python 3.14.3
- PyTorch 2.10.0
- macOS 15.5
- Apple M1 processor
- 8 GB unified memory
- Random seed: 42

The reported training times correspond to complete training and validation runs performed on the above hardware.

---

# Dataset

The experiments use the NEU Surface Defect Database (NEU-DET).

The dataset contains six defect categories:

- crazing
- inclusion
- patches
- pitted surface
- rolled-in scale
- scratches

The experiments reported in the paper use a fixed split consisting of:

| Split | Images |
|-------|-------:|
| Training | 1259 |
| Validation | 270 |
| Test | 270 |
| Total | 1799 |

The original dataset is **not redistributed** in this repository because of licensing restrictions.

After downloading the dataset, place it under

```
data/NEU-DET/
```

The experiments reported in the manuscript use the fixed split located in

```
data/NEU-DET/split_1799/
```

---

# Test Results

| Model | Accuracy | Macro F1 |
|------|---------:|---------:|
| MobileNetV3-Large | 0.9926 | 0.9927 |
| DenseNet121 | 0.9889 | 0.9893 |
| EfficientNet-B0 | 0.9519 | 0.9516 |
| ResNet50 | 0.7963 | 0.7324 |

MobileNetV3-Large was selected as the deployment model following the study's validation-guided selection procedure and subsequently achieved the highest performance on the held-out test partition while maintaining low computational cost.

---

# Included Evaluation Scripts

The repository includes implementations for

- model training
- model evaluation
- bootstrap confidence intervals
- exact McNemar testing
- confidence analysis
- Top-2 margin analysis
- synthetic corruption robustness testing
- Grad-CAM visualization
- model complexity analysis
- CPU inference benchmarking
- controlled LLM validation
- dataset leakage checking

---

# Installation

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

For complete reproducibility, the exact software environment can also be recreated using

```bash
pip install -r requirements-lock.txt
```

---

# Example Commands

Train MobileNetV3-Large

```bash
python src/train_mobilenetv3.py
```

Evaluate the trained models

```bash
python src/evaluate_models_with_densenet.py
```

Calculate bootstrap confidence intervals

```bash
python src/evaluate_models_with_bootstrap.py
```

Run the exact McNemar test

```bash
python src/mcnemar_test.py
```

Run robustness analysis

```bash
python src/stress_test.py
```

Run controlled LLM validation

```bash
python src/evaluate_llm.py
```

---

# Expected Outputs

Running the scripts generates results under the

```
outputs/
```

directory, including

- trained checkpoints
- training logs
- confusion matrices
- accuracy curves
- loss curves
- confidence analysis
- confidence margin analysis
- review coverage analysis
- Grad-CAM visualizations
- bootstrap evaluation results
- stress test outputs

---

# Model Files

The repository contains trained checkpoints under

```
outputs/models/
```

including

- best_model_mobilenetv3.pt
- best_model_densenet121.pt
- resnet50_best_1800.pt

---

# Model Checksums

The SHA-256 hashes below can be used to verify checkpoint integrity.

| Checkpoint | SHA-256 |
|------------|----------|
| best_model_mobilenetv3.pt | 542869e97800bb7357388bb4d61737ed20bf75fd1f4d412d6cb1c5ddd3d705aa |
| best_model_densenet121.pt | a0783381aef0ca6f61a45b52272b981c0b7b9cbe8c00dcc609e1ac4459c489bd |
| resnet50_best_1800.pt | fbfceaa9ba8fb01ba63a4ef66d8b088bd53b4883bf1d6e30a1ec3ab679d7fa9f |

---

# Environment Variables

Create a `.env` file (or copy `.env.example`) and configure

- GROQ_API_KEY
- SECRET_KEY
- MONGODB_URI

before running the FastAPI backend.

---

# Mobile Application

The React Native / Expo mobile application is available at

https://github.com/nazlican530/mobile-app

---

# Authors

Nazlıcan Düşünmez

Prof. Dr. Halûk Gümüşkaya

Department of Computer Engineering

İstanbul Arel University

---

# Citation

If you use this repository in your research, please cite the associated RobustDefect-LLM publication.

---

# License

This repository is distributed under the license provided in the LICENSE file.