# RobustDefect-LLM

RobustDefect-LLM is an explainable and robustness-aware industrial surface defect inspection framework developed using the NEU-DET steel surface defect dataset.

## Main Components

- Transfer-learning-based defect classification
- ResNet50, EfficientNet-B0, DenseNet121, and MobileNetV3-Large
- Grad-CAM visual explanations
- Confidence and Top-2 margin analysis
- HUMAN REVIEW, ACCEPT, REWORK, and REJECT decision pathways
- Controlled LLM-assisted inspection reporting
- FastAPI backend and MongoDB storage
- React Native / Expo mobile application

## Dataset

The experiments use 1,799 available images from the NEU Surface Defect Database, covering six categories:

- crazing
- inclusion
- patches
- pitted surface
- rolled-in scale
- scratches

Experimental split:

- Training: 1,259 images
- Validation: 270 images
- Test: 270 images
- Random seed: 42

The NEU-DET images are not redistributed in this repository.

## Test Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| MobileNetV3-Large | 0.9926 | 0.9927 |
| DenseNet121 | 0.9889 | 0.9893 |
| EfficientNet-B0 | 0.9519 | 0.9516 |
| ResNet50 | 0.7963 | 0.7324 |

MobileNetV3-Large was selected as the primary model based on its numerically highest test accuracy and low computational cost.

## Included Evaluation Scripts

The repository includes scripts for:

- model training and independent test evaluation
- bootstrap confidence intervals
- exact McNemar testing
- confidence and calibration analysis
- synthetic corruption stress testing
- Grad-CAM visualization
- model complexity analysis
- CPU inference benchmarking
- controlled LLM report validation
- dataset leakage checking

## Installation

Create and activate a virtual environment:

`python -m venv venv`

`source venv/bin/activate`

Install the dependencies:

`pip install -r requirements.txt`

## Example Commands

Train MobileNetV3-Large:

`python src/train_mobilenetv3.py`

Evaluate the models:

`python src/evaluate_models_with_densenet.py`

Calculate bootstrap confidence intervals:

`python src/evaluate_models_with_bootstrap.py`

Run the McNemar test:

`python src/mcnemar_test.py`

Run robustness analysis:

`python src/stress_test.py`

Run controlled LLM validation:

`python src/evaluate_llm.py`

## Data and Model Files

Dataset images, generated uploads, trained model checkpoints, environment files, and API credentials are excluded from the repository.

## Mobile Application

The React Native / Expo mobile application is available at:

https://github.com/nazlican530/mobile-app

## Authors

Nazlıcan Düşünmez  
Prof. Dr. Halûk Gümüşkaya  
Department of Computer Engineering, İstanbul Arel University

## Environment Variables

Copy `.env.example` to `.env` and configure `GROQ_API_KEY`, `SECRET_KEY`, and `MONGODB_URI` before running the backend.
