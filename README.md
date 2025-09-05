# End-to-End-ML-Application-for-Cyber_Bullying_Detection
# Cyberbullying Detection Web Application

## Overview
This project is a **real-time cyberbullying detection web application** built using **FastAPI**. It classifies user comments into three categories:

- **Normal**
- **Offensive**
- **Hate Speech**

The app leverages **state-of-the-art NLP models**, including **DistilBERT**, **partial fine-tuning**, and **LoRA adapters**, for accurate predictions. It is designed to be lightweight, modular, and production-ready.

---

## Features

- **Fast and efficient inference** for single or batch comments.
- **Multiple model support**:
  - Full fine-tuned DistilBERT
  - Partial fine-tuning (classifier + last transformer layer)
  - LoRA-enhanced DistilBERT for parameter-efficient training
- **Web interface** for easy comment input and prediction display.
- **Artifact-based workflow**:
  - Train, validation, and test data stored in `artifacts/`
  - Model checkpoints stored for quick loading and inference
- **Custom pipeline** to handle preprocessing, tokenization, and model inference.

---

## Project Structure
CyberBullying/
├─ app.py # FastAPI app
├─ requirements.txt # Python dependencies
├─ src/
│ ├─ components/
│ │ ├─ data_ingestion.py
│ │ ├─ data_transformation.py
│ │ └─ model_trainer.py
│ ├─ pipeline/
│ │ └─ predict_pipeline.py
│ ├─ logger.py
│ └─ exception.py
├─ templates/
│ └─ index.html # Front-end template
├─ artifacts/
│ ├─ train.csv
│ ├─ test.csv
│ └─ models/ # Saved model checkpoints
└─ README.md


---
## Installation

To get started, first clone the repository with `git clone <repo_url>` and navigate into the project folder using `cd CyberBullying`. Next, create and activate a virtual environment with `conda create -n cyberbully python=3.10 -y` followed by `conda activate cyberbully`. Once the environment is ready, install all the required dependencies by running `pip install -r requirements.txt`. Finally, make sure the `artifacts` directory contains the preprocessed datasets (`train.csv` and `test.csv`) as well as the saved model checkpoints for the fine-tuned DistilBERT models (full, partial, and LoRA).  




Pipeline

Data Ingestion:

Loads raw CSV data

Splits into train, validation, and test sets

Saves processed data in artifacts/

Data Transformation:

Tokenizes comments

Maps labels to integers

Prepares Hugging Face DatasetDict

Model Training (optional):

Full fine-tuning of DistilBERT

Partial fine-tuning (classifier + last transformer layer)

LoRA adapters for parameter-efficient fine-tuning

Saves trained models in artifacts/models/

Prediction Pipeline:

Loads the desired checkpoint

Performs tokenization

Returns the predicted label



