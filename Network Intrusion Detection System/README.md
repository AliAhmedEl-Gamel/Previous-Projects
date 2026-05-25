# Network Intrusion Detection System

**Model:** [Alielgamal22/Network-Intrusion-Detection-System](https://huggingface.co/Alielgamal22/Network-Intrusion-Detection-System)

A machine learning system that classifies network traffic as benign or identifies the specific type of cyber attack, achieving **98% accuracy** using an XGBoost classifier trained on the CICIDS 2017 dataset.

---

## Overview

This project builds a multi-class network traffic classifier that can detect and categorize network intrusions in real time. The model is trained on real-world network flow data and can distinguish between normal traffic and several types of attacks.

---

## Attack Classes

| Label | Description |
|-------|-------------|
| `BENIGN` | Normal network traffic |
| `DDos` | Distributed Denial of Service (DoS Hulk, GoldenEye, Slowloris, Slowhttptest, DDoS) |
| `PortScan` | Port scanning activity |
| `Brute Force` | Web-based brute force attacks |
| `FTP-Patator` | FTP brute force attacks |

---

## Model Architecture

- **Algorithm:** XGBoost (`multi:softmax`)
- **Framework:** Scikit-learn Pipeline + XGBoost
- **Preprocessing:** StandardScaler
- **Encoding:** LabelEncoder for target classes

### Pipeline

```
Raw Features → StandardScaler → XGBClassifier → Predicted Attack Type
```

### Training Details

- **Train/Test Split:** 80% / 20% (stratified)
- **Class Imbalance:** Handled using balanced sample weights
- **Validation:** 3-fold Stratified Cross-Validation
- **Evaluation Metrics:** Accuracy, F1 Weighted, F1 Macro

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **98%** |
| Validation | 3-Fold Stratified CV |
| Objective | `multi:softmax` |

---

## Dataset

**CICIDS 2017** (Canadian Institute for Cybersecurity Intrusion Detection Dataset)
- Source: [University of New Brunswick](https://www.unb.ca/cic/datasets/ids-2017.html)
- Contains realistic network traffic with labeled attack types
- Multiple CSV files merged into a single dataset

### Preprocessing Steps

- Dropped rows with `NaN` values
- Removed infinite values in `Flow Bytes/s` and `Flow Packets/s`
- Dropped duplicate rows
- Consolidated similar attack types (e.g. all DoS variants → `DDos`)
- Dropped redundant column `Fwd Header Length.1`

---

## Getting Started

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib joblib
```

### Dataset Setup

1. Download the CICIDS 2017 dataset from [UNB](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Place all CSV files in a folder:

```
data/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── ...
```

3. Update the file path in the notebook:

```python
files = glob.glob("data/*.csv")
```

### Training

Open the notebook and run all cells in order. The notebook will:

1. Load and merge all CSV files
2. Clean and preprocess the data
3. Encode labels and split into train/test sets
4. Run 3-fold cross-validation
5. Train the final XGBoost pipeline
6. Evaluate on the test set and display the confusion matrix
7. Save the model as `model.pkl`

### Loading the Saved Model

```python
import joblib

artifacts = joblib.load("model.pkl")
pipeline = artifacts["model1"]
le = artifacts["label_encoder"]
feature_names = artifacts["feature_names"]

# Predict on new data
import pandas as pd
sample = pd.DataFrame([your_network_flow_data], columns=feature_names)
prediction = pipeline.predict(sample)
print(le.inverse_transform(prediction))
```

### Loading from HuggingFace

```python
from huggingface_hub import hf_hub_download
import joblib

path = hf_hub_download(
    repo_id="Alielgamal22/Network-Intrusion-Detection-System",
    filename="model.pkl"
)
artifacts = joblib.load(path)
```

---

## Tech Stack

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- Joblib

---

## License

This project is for educational purposes. The CICIDS 2017 dataset is provided by the Canadian Institute for Cybersecurity.
