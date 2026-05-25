from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model
artifacts = joblib.load("model.pkl")
pipeline = artifacts["model1"]
le = artifacts["label_encoder"]
feature_names = artifacts["feature_names"]

app = FastAPI(
    title="Attack Classifier API",
    description="Classifies network flow into attack types using XGBoost",
    version="1.0.0"
)

# Input schema — one field per feature
class FlowData(BaseModel):
    model_config = {"populate_by_name": True}

    Destination_Port: float
    Flow_Duration: float
    Total_Fwd_Packets: float
    Total_Backward_Packets: float
    Total_Length_of_Fwd_Packets: float
    Total_Length_of_Bwd_Packets: float
    Fwd_Packet_Length_Max: float
    Fwd_Packet_Length_Min: float
    Fwd_Packet_Length_Mean: float
    Fwd_Packet_Length_Std: float
    Bwd_Packet_Length_Max: float
    Bwd_Packet_Length_Min: float
    Bwd_Packet_Length_Mean: float
    Bwd_Packet_Length_Std: float
    Flow_Bytes_per_s: float
    Flow_Packets_per_s: float
    Flow_IAT_Mean: float
    Flow_IAT_Std: float
    Flow_IAT_Max: float
    Flow_IAT_Min: float
    Fwd_IAT_Total: float
    Fwd_IAT_Mean: float
    Fwd_IAT_Std: float
    Fwd_IAT_Max: float
    Fwd_IAT_Min: float
    Bwd_IAT_Total: float
    Bwd_IAT_Mean: float
    Bwd_IAT_Std: float
    Bwd_IAT_Max: float
    Bwd_IAT_Min: float
    Fwd_PSH_Flags: float
    Bwd_PSH_Flags: float
    Fwd_URG_Flags: float
    Bwd_URG_Flags: float
    Fwd_Header_Length: float
    Bwd_Header_Length: float
    Fwd_Packets_per_s: float
    Bwd_Packets_per_s: float
    Min_Packet_Length: float
    Max_Packet_Length: float
    Packet_Length_Mean: float
    Packet_Length_Std: float
    Packet_Length_Variance: float
    FIN_Flag_Count: float
    SYN_Flag_Count: float
    RST_Flag_Count: float
    PSH_Flag_Count: float
    ACK_Flag_Count: float
    URG_Flag_Count: float
    CWE_Flag_Count: float
    ECE_Flag_Count: float
    Down_Up_Ratio: float
    Average_Packet_Size: float
    Avg_Fwd_Segment_Size: float
    Avg_Bwd_Segment_Size: float
    Fwd_Avg_Bytes_Bulk: float
    Fwd_Avg_Packets_Bulk: float
    Fwd_Avg_Bulk_Rate: float
    Bwd_Avg_Bytes_Bulk: float
    Bwd_Avg_Packets_Bulk: float
    Bwd_Avg_Bulk_Rate: float
    Subflow_Fwd_Packets: float
    Subflow_Fwd_Bytes: float
    Subflow_Bwd_Packets: float
    Subflow_Bwd_Bytes: float
    Init_Win_bytes_forward: float
    Init_Win_bytes_backward: float
    act_data_pkt_fwd: float
    min_seg_size_forward: float
    Active_Mean: float
    Active_Std: float
    Active_Max: float
    Active_Min: float
    Idle_Mean: float
    Idle_Std: float
    Idle_Max: float
    Idle_Min: float


def input_to_dataframe(flow: FlowData) -> pd.DataFrame:
    """Map pydantic fields back to original feature names the model expects."""
    mapping = {
        "Destination_Port": "Destination Port",
        "Flow_Duration": "Flow Duration",
        "Total_Fwd_Packets": "Total Fwd Packets",
        "Total_Backward_Packets": "Total Backward Packets",
        "Total_Length_of_Fwd_Packets": "Total Length of Fwd Packets",
        "Total_Length_of_Bwd_Packets": "Total Length of Bwd Packets",
        "Fwd_Packet_Length_Max": "Fwd Packet Length Max",
        "Fwd_Packet_Length_Min": "Fwd Packet Length Min",
        "Fwd_Packet_Length_Mean": "Fwd Packet Length Mean",
        "Fwd_Packet_Length_Std": "Fwd Packet Length Std",
        "Bwd_Packet_Length_Max": "Bwd Packet Length Max",
        "Bwd_Packet_Length_Min": "Bwd Packet Length Min",
        "Bwd_Packet_Length_Mean": "Bwd Packet Length Mean",
        "Bwd_Packet_Length_Std": "Bwd Packet Length Std",
        "Flow_Bytes_per_s": "Flow Bytes/s",
        "Flow_Packets_per_s": "Flow Packets/s",
        "Flow_IAT_Mean": "Flow IAT Mean",
        "Flow_IAT_Std": "Flow IAT Std",
        "Flow_IAT_Max": "Flow IAT Max",
        "Flow_IAT_Min": "Flow IAT Min",
        "Fwd_IAT_Total": "Fwd IAT Total",
        "Fwd_IAT_Mean": "Fwd IAT Mean",
        "Fwd_IAT_Std": "Fwd IAT Std",
        "Fwd_IAT_Max": "Fwd IAT Max",
        "Fwd_IAT_Min": "Fwd IAT Min",
        "Bwd_IAT_Total": "Bwd IAT Total",
        "Bwd_IAT_Mean": "Bwd IAT Mean",
        "Bwd_IAT_Std": "Bwd IAT Std",
        "Bwd_IAT_Max": "Bwd IAT Max",
        "Bwd_IAT_Min": "Bwd IAT Min",
        "Fwd_PSH_Flags": "Fwd PSH Flags",
        "Bwd_PSH_Flags": "Bwd PSH Flags",
        "Fwd_URG_Flags": "Fwd URG Flags",
        "Bwd_URG_Flags": "Bwd URG Flags",
        "Fwd_Header_Length": "Fwd Header Length",
        "Bwd_Header_Length": "Bwd Header Length",
        "Fwd_Packets_per_s": "Fwd Packets/s",
        "Bwd_Packets_per_s": "Bwd Packets/s",
        "Min_Packet_Length": "Min Packet Length",
        "Max_Packet_Length": "Max Packet Length",
        "Packet_Length_Mean": "Packet Length Mean",
        "Packet_Length_Std": "Packet Length Std",
        "Packet_Length_Variance": "Packet Length Variance",
        "FIN_Flag_Count": "FIN Flag Count",
        "SYN_Flag_Count": "SYN Flag Count",
        "RST_Flag_Count": "RST Flag Count",
        "PSH_Flag_Count": "PSH Flag Count",
        "ACK_Flag_Count": "ACK Flag Count",
        "URG_Flag_Count": "URG Flag Count",
        "CWE_Flag_Count": "CWE Flag Count",
        "ECE_Flag_Count": "ECE Flag Count",
        "Down_Up_Ratio": "Down/Up Ratio",
        "Average_Packet_Size": "Average Packet Size",
        "Avg_Fwd_Segment_Size": "Avg Fwd Segment Size",
        "Avg_Bwd_Segment_Size": "Avg Bwd Segment Size",
        "Fwd_Avg_Bytes_Bulk": "Fwd Avg Bytes/Bulk",
        "Fwd_Avg_Packets_Bulk": "Fwd Avg Packets/Bulk",
        "Fwd_Avg_Bulk_Rate": "Fwd Avg Bulk Rate",
        "Bwd_Avg_Bytes_Bulk": "Bwd Avg Bytes/Bulk",
        "Bwd_Avg_Packets_Bulk": "Bwd Avg Packets/Bulk",
        "Bwd_Avg_Bulk_Rate": "Bwd Avg Bulk Rate",
        "Subflow_Fwd_Packets": "Subflow Fwd Packets",
        "Subflow_Fwd_Bytes": "Subflow Fwd Bytes",
        "Subflow_Bwd_Packets": "Subflow Bwd Packets",
        "Subflow_Bwd_Bytes": "Subflow Bwd Bytes",
        "Init_Win_bytes_forward": "Init_Win_bytes_forward",
        "Init_Win_bytes_backward": "Init_Win_bytes_backward",
        "act_data_pkt_fwd": "act_data_pkt_fwd",
        "min_seg_size_forward": "min_seg_size_forward",
        "Active_Mean": "Active Mean",
        "Active_Std": "Active Std",
        "Active_Max": "Active Max",
        "Active_Min": "Active Min",
        "Idle_Mean": "Idle Mean",
        "Idle_Std": "Idle Std",
        "Idle_Max": "Idle Max",
        "Idle_Min": "Idle Min",
    }

    data = flow.model_dump()
    renamed = {mapping[k]: v for k, v in data.items()}
    df = pd.DataFrame([renamed])

    # Ensure column order matches training
    df = df[feature_names]
    return df


@app.get("/")
def root():
    return {"message": "Attack Classifier API is running"}


@app.get("/classes")
def get_classes():
    """Returns the list of possible attack classes."""
    return {"classes": le.classes_.tolist()}


@app.post("/predict")
def predict(flow: FlowData):
    try:
        df = input_to_dataframe(flow)

        # Replace inf values just in case
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        prediction = pipeline.predict(df)
        label = le.inverse_transform(prediction)[0]

        return {
            "prediction": label,
            "is_attack": label != "BENIGN"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
