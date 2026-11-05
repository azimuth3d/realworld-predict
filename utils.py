import numpy as np
import pandas as pd
import torch
import os

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ETTm1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"

def load_ett_dataset(dataset_name="ETTh1 (Hourly)", sample_size=2000):
    """
    โหลดข้อมูลจริง ETDataset (Electricity Transformer Dataset)
    - ETTh1: ข้อมูลความถี่รายชั่วโมง (Hourly)
    - ETTm1: ข้อมูลความถี่ทุก 15 นาที (15-min)
    - Synthetic Data: ชุดข้อมูลจำลองความถี่ซับซ้อน (Synthetic)
    """
    if dataset_name == "ETTh1 (Hourly)":
        try:
            df = pd.read_csv(ETTH1_URL)
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'date': 'Timestamp', 'OT': 'Value'})
            df = df[['Timestamp', 'Value']].head(sample_size)
            return df, "ETTh1: Electricity Transformer Temperature (Hourly)"
        except Exception as e:
            print(f"Error loading ETTh1: {e}, falling back to Synthetic.")
            return generate_synthetic_data(sample_size), "Synthetic Data (Fallback)"

    elif dataset_name == "ETTm1 (15-min)":
        try:
            df = pd.read_csv(ETTm1_URL)
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'date': 'Timestamp', 'OT': 'Value'})
            df = df[['Timestamp', 'Value']].head(sample_size)
            return df, "ETTm1: Electricity Transformer Temperature (15-min)"
        except Exception as e:
            print(f"Error loading ETTm1: {e}, falling back to Synthetic.")
            return generate_synthetic_data(sample_size), "Synthetic Data (Fallback)"
            
    else:
        return generate_synthetic_data(sample_size), "Synthetic Multi-frequency Complex Data"

def generate_synthetic_data(n_samples=1500):
    """สร้างข้อมูลอนุกรมเวลาที่มีความซับซ้อนสูง (Multi-frequency + Non-linear Trend)"""
    np.random.seed(42)
    time = np.arange(n_samples)
    
    trend = 0.03 * time + 0.00005 * (time ** 2)
    season_daily = 12 * np.sin(2 * np.pi * time / 24)
    season_weekly = 8 * np.cos(2 * np.pi * time / (24 * 7))
    non_linear_wave = 5 * np.sin(2 * np.pi * time / 12) * np.cos(2 * np.pi * time / 48)
    noise = np.random.normal(0, 1.2, size=n_samples)
    
    data = 40 + trend + season_daily + season_weekly + non_linear_wave + noise
    
    df = pd.DataFrame({'Timestamp': pd.date_range(start='2026-01-01', periods=n_samples, freq='h'), 'Value': data})
    return df

def create_multi_step_sequences(data, seq_length=48, forecast_horizon=12):
    """แปลงข้อมูลเป็น Window Sequence สำหรับ Multi-step Forecasting"""
    xs, ys = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + forecast_horizon)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
