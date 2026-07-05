# PulseEdge

Real-time vibration anomaly detection for predictive maintenance of industrial heavy machinery, using an LSTM autoencoder deployed entirely at the edge.

## Problem

Heavy machinery fails without warning, causing costly unplanned downtime. Cloud-based monitoring introduces latency and requires constant connectivity. PulseEdge runs inference on-device, flagging anomalies in real time with no cloud dependency.

## Approach

An LSTM autoencoder is trained only on vibration data from healthy machine operation. Reconstruction error spikes on abnormal vibration signatures (bearing wear, imbalance, misalignment), flagging anomalies. The model is quantized and deployed on low-cost edge hardware.

## Dataset

CWRU Bearing Dataset (Case Western Reserve University).

## Structure

- `scripts/download_dataset.py` — fetches and extracts the dataset
- `src/data_loader.py` — loads raw vibration signals
- `src/preprocess.py` — windowing, normalization, feature extraction
- `src/model.py` — LSTM autoencoder architecture
- `src/train.py` — training loop
- `src/evaluate.py` — reconstruction error thresholding, metrics
- `src/realtime_inference.py` — streaming inference for edge deployment

## Setup

pip install -r requirements.txt

## Usage

python scripts/download_dataset.py
python src/train.py
python src/evaluate.py
