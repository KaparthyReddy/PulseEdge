# PulseEdge

Real-time vibration anomaly detection for predictive maintenance of industrial heavy machinery, using an LSTM autoencoder deployed entirely at the edge.

**Author:** Kaparthy Reddy
**Submission:** Tata Technologies InnoVent — Edge AI for Autonomous & Intelligent Heavy Machines

## Problem

Heavy machinery fails without warning, causing costly unplanned downtime. Cloud-based monitoring introduces latency and requires constant connectivity. PulseEdge runs inference on-device, flagging anomalies in real time with no cloud dependency.

## Approach

An LSTM autoencoder is trained only on vibration data from healthy machine operation. Reconstruction error spikes on abnormal vibration signatures (bearing wear, imbalance, misalignment), flagging anomalies. The model is deployed on low-cost edge hardware for real-time, on-device inference.

## Results

Trained on the CWRU Bearing Dataset (normal baseline + inner race, ball, and outer race faults):

| Metric | Value |
|---|---|
| Training loss (start → converged) | 1.00 → 0.05 |
| Anomaly detection rate (ball fault) | 100% |
| Anomaly detection rate (inner race fault) | 100% |
| Anomaly detection rate (outer race fault) | 100% |
| Reconstruction error threshold | 0.135 |
| On-device inference latency | ~1 ms per window |

See `results/error_distribution.png` for the reconstruction error distribution plot separating normal vs. faulty vibration signatures.

## Dataset

[CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter) (Case Western Reserve University) — normal baseline plus inner race, ball, and outer race fault signals.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 scripts/download_dataset.py
python3 src/train.py
python3 src/evaluate.py
```

## Architecture

The `LSTMAutoencoder` encodes a 64-timestep vibration window into a 16-dimensional latent vector via an LSTM + linear layer, then decodes it back through a second LSTM + linear output layer. Reconstruction error (MSE) on unseen windows is used as the anomaly score — high error indicates a pattern the model never saw during training on healthy data.

## Real-Time Inference

`LSTMAnomalyDetectorRealtime` in `src/realtime_inference.py` maintains a sliding buffer of incoming sensor readings, runs inference once the buffer fills a 64-sample window, and returns the reconstruction error, anomaly flag, and inference latency per call — ready for deployment on Raspberry Pi / Jetson Nano hardware.

## License

For academic and competition submission purposes.
