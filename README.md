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
