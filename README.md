# Cascaded Stealth-Aware ATT&CK Path Prediction

Code and data accompanying the paper:

> **Cascaded Stealth-Aware ATT&CK Path Prediction**

## Structure

```
src/chain_paths/         Core library (models, preprocessing, evaluation, inference)
  models/                LSTM, BiLSTM, GRU, TCN, Transformer, HMM, N-gram
  trainers/              Training loops
configs/                 YAML configuration files
data/                    Preprocessed corpora and ontology mappings
outputs/                 Trained model weights (lstm_weights.pt) and embeddings
sigma-rules/             Sigma detection rules for stealth scoring
tests/                   Unit and integration tests
train_lstm.py            Entry point — train the LSTM
evaluate_lstm.py         Entry point — evaluate a trained checkpoint
sightings_v2_public.csv  MITRE ATT&CK Sightings v2 corpus
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Training

```bash
python train_lstm.py
```

## Evaluation (prefix-disjoint protocol)

```bash
python evaluate_lstm.py
```

## Tests

```bash
pip install pytest
pytest tests/
```

## License

See LICENSE file.
