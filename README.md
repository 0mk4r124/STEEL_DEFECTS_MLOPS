# Steel Surface Defect Classification: an MLOps pipeline

A steel surface defect classifier (6 classes, NEU surface defect dataset) built end to end with the tools a production computer vision team uses: Docker, MLflow, ONNX and TensorRT, pytest and GitHub Actions.

> Work in progress: a 10-day build, 18 to 27 Nov 2026. Progress notes are in [LEARNING_LOG.md](LEARNING_LOG.md).

## Pipeline

1. **Train**: ResNet-18 in PyTorch, with every run tracked in MLflow
2. **Export**: PyTorch to ONNX, with a parity test against the original model
3. **Optimise**: TensorRT FP16 engine, benchmarked against ONNX Runtime
4. **Serve**: FastAPI `/predict` endpoint running in Docker
5. **CI**: GitHub Actions runs lint, tests and a Docker build on every push

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the NEU dataset in `data/` (it is git-ignored).

## Results

Benchmark table coming on 23 Nov.
