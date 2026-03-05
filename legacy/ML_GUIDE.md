# More Than Words - ML Training Pipeline

This document outlines the architecture and usage of the Machine Learning pipeline for **More Than Words** (Symbolic Tamagotchi AI).

## Overview

The system uses a **Symbolic Transformer** model to predict the creature's next action, the visible world state (modulated by ambiguity λ), and the emotional state. 

> **Note**: This engine is designed to evolve into a full **Generative Quest System**. See [VISION.md](VISION.md) for the roadmap and brainstorming on how this architecture extends beyond simple pet simulation.

Unlike LLMs that process text, this model processes discrete symbolic tokens representing game concepts (`HUNGRY`, `ENEMY`, `NORTH`). This allows for:
1.  **Ultra-lightweight inference** (runs locally on iPhone).
2.  **Perfect consistency** (no hallucinations of non-existent game mechanics).
3.  **Ambiguity control** via the λ parameter.

### Key Components

1.  **Tokenizer (`src/tokenizer.py`)**:
    - Maps symbolic tokens (e.g., `FEED`, `HUNGRY_H`, `LOC_KITCHEN`) to integer IDs.
    - Handles special tokens (`PAD`, `START`, `END`).

2.  **Simulator (`src/simulator.py`)**:
    - A Python implementation of the game logic.
    - Generates synthetic training data by playing millions of steps of "ideal" gameplay (based on rule-based heuristics).
    - Calculates the "Lambda" (λ) ambiguity value based on stress/sickness.

3.  **Dataset (`src/dataset.py`)**:
    - Converts simulator trajectories into PyTorch datasets.
    - Uses a causal language modeling approach (predict next token) but tailored for the triplet output (Action, World, Emotion).

4.  **Model (`src/model.py`)**:
    - **Architecture**: Transformer Encoder (GPT-style) but with custom heads.
    - **Inputs**: Sequence of Token IDs + Type IDs (Action/World/Emotion).
    - **Outputs**:
        - `action_logits`: Prediction of the next action.
        - `world_logits`: Multi-label prediction of visible world tokens.
        - `emotion_preds`: Regression of 5 emotional dimensions.
        - `lambda_vals`: Scalar prediction of current ambiguity.

## Usage Guide

### 1. Training the Model

The `train.py` script handles data generation and training in one go.

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py
```

**Configuration**:
- You can adjust `num_epochs`, `batch_size`, and model size in `train.py`.
- The script automatically saves checkpoints to `tamagotchi_model.pth`.

### 2. Exporting to Core ML

To use the model in the iOS app, it must be converted to `.mlpackage` format.

```bash
python export_coreml.py --model_path tamagotchi_model.pth --output_path TamagotchiTransformer.mlpackage
```

This script:
1.  Loads the PyTorch model.
2.  Traces it with JIT (Just-In-Time) compilation.
3.  Converts it using `coremltools`.
4.  Sets the appropriate input/output names and types (`Int64` for tokens).

### 3. Integration with iOS

1.  Copy `TamagotchiTransformer.mlpackage` into your Xcode project.
2.  Ensure it is added to the "Compile Sources" build phase.
3.   The app's `ModelHandler.swift` will interface with this model using the auto-generated Swift class.

## Project Structure

```
ML_Training/
├── src/
│   ├── tokenizer.py    # Vocabulary management
│   ├── simulator.py    # Game logic & data gen
│   ├── dataset.py      # PyTorch data loading
│   └── model.py        # PyTorch Transformer definition
├── boilerplates/       # Original reference files
├── train.py            # Main training script
├── export_coreml.py    # Core ML converter
└── requirements.txt    # Python dependencies
```
