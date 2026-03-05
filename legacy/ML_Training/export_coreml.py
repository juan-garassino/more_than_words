import torch
import coremltools as ct
import argparse
import os
import json
from src.model import SymbolicTransformer
from src.tokenizer import SymbolicTokenizer

def export_model(model_path: str, output_path: str, vocab_path: str):
    print(f"Loading vocabulary from {vocab_path}...")
    if not os.path.exists(vocab_path):
        print(f"Error: Vocab file not found at {vocab_path}. Run training first!")
        return

    # Initialize tokenizer with the SAVED vocabulary
    tokenizer = SymbolicTokenizer(vocab_path=vocab_path)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    print(f"Loading model from {model_path}...")
    
    # Configure model to match trained dimensions
    model = SymbolicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256, # Must match train.py
        n_heads=4,
        n_layers=6,
        d_ff=1024,
        n_action_tokens=len(tokenizer.action_vocab),
        n_world_tokens=len(tokenizer.world_vocab),
        n_emotion_dims=len(tokenizer.emotion_dims)
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except FileNotFoundError:
        print("Model file not found.")
        return
    
    model.eval()
    
    # Trace inputs
    example_tokens = torch.randint(0, tokenizer.vocab_size, (1, 128)).long()
    example_types = torch.randint(0, 4, (1, 128)).long()
    
    traced_model = torch.jit.trace(model, (example_tokens, example_types))
    
    # Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="tokens", shape=(1, 128), dtype=np.int64),
            ct.TensorType(name="type_ids", shape=(1, 128), dtype=np.int64)
        ],
        outputs=[
            ct.TensorType(name="action_logits"),
            ct.TensorType(name="world_logits"),
            ct.TensorType(name="emotion_preds"),
            ct.TensorType(name="lambda_vals")
        ],
        minimum_deployment_target=ct.target.iOS16
    )
    
    # Add metadata
    mlmodel.author = "More Than Words AI"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Symbolic Transformer for Quest/Pet Simulation"
    mlmodel.user_defined_metadata["vocab_size"] = str(tokenizer.vocab_size)
    mlmodel.user_defined_metadata["action_vocab"] = ",".join(tokenizer.action_vocab)
    
    mlmodel.save(output_path)
    print(f"Saved Core ML model to {output_path}")

if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="tamagotchi_model.pth", help="Path to .pth checkpoint")
    parser.add_argument("--output_path", type=str, default="../SymbolicTamagotchiAI/TamagotchiTransformer.mlpackage", help="Output .mlpackage path")
    parser.add_argument("--vocab_path", type=str, default="ML_Training/vocab.json", help="Path to vocab.json")
    args = parser.parse_args()
    
    export_model(args.model_path, args.output_path, args.vocab_path)
