import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress
import time
import os
import glob
import json

from src.tokenizer import SymbolicTokenizer
from src.simulator import TamagotchiSimulator
from src.dataset import TamagotchiDataset, collate_fn
from src.model import SymbolicTransformer

def load_data_from_jsonl(file_paths, tokenizer):
    """Load trajectories from JSONL files"""
    trajectories = []
    from src.tokenizer import Token
    
    for path in file_paths:
        if not os.path.exists(path):
            continue
            
        with open(path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    traj = []
                    steps = entry.get("steps", [])
                    for step in steps:
                        # Parse action
                        action = Token(step["action"], "ACTION")
                        
                        # Parse world
                        world_tokens = [Token(w, "WORLD") for w in step.get("world", [])]
                        
                        # Parse emotions (ensure order matches dims)
                        emotions_dict = step.get("emotions", {})
                        emotions_vec = [emotions_dict.get(dim, 0.0) for dim in tokenizer.emotion_dims]
                        
                        traj.append((action, world_tokens, emotions_vec))
                    
                    if traj:
                        trajectories.append(traj)
                except json.JSONDecodeError:
                    continue
    return trajectories

def train_model(
    model: SymbolicTransformer,
    train_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = "trained_model.pth"
):
    """Training loop with multi-objective loss"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    console = Console()
    console.print(f"[yellow]Starting training on {device}...[/yellow]")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        action_acc = 0
        
        for batch_idx, batch in enumerate(train_loader):
            tokens = batch['tokens'].to(device)
            type_ids = batch['type_ids'].to(device)
            target_action = batch['target_action'].to(device)
            target_world = batch['target_world'].to(device)
            target_emotions = batch['target_emotions'].to(device)

            # Forward pass
            action_logits, world_logits, emotion_preds, lambda_vals = model(tokens, type_ids)

            # Losses
            loss_action = F.cross_entropy(action_logits, target_action)
            loss_world = F.binary_cross_entropy_with_logits(world_logits, target_world)
            loss_emotion = F.mse_loss(emotion_preds, target_emotions)

            # Total loss
            loss = loss_action + 1.5 * loss_world + 0.8 * loss_emotion

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            batch_acc = (action_logits.argmax(dim=-1) == target_action).float().mean().item()
            action_acc += batch_acc
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: Loss {loss.item():.4f} Acc {batch_acc:.3f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_acc = action_acc / len(train_loader)
        console.print(f"[bold green]Epoch {epoch+1} Complete: Loss={avg_loss:.4f}, Action Acc={avg_acc:.3f}[/bold green]")
        
        # Save checkpoint per epoch
        torch.save(model.state_dict(), save_path)

    console.print(f"[bold green]Training Complete! Model saved to {save_path}[/bold green]")
    return model

def main():
    console = Console()
    
    # 1. Initialize Tokenizer & Auto-Build Vocabulary
    console.print("Initializing Tokenizer...")
    tokenizer = SymbolicTokenizer()
    
    # Locate all JSONL files in data folder
    data_files = glob.glob("ML_Training/data/*.jsonl")
    if data_files:
        console.print(f"Found data files: {data_files}")
        console.print("Scanning files to build vocabulary...")
        tokenizer.fit_on_instruction_data(data_files)
        
        # SAVE VOCABULARY! Critical for export
        tokenizer.save_vocab("ML_Training/vocab.json")
    else:
        console.print("[red]No data files found in ML_Training/data/![/red]")
        return
    
    # 2. Load Data
    console.print("Loading Training Data...")
    trajectories = load_data_from_jsonl(data_files, tokenizer)
    
    if not trajectories:
        console.print("[red]No valid trajectories found![/red]")
        return
        
    dataset = TamagotchiDataset(trajectories, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # 3. Model
    console.print(f"Initializing Model with Vocab Size: {tokenizer.vocab_size}")
    model = SymbolicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=6,
        d_ff=1024,
        n_action_tokens=len(tokenizer.action_vocab),
        n_world_tokens=len(tokenizer.world_vocab),
        n_emotion_dims=len(tokenizer.emotion_dims)
    )
    
    # 4. Train
    train_model(model, loader, num_epochs=20, save_path="tamagotchi_model.pth")
    
if __name__ == "__main__":
    main()
