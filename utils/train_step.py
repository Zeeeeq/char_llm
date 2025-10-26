import time
import torch
import numpy as np
from utils.data_loader import get_batch

def train_model(
    model,
    train_data,
    val_data,
    optimizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hyperparams: dict = {
        "batch_size": 128,
        "seq_len": 64,
        "niter": 1000,
    }
):
    """
    Train a given model on the provided training and validation loaders.

    Args:
        model: torch.nn.Module — model instance to train and evaluate.
        get_batch: function for getting training and validation batches.
        optimizer: torch.optim — optimizer to use.
        device: str — 'cuda' or 'cpu'.
        print_every: int — how often to print progress.

    Returns:
        dict with training/validation losses and accuracies.
    """

    model = model.to(device)
    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_time": [],
        "val_time": []
    }

    # Initialize print_every based on niter. ideally, larger values of niter lead to less frequent printing
    print_every = max(100, hyperparams["niter"] // 10)

    batch_size, seq_len, niter = hyperparams["batch_size"], hyperparams["seq_len"], hyperparams["niter"]

    print(f"\n===== Training {model.__class__.__name__} =====")
    print(f"Device: {device} | Iterations: {niter}\n")
    
    start_time = time.time()
    for iter in range(niter):
        model.train()

        # --- Get training batch ---
        X, Y = get_batch(train_data, batch_size, seq_len) 
        X, Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        results["train_loss"].append(loss.item())

        # --- Compute training accuracy ---
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            preds = preds.view(batch_size, seq_len) # Reshape preds to match Y
            acc = (preds == Y).float().mean()
            acc_values = acc.item()  # convert tensor scalar to Python float
            results["train_acc"].append(acc_values)
        
        # --- Optimization step ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
        optimizer.step()

        elapsed = time.time() - start_time
        results["train_time"].append(elapsed)

        # Evaluate on validation set every few iterations
        if (iter + 1) % print_every == 0 or iter == 0:
            # Validation phase
            model.eval()
            
            total_val_loss = 0
            total_val_acc = 0
            val_iter = 100
            for _ in range(val_iter):
                # --- Get validation batch ---
                val_X, val_Y = get_batch(val_data, batch_size, seq_len)
                val_X, val_Y = val_X.to(device), val_Y.to(device)
                val_logits, val_loss = model(val_X, val_Y)
                total_val_loss += val_loss.item()
                

                with torch.no_grad():
                    val_preds = torch.argmax(val_logits, dim=-1)
                    val_preds = val_preds.view(batch_size, seq_len) # Reshape val_preds to match val_Y
                    val_acc = (val_preds == val_Y).float().mean()
                    total_val_acc += val_acc.item()  # convert tensor scalar to Python float
                    

            results["val_loss"].append(total_val_loss/val_iter) # Average loss over all batches
            results["val_acc"].append(total_val_acc/val_iter)                         # Average acc over all batches

            elapsed = time.time() - start_time
            results["val_time"].append(elapsed)

            print(f"[Iteration {iter+1}/{niter} | Time {elapsed:5.1f}s]")
            print(f"Train Loss: {loss.item():.4f} | Train Acc: {100*acc:.1f}% \t Validation Loss: {val_loss.item():.4f} | Validation Acc: {100*val_acc:.1f}%\n")

    print(f"===== Finished Training {model.__class__.__name__} =====\n")

    return results
