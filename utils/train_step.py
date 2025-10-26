import time
import torch

def train_model(
    model,
    get_batch,
    train_data,
    val_data,
    optimizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hyperparams: dict = {
        "batch_size": 128,
        "seq_len": 64,
        "niter": 1000,
    },
    print_every: int = 100,
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
            results["train_acc"].append(acc)
            # acc_last = (preds[:, -1] == Y[:, -1]).float().mean()
        
        # --- Optimization step ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
        optimizer.step()

        elapsed = time.time() - start_time
        results["train_time"].append(elapsed)

        if (iter + 1) % print_every == 0:
            print(f"[Iteration {iter+1}/{niter} | Time {elapsed:5.1f}s] \t Train Loss: {loss.item():.4f} | Train Acc: {100*acc:.1f}%")

    print(f"\n--- Starting Validation Phase ---\n")

    for iter in range(niter):
        # Validation phase
        model.eval()
        
        # --- Get validation batch ---
        val_X, val_Y = get_batch(val_data, batch_size, seq_len)
        val_X, val_Y = val_X.to(device), val_Y.to(device)
        val_logits, val_loss = model(val_X, val_Y)
        results["val_loss"].append(val_loss.item())

        with torch.no_grad():
            val_preds = torch.argmax(val_logits, dim=-1)
            val_preds = val_preds.view(batch_size, seq_len) # Reshape val_preds to match val_Y
            val_acc = (val_preds == val_Y).float().mean()
            results["val_acc"].append(val_acc)
            # val_acc_last = (val_preds[:, -1] == val_Y[:, -1]).float().mean()

        elapsed = time.time() - start_time
        results["val_time"].append(elapsed)

        if (iter + 1) % print_every == 0:
            print(f"[Iteration {iter+1}/{niter} | Time {elapsed:5.1f}s] \t Validation Loss: {val_loss.item():.4f} | Validation Acc: {100*val_acc:.1f}%")

    print(f"===== Finished Training {model.__class__.__name__} =====\n")

    return results
