from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score

# Define training and testing loops
def train_loop(dataloader, model, loss_fn, optimizer, scheduler, epoch="", max_epochs=20, device="cuda", use_amp=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.train()
    train_loss = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loop = tqdm(dataloader)
    loop.set_description(f"Epoch [{epoch+1}/{max_epochs}]")
    
    for batch, (X, y) in enumerate(loop):
        X = X.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad() # set_to_none=True here can modestly improve performance
    
        # Update progress bar
        train_loss += loss.item()
        loop.set_postfix({"loss":train_loss / (batch+1)})


    train_loss /= num_batches
    
    if scheduler:
        scheduler.step(train_loss)

    return train_loss


def test_loop(dataloader, model, loss_fn, split="test", device="cpu"):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0.0
    all_preds = []
    all_labels = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            running_loss += loss_fn(outputs, y).item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
  
    # Compute metrics
    total_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"{split} metrics: \n Accuracy: {(100*accuracy):>0.2f}%, F1-score: {(100*macro_f1):>0.2f}%, Avg loss: {total_loss:>8f} \n")
    return total_loss, accuracy, macro_f1
