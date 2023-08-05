# %% [code]
# %% [code]
# %% [code]
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def run(model, criterion, data_loader, mode, optimizer=None, is_cuda=None):
    """
    Run one epoch of training or validation.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        criterion (torch.nn.Module): The loss function.
        data_loader (torch.utils.data.DataLoader): DataLoader for loading batches of data.
        is_cuda (bool): Whether to use GPU for computation.
        mode (str): Either "Train" or "Val" to indicate training or validation mode.

    Returns:
        float: The average loss for the epoch.
    """
    if is_cuda:
        model.cuda()

    if mode == "Train":
        model.train()  # Set the model in training mode
    elif mode == "Val":
        model.eval()  # Set the model in evaluation mode

    epoch_loss = 0.0

    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()

        if mode == "Train":
            optimizer.zero_grad()  # Clear gradients before backward pass
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        if mode == "Train":
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights

        epoch_loss += loss.item() * data.size(0)  # Accumulate loss

    return epoch_loss / len(data_loader.dataset)  # Return average loss


def test_run(model, criterion, data_loader, is_cuda):
    """
    Run the model on the test dataset for evaluation.

    Args:
        model (torch.nn.Module): The neural network model.
        criterion (torch.nn.Module): The loss function.
        data_loader (torch.utils.data.DataLoader): DataLoader for loading test data.
        is_cuda (bool): Whether to use GPU for computation.

    Returns:
        tuple: A tuple containing test loss, predicted labels, actual labels,
               correct predictions per class, and total samples per class.
    """
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]
    test_loss = 0

    if is_cuda:
        model.cuda()

    with torch.no_grad():
        model.eval()  # Set the model in evaluation mode

        preds = []
        actuals = []

        for batch_idx, (data, target) in tqdm(
                enumerate(data_loader),
                desc='Testing',
                total=len(data_loader),
                leave=True,
                ncols=80
        ):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            output = model(data)  # Forward pass
            loss_value = criterion(output, target).detach()  # Compute loss
            test_loss += loss_value * data.size(0)  # Accumulate loss

            _, pred = torch.max(output, 1)  # Get predicted labels
            pred = pred.cpu().numpy().tolist()
            preds.extend(pred)
            actuals.extend(target.cpu().numpy().tolist())

            for i in range(target.shape[0]):  # Calculate per-class statistics
                label = target[i]
                class_correct[label] += 1 if pred[i] == label else 0
                class_total[label] += 1

    test_loss = test_loss / len(data_loader.dataset)  # Calculate average test loss

    return test_loss, preds, actuals, class_correct, class_total


def train(model, epochs, train_loader, optimizer, criterion, val_loader=None, is_cuda=False):
    val_losses = []
    losses = []

    for epoch in tqdm(range(epochs), desc="Epochs", total=epochs, leave=True, ncols=80):
        train_loss = run(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            is_cuda=is_cuda,
            mode="Train"
        )
        losses.append(train_loss)
        
        if val_loader:
            val_loss = run(
                model=model,
                criterion=criterion,
                data_loader=val_loader,
                is_cuda=is_cuda,
                mode="Val"
            )
            val_losses.append(val_loss)
            print("Epoch: {} \Validation Loss: {:.6f}".format(epoch + 1, train_loss))
            
        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch + 1, train_loss))
    
    return losses, val_losses


def plot_loss(losses, epochs):
    """
    Plot the training and validation losses over epochs.

    Args:
        losses (dict): A dictionary containing training and validation losses for different labels.
        epochs (int): Total number of epochs.
    """
    plt.figure(figsize=(8, 5))

    for label, loss in losses.items():
        plt.plot(range(1, epochs + 1), loss, label=label)  # Plot loss for each label

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)
    plt.show()
