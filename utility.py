# %% [code]
# %% [code]
# %% [code]
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_epoch(model, criterion, data_loader, mode, optimizer=None, is_cuda=None):
    """
    Run a single epoch of training or validation.

    Args:
        model (torch.nn.Module): The neural network model.
        criterion (torch.nn.Module): The loss function.
        data_loader (torch.utils.data.DataLoader): DataLoader for loading batches of data.
        mode (str): Either "Train" or "Val" to indicate training or validation mode.
        optimizer (torch.optim.Optimizer, optional): The optimizer for updating model weights.
        is_cuda (bool, optional): Whether to use GPU for computation.

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

    return epoch_loss / len(data_loader.dataset)  # Return the average loss for the epoch



def test_run(model, criterion, data_loader, is_cuda):
    """
    Evaluate a model's performance on a given dataset using the specified criterion.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        criterion: The loss function used to compute the evaluation loss.
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        is_cuda (bool): A flag indicating whether to use GPU (CUDA) for evaluation.

    Returns:
        tuple: A tuple containing evaluation results and statistics.
            - test_loss (float): Average loss computed over the evaluation dataset.
            - preds (list): List of predicted labels for each evaluation sample.
            - actuals (list): List of true labels for each evaluation sample.
            - class_correct (list): List of counts of correctly predicted samples per class.
            - class_total (list): List of total samples per class in the evaluation dataset.

    Note:
        This function assumes that the model has been properly initialized and trained before evaluation.
        It computes the average loss, predicted labels, true labels, and per-class statistics for the evaluation dataset.
    """
    # Initialize per-class correct prediction and total sample counts
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]
    test_loss = 0

    if is_cuda:
        model.cuda()

    with torch.no_grad():
        model.eval()  # Set the model in evaluation mode

        preds = []  # List to store predicted labels
        actuals = []  # List to store true labels

        # Iterate through the evaluation dataset
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

            # Update per-class statistics
            for i in range(target.shape[0]):
                label = target[i]
                class_correct[label] += 1 if pred[i] == label else 0
                class_total[label] += 1

    test_loss = test_loss / len(data_loader.dataset)  # Calculate average test loss

    return test_loss, preds, actuals, class_correct, class_total



def train(model, epochs, train_loader, optimizer, criterion, val_loader=None, is_cuda=False):
    """
    Train a neural network model for a specified number of epochs using the provided data and parameters.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        epochs (int): The number of training epochs.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer: The optimization algorithm for updating the model's parameters.
        criterion: The loss function used for training.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. Default: None.
        is_cuda (bool): A flag indicating whether to use GPU (CUDA) for training. Default: False.

    Returns:
        tuple: A tuple containing lists of training and validation losses.
            - losses (list): List of training losses at each epoch.
            - val_losses (list): List of validation losses at each epoch (if val_loader is provided).

    Note:
        This function iterates through the specified number of epochs and trains the model using the provided
        training dataset and optimization parameters. If a validation DataLoader is provided, it computes
        validation losses as well. The function returns the training and validation loss lists.
    """
    val_losses = []  # List to store validation losses (if validation data is provided)
    losses = []  # List to store training losses

    for epoch in tqdm(range(epochs), desc="Epochs", total=epochs, leave=True, ncols=80):
        # Train the model for one epoch
        train_loss = run(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            is_cuda=is_cuda,
            mode="Train"
        )
        losses.append(train_loss)

        # If validation data is provided, compute validation loss
        if val_loader:
            val_loss = run(
                model=model,
                criterion=criterion,
                data_loader=val_loader,
                is_cuda=is_cuda,
                mode="Val"
            )
            val_losses.append(val_loss)
            print("Epoch: {} \tValidation Loss: {:.6f}".format(epoch + 1, val_loss))

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch + 1, train_loss))

    return losses, val_losses



def plot_loss(losses, epochs):
    """
    Generate a plot illustrating the training and validation losses across epochs.

    Args:
        losses (dict): A dictionary containing training and validation losses for different labels.
                       The keys represent label names, and the values are lists of corresponding losses.
        epochs (int): Total number of epochs for which the losses are recorded.

    Note:
        This function requires the 'matplotlib' library to be installed in the environment.
        It generates a line plot showing the changes in losses over the course of training.
        The provided 'losses' dictionary should contain losses for different labels, allowing comparison.
    """
    plt.figure(figsize=(8, 5))

    for label, loss in losses.items():
        plt.plot(range(1, epochs + 1), loss, label=label)  # Plot loss for each label

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)
    plt.show()

