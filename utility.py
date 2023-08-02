# %% [code]
# %% [code]
# %% [code]
# %% [code]

import matplotlib.pyplot as plt

from tqdm import tqdm
import torch 

def run(model, optimizer, criterion, data_loader, is_cuda, mode):
    if is_cuda:
        model.cuda()
    
    if mode == "Train":
        model.train()
    elif mode == "Val":
        model.eval()
    
    epoch_loss = 0.0
    
    for batch_idx, (data, target) in tqdm(
        enumerate(data_loader),
        desc=mode,
        total=len(data_loader),
        leave=True,
        ncols=80
    ):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
            
        if mode == "Train": optimizer.zero_grad()
        output = model(data)
        loss = criterion (output, target)
        if mode == "Train": loss.backward()
        if mode == "Train": optimizer.step()
        
        epoch_loss += loss.item()*data.size(0)
    
    return epoch_loss/ len(data_loader.dataset)


def test_run(model, criterion, data_loader, is_cuda):
    class_correct = list(0 for i in range(10))
    class_total = list(0 for i in range(10))
    test_loss = 0
    correct = 0
    total = 0
    
    if is_cuda:
        model.cuda()
        
    with torch.no_grad():
        model.eval()
        
    preds = list()
    actuals = list()
    
    for batch_idx, (data, target) in tqdm(
            enumerate(data_loader),
            desc='Testing',
            total=len(data_loader),
            leave=True,
            ncols=80
        ):
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            
        output = model(data)
        loss_value = criterion(output, target).detach()
        test_loss += loss_value*data.size(0)
        
        _, pred = torch.max(output, 1)
        pred = pred.cpu().numpy().tolist()
        preds.extend(pred)
        actuals.extend(target.cpu().numpy().tolist())

        
        for i in range(target.shape[0]):
            label = target[i]
            class_correct[label] += (1 if pred[i] == label else 0)
            class_total[label] += 1
    
    test_loss = test_loss/len(data_loader.dataset)
    
    return test_loss, preds, actuals, class_correct, class_total


def plot_loss(losses, epochs):
    plt.figure(figsize=(8, 5))
    
    for label, loss  in losses.items():   
        plt.plot(range(1, epochs + 1), loss, label=label)   
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.grid(True)
    plt.show()