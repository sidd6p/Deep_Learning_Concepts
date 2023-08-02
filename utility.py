# %% [code]
# %% [code]
# %% [code]

import matplotlib.pyplot as plt

from tqdm import tqdm

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
        
    


def plot_loss(losses, epochs):
    plt.figure(figsize=(8, 5))
    
    for label, loss  in losses.items():   
        plt.plot(range(1, epochs + 1), loss, label=label)   
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.grid(True)
    plt.show()