# ABOUTME: QMC training loop — train/test epoch + full loops over a fixed latent base_sequence.
# ABOUTME: Random per-epoch lattice shift; evidence loss; optional Jacobian-energy regularization.
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch
from .losses import jacEnergy


def train_epoch(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True,conditional=False,importance_weights=[]):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx,batch in enumerate(loader):
        data= batch[0]
        data = data.to(model.device)
        optimizer.zero_grad()
        if conditional:
            c = batch[1].to(torch.float32).to(model.device).view(1,-1)
            samples = model(base_sequence,random,mod,c)
        else:
            samples = model(base_sequence,random,mod)
        if len(importance_weights) == 0:
            loss = loss_function(samples, data)
        else:
            loss = loss_function(samples,data,importance_weights)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True,conditional=False,importance_weights=[]):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, batch in tqdm(enumerate(loader),total=len(loader)):
        data = batch[0]
        data = data.to(model.device)
        optimizer.zero_grad()
        if conditional:
            c = batch[1].to(torch.float32).to(model.device).view(1,-1)
            samples = model(base_sequence,random,mod,c)
        else:
            samples = model(base_sequence,random,mod)
        if len(importance_weights) == 0:
            loss = loss_function(samples, data)
        else:
            loss = loss_function(samples,data,importance_weights)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def test_epoch(model,loader,base_sequence,loss_function,conditional=False,random=True,mod=True,importance_weights=[]):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    epoch_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data = batch[0]
            data = data.to(model.device)
            if conditional:
                c = batch[1].to(torch.float32).to(model.device).view(1,-1)
                samples = model(base_sequence,random=True,mod=True,c=c)
            else:
                samples = model(base_sequence,random=True,mod=True)
            #samples = model(base_sequence)
            if len(importance_weights) == 0:
                loss = loss_function(samples, data)
            else:
                loss = loss_function(samples,data,importance_weights)
            test_loss += loss.item()
            epoch_losses.append(loss.item())

    return epoch_losses

def train_loop(model,loader,base_sequence,loss_function,nEpochs=100,verbose=False,
               random=True,mod=True,conditional=False,importance_weights=[],print_gpu_mem=False):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):

        if verbose:
            batch_loss,model,optimizer = train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod,conditional=conditional,importance_weights=importance_weights)
        else:
            batch_loss,model,optimizer = train_epoch(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod,conditional=conditional,importance_weights=importance_weights)

        losses += batch_loss
        if verbose:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')
        if print_gpu_mem and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved  = torch.cuda.memory_reserved()  / 1024**2
            peak      = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  [GPU epoch {epoch+1}] allocated={allocated:.1f}MB  reserved={reserved:.1f}MB  peak={peak:.1f}MB")

    return model, optimizer,losses

def train_epoch_mc(model,optimizer,loader,mc_func,loss_function):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(loader):
        base_sequence=mc_func().to(model.device)
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random=False,mod=False)
        loss = loss_function(samples, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(loss.item())

    return epoch_losses,model,optimizer

def train_loop_mc(model,loader,loss_function,mc_func,nEpochs=100,print_losses=False):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):

        batch_loss,model,optimizer = train_epoch_mc(model,optimizer,loader,mc_func,loss_function)

        losses += batch_loss
        if print_losses:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')


    return model, optimizer,losses


############ TO DO: FILL IN HERE, ADD REGULARIZER STUFF ####

def train_epoch_reg(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        reg = jacEnergy(samples)
        l = loss + reg
        l.backward()
        train_loss += loss.item()
        optimizer.step()
        epoch_losses.append(l.item())

    return epoch_losses,model,optimizer

def train_epoch_reg_verbose(model,optimizer,loader,base_sequence,loss_function,random=True,mod=True):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss = 0
    epoch_losses = []
    for batch_idx, (data, _) in tqdm(enumerate(loader)):
        data = data.to(model.device)
        optimizer.zero_grad()
        samples = model(base_sequence,random,mod)
        loss = loss_function(samples, data)
        reg = jacEnergy(samples)
        l = loss + reg
        l.backward()
        train_loss += l.item()
        optimizer.step()
        epoch_losses.append(l.item())

    return epoch_losses,model,optimizer

def test_epoch_reg(model,loader,base_sequence,loss_function):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    epoch_losses = []
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(loader)):
            data = data.to(model.device)
            samples = model(base_sequence)
            loss = loss_function(samples, data)
            test_loss += loss.item()
            epoch_losses.append(loss.item())

    return epoch_losses

def train_loop_reg(model,loader,base_sequence,loss_function,nEpochs=100,verbose=False,
               random=True,mod=True):

    optimizer = Adam(model.parameters(),lr=1e-3)
    losses = []
    for epoch in tqdm(range(nEpochs)):

        if verbose:
            batch_loss,model,optimizer = train_epoch_verbose(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod)
        else:
            batch_loss,model,optimizer = train_epoch(model,optimizer,loader,base_sequence,loss_function,
                                                 random=random,mod=mod)

        losses += batch_loss
        if verbose:
            print(f'Epoch {epoch + 1} Average loss: {np.sum(batch_loss)/len(loader.dataset):.4f}')

    return model, optimizer,losses
