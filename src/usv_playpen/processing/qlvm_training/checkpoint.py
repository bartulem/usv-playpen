# ABOUTME: QMC checkpoint save/load — bundles model+optimizer state_dicts plus run info.
# ABOUTME: convert_qmc_dict migrates weight keys from basis-in-decoder to basis-in-model layouts.
import torch
import copy
from collections import OrderedDict

def save(model,optimizer,run_info,fn = ''):
    """
    save a model. This will save model parameters,
    optimizer parameters, and info about the run
    the run info should include training epoch and
    loss trajectory (in the case of VAEs, both the
    reconstruction loss and the kl term)
    """

    model_state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()
    model_state = {'model': model_state_dict,'optimizer':opt_state_dict,'run info':run_info}
    torch.save(model_state,fn)

def load(model,optimizer,fn = ''):


    checkpoint = torch.load(fn,weights_only=False)
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        print(f"tried to load weights from {fn}; something went wrong!")
        print("trying to alter weight dict to match new structure")
        model = convert_qmc_dict(checkpoint,model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    run_info = checkpoint['run info']

    return model,optimizer, run_info

def convert_qmc_dict(checkpoint,model):

    """
    convert dictionaries from models where the basis
    WAS in the decoder to dictionaries where the basis
    is in the model object
    """
    try:
        model.load_state_dict(checkpoint['model'])

    except:
        print(f"weights mismatch; converting")
        dict_copy = OrderedDict()

        keylist = checkpoint['model'].keys()
        for key in keylist:
            split_key = key.split('.')

            split_key[1] = str(int(split_key[1]) - 1)
            dict_copy['.'.join(split_key)] = checkpoint['model'][key]

        model.load_state_dict(dict_copy)

    return model
