"""
Filename: model_utils.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Calibrate the OpenSpliceAI model.
"""

import torch
import numpy as np
from openspliceai.constants import *
from openspliceai.train_base.openspliceai import SpliceAI

def initialize_model_and_optim(device, flanking_size, pretrained_model):
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18 * N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12 * N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6 * N_GPUS    
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    # Initialize the model
    model = SpliceAI(L, W, AR, apply_softmax=False).to(device)
    # Print the shapes of the parameters in the initialized model
    print("\nInitialized model parameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", end=", ")

    # Load the pretrained model
    state_dict = torch.load(pretrained_model, map_location=device)

    # Filter out unnecessary keys and load matching keys into model
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Load state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print missing and unexpected keys
    print("\nMissing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, params

