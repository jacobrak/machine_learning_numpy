import numpy as np

def _sqrt_decay(lr, epoch):
    new_lr = lr/np.sqrt(1+epoch)
    new_lr = max(new_lr, 1e-6)
    return new_lr