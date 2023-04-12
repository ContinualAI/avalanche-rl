import os
import torch


def get_device():
    use_gpu = os.environ.get('USE_GPU', False).lower() == "true"
    print("Test on GPU:", use_gpu)
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return device
