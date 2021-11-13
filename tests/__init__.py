import os
import torch


def get_device():
    if "USE_GPU" in os.environ:
        use_gpu = os.environ['USE_GPU'].lower() in ["true"]
    else:
        use_gpu = False
    print("Test on GPU:", use_gpu)
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return device
