import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)
    print("PyTorch is using GPU:", gpu_name)
else:
    print("PyTorch is using CPU")