# if you don't install for torch copy this code in terminal
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128import torch
import torch
def gpu_preparation():
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        torch.set_default_device("cuda")
        print("PyTorch default device set to CUDA(GPU)")
        return True
    else:
        print("Warning: No GPU detected. performance will be slow.")
        return False