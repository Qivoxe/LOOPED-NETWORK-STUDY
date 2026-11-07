import torch
import numpy as np

print("=" * 45)
print("LOOPED NETWORK STUDY")
print("=" * 45)

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

x = torch.randn(5, 3)
print("Tensor test:", x.shape)

print("Environment verification successful.")