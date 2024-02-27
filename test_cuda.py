import torch

print(torch.__version__)       # Выведет версию PyTorch
print(torch.cuda.is_available())  # Выведет True, если CUDA доступна, иначе False
print(torch.version.cuda)     # Выведет версию CUDA, с которой собран PyTorch
