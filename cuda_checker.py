import torch

print("CUDA available:", torch.cuda.is_available())
print("PyTorch:", torch.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("선택된 장치:", device)
