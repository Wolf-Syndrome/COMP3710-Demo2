import torch

print("Hello world")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning couldn't find cuda using cpu.")