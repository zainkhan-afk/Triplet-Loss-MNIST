from model import SimpleCNN
import torch

model = SimpleCNN()

x = torch.randn((1, 28, 28))
model(x)