from model import SimpleCNN
import torch
import torch.nn as nn

model = SimpleCNN()
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.SG