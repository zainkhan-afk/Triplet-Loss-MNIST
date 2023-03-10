from model import SimpleCNN
import torch
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from dataset import ASLDataset
from torchvision import transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dims = 2
batch_size = 32
epochs = 50

dl = ASLDataset("Dataset/ASL_DATASET/asl_alphabet_train/asl_alphabet_train")
train_loader = DataLoader(dl, batch_size = batch_size, shuffle = True)


# dl = ASLDataset("Dataset/ASL_DATASET/asl_alphabet_test/asl_alphabet_test")
# test_data_generator = DataLoader(dl, batch_size = batch_size, shuffle = True)
# test_data_iter = iter(test_data_generator)



model = SimpleCNN().to(device)
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

# train_data_iter = iter(train_loader)
# for i in train_data_iter:
#     print(i)
#     break
# exit()

for epoch in tqdm(range(epochs), desc="Epochs"):
    running_loss = []
    train_data_iter = iter(train_loader)
    step = 0
    for anchor_img, positive_img, negative_img, anchor_label in tqdm(train_data_iter, desc="Training", leave=False):
        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)
        
        optimizer.zero_grad()
        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.cpu().detach().numpy())
        step += 1
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))