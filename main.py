from model import SimpleCNN
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from dataset import ASLDataset_Triplet_Loss, ASLDataset_Classification
from torchvision import transforms
from torch.utils.data import DataLoader
import os

from utils import *
from training_utils import train_loop_triplet_loss, validation_loop_triplet_loss, train_loop_classification, validation_loop_classifitaion, check_checkpoint
import argparse

import config as cfg

parser = argparse.ArgumentParser(description ='Choose model to train')


parser.add_argument('--model_type', default = "triplet",
					type = str,
					help ='Model type to train (triplet or classification)')

parser.add_argument('--epochs', default = 20,
					type = int,
					help ='Number of epochs for training the model')

args = parser.parse_args()

model_type = args.model_type

if model_type not in ['triplet', 'classification']:
	print("Invalid model type")
	exit(1)


if not os.path.exists("models"):
	os.mkdir("models")

checkpoints_dir = os.path.join("models", model_type)

if not os.path.exists(checkpoints_dir):
	os.mkdir(checkpoints_dir)

epochs = args.epochs

train_paths, train_labels, val_paths, val_labels, label_map = load_paths("Dataset/ASL_DATASET/asl_alphabet_train/asl_alphabet_train", 
																			val_ratio = 0.2, 
																			num_labels_to_load = cfg.NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == 'triplet':
	train_dl = ASLDataset_Triplet_Loss(train_paths, train_labels, num_labels = cfg.NUM_CLASSES)
	val_dl = ASLDataset_Triplet_Loss(val_paths, val_labels, num_labels = cfg.NUM_CLASSES)

	criterion = nn.TripletMarginLoss(margin=1.0, p=2)

	training_function = train_loop_triplet_loss
	validation_function = validation_loop_triplet_loss

else:
	train_dl = ASLDataset_Classification(train_paths, train_labels, num_labels = cfg.NUM_CLASSES)
	val_dl = ASLDataset_Classification(val_paths, val_labels, num_labels = cfg.NUM_CLASSES)

	criterion = nn.CrossEntropyLoss()

	training_function = train_loop_classification
	validation_function = validation_loop_classifitaion

train_loader = DataLoader(train_dl, batch_size = cfg.BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dl, batch_size = cfg.BATCH_SIZE, shuffle = True)


model = SimpleCNN(model_type, embedding_size = cfg.EMBEDDING_SIZE, num_out = cfg.NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if model_type == "classification":
	best_metric = 0
else:
	best_metric = 10000

model_path = os.path.join(checkpoints_dir, "best.pt")


f = open("stats.csv", "w")
f.write("Train Loss, Val Loss, Train Acc, Val Acc\n")
f.close()
for epoch in range(1, epochs+1):
	print(f"\n\nEpoch: {epoch}/{epochs}")
	training_loss, training_acc = training_function(train_loader, optimizer, criterion, model, device)
	validation_loss, validation_acc = validation_function(val_loader, criterion, model, device)

	print(f"Training Loss: {training_loss}, Training Acc: {training_acc}")
	print(f"Validation Loss: {validation_loss}, Validation Acc: {validation_acc}")

	f = open("stats.csv", "a")
	f.write(f"{training_loss},{validation_loss},{training_acc},{validation_acc}\n")
	f.close()

	if model_type == "classification":
		best_metric = check_checkpoint(validation_acc, best_metric, model_path, model, model_type)

	else:
		best_metric = check_checkpoint(validation_loss, best_metric, model_path, model, model_type)
