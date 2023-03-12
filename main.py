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

from utils import *
from training_utils import train_loop_triplet_loss, validation_loop_triplet_loss, train_loop_classification, validation_loop_classifitaion
import argparse

import config as cfg

parser = argparse.ArgumentParser(description ='Choose model to train')


parser.add_argument('--model_type', default = "triplet",
					type = str,
					help ='Model type to train (triplet or classification)')

parser.add_argument('--epochs', default = 50,
					type = int,
					help ='Number of epochs for training the model')

args = parser.parse_args()

model_type = args.model_type

if model_type not in ['triplet', 'classification']:
	print("Invalid model type")
	exit(1)


epochs = args.epochs

train_paths, train_labels, val_paths, val_labels, label_map = load_paths("Dataset/ASL_DATASET/asl_alphabet_train/asl_alphabet_train", 
																			val_ratio = 0.2, 
																			num_labels_to_load = cfg.NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type == 'triplet':
	train_dl = ASLDataset_Triplet_Loss(train_paths, train_labels, num_labels = cfg.NUM_CLASSES)
	val_dl = ASLDataset_Triplet_Loss(val_paths, val_labels, num_labels = cfg.NUM_CLASSES)

	num_outputs = cfg.EMBEDDING_SIZE
	criterion = nn.TripletMarginLoss(margin=1.0, p=2)

	training_function = train_loop_triplet_loss
	validation_function = validation_loop_triplet_loss

else:
	train_dl = ASLDataset_Classification(train_paths, train_labels, num_labels = cfg.NUM_CLASSES)
	val_dl = ASLDataset_Classification(val_paths, val_labels, num_labels = cfg.NUM_CLASSES)

	num_outputs = cfg.NUM_CLASSES
	criterion = nn.CrossEntropyLoss()

	training_function = train_loop_classification
	validation_function = validation_loop_classifitaion

train_loader = DataLoader(train_dl, batch_size = cfg.BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dl, batch_size = cfg.BATCH_SIZE, shuffle = True)


model = SimpleCNN(num_out = num_outputs).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



for epoch in tqdm(range(epochs), desc="Epochs"):
	training_loss = training_function(train_loader, optimizer, criterion, model, device)
	validation_loss = validation_function(val_loader, criterion, model, device)