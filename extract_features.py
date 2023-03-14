from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from model import SimpleCNN
import torch
import numpy as np
from tqdm import tqdm
from dataset import ASLDataset_Triplet_Loss, ASLDataset_Classification

from torch.utils.data import DataLoader
import os

from utils import *
import argparse

import config as cfg

parser = argparse.ArgumentParser(description ='Choose model to train')


parser.add_argument('--model_type', default = "triplet",
					type = str,
					help ='Model type to train (triplet or classification)')

args = parser.parse_args()

model_type = args.model_type

if model_type not in ['triplet', 'classification']:
	print("Invalid model type")
	exit(1)

print(f"Using {model_type} model.")
model_path = os.path.join("models", model_type, "best.pt")

_, _, val_paths, val_labels, label_map = load_paths("Dataset/ASL_DATASET/asl_alphabet_train/asl_alphabet_train", 
																			val_ratio = 0.2, 
																			num_labels_to_load = cfg.NUM_CLASSES)

label_map_rev = {}

for key in label_map:
	label_map_rev[label_map[key]] = key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dl = ASLDataset_Classification(val_paths, val_labels, num_labels = cfg.NUM_CLASSES)

val_loader = DataLoader(val_dl, batch_size = cfg.BATCH_SIZE, shuffle = True)

model = SimpleCNN(model_type, embedding_size = cfg.EMBEDDING_SIZE, num_out = cfg.NUM_CLASSES).to(device)
model.load_state_dict(torch.load(model_path))


val_data_iter = iter(val_loader)
model.eval()
progress_bar = tqdm(val_data_iter, desc="Extracting Features")

all_features = []
all_labels = []

with torch.no_grad():
	for imgs, labels in progress_bar:
		imgs = imgs.to(device)
		labels = labels.to(device)

		labels = torch.argmax(labels, dim = 1)
		
		features = model.get_features(imgs)

		all_features += features.cpu().detach().numpy().tolist()
		all_labels += labels.cpu().detach().numpy().tolist()

all_features = np.array(all_features)
all_labels = np.array(all_labels)

alphabet_labels = []

for lab in all_labels:
	alphabet_labels.append(label_map_rev[lab])


tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, n_jobs = 4)
X_embedded = tsne.fit_transform(all_features)


plt.figure()
sns.set(style = "darkgrid")
palette = sns.color_palette("bright", cfg.NUM_CLASSES)
sns.scatterplot(x = X_embedded[:, 0], y = X_embedded[:, 1], hue = alphabet_labels, legend = "full", palette=palette)
plt.savefig(f"TSNE_{model_type}.png")

plt.show()