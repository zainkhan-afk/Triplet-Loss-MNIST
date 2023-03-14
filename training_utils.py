import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

import config as cfg

def train_loop_triplet_loss(train_loader, optimizer, criterion, model, device):
	train_data_iter = iter(train_loader)
	model.train()
	running_loss = []
	step = 0
	progress_bar = tqdm(train_data_iter, desc="Training")
	for anchor_img, positive_img, negative_img, anchor_label in progress_bar:
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

		description_str = "Train Loss: {:.3f}".format(np.mean(running_loss))

		progress_bar.set_description(description_str)
		step += 1

	return np.mean(running_loss), None


def validation_loop_triplet_loss(val_loader, criterion, model, device):
	val_data_iter = iter(val_loader)
	step = 0
	running_loss = []
	model.eval()
	progress_bar = tqdm(val_data_iter, desc="Validation")
	with torch.no_grad():
		for anchor_img, positive_img, negative_img, anchor_label in progress_bar:
			anchor_img = anchor_img.to(device)
			positive_img = positive_img.to(device)
			negative_img = negative_img.to(device)
			
			anchor_out = model(anchor_img)
			positive_out = model(positive_img)
			negative_out = model(negative_img)
			
			loss = criterion(anchor_out, positive_out, negative_out)
			
			running_loss.append(loss.cpu().detach().numpy())

			description_str = "Val Loss: {:.3f}".format(np.mean(running_loss))

			progress_bar.set_description(description_str)
			step += 1

	return np.mean(running_loss), None


def train_loop_classification(train_loader, optimizer, criterion, model, device):
	train_data_iter = iter(train_loader)
	model.train()
	running_loss = []
	running_acc = []
	step = 0
	progress_bar = tqdm(train_data_iter, desc="Training")
	for imgs, labels in progress_bar:
		imgs = imgs.to(device)
		labels = labels.to(device)
		
		optimizer.zero_grad()
		pred = model(imgs)

		pred_softmax = F.softmax(pred, dim = 1)

		pred_cat = torch.argmax(pred_softmax, dim = 1)
		labels_cat = torch.argmax(labels, dim = 1)

		acc = (pred_cat == labels_cat).sum()/len(pred_cat)



		loss = criterion(pred, labels)
		loss.backward()
		optimizer.step()
		
		running_loss.append(loss.cpu().detach().numpy())
		running_acc.append(acc.cpu().detach().numpy())

		description_str = "Train Loss: {:.3f}, Acc: {:.3f}".format(np.mean(running_loss), np.mean(running_acc))

		progress_bar.set_description(description_str)
		step += 1

	return np.mean(running_loss), np.mean(running_acc)


def validation_loop_classifitaion(val_loader, criterion, model, device):
	val_data_iter = iter(val_loader)
	step = 0
	running_loss = []
	running_acc = []
	model.eval()
	progress_bar = tqdm(val_data_iter, desc="Validation")
	with torch.no_grad():
		for imgs, labels in progress_bar:
			imgs = imgs.to(device)
			labels = labels.to(device)
			
			pred = model(imgs)

			pred_softmax = F.softmax(pred, dim = 1)

			pred_cat = torch.argmax(pred_softmax, dim = 1)
			labels_cat = torch.argmax(labels, dim = 1)

			acc = (pred_cat == labels_cat).sum()/len(pred_cat)
			
			loss = criterion(pred, labels)
			
			running_loss.append(loss.cpu().detach().numpy())
			running_acc.append(acc.cpu().detach().numpy())

			description_str = "Val Loss: {:.3f}, Acc: {:.3f}".format(np.mean(running_loss), np.mean(running_acc))

			progress_bar.set_description(description_str)
			step += 1

	return np.mean(running_loss), np.mean(running_acc)

def check_checkpoint(val_metric, best_metric, path, model, model_type):
	if model_type == "classification":
		if val_metric>best_metric:
			torch.save(model.state_dict(), path)
			print(f"\nModel saved at: {path}")
			best_metric = val_metric

	else:
		if val_metric<best_metric:
			torch.save(model.state_dict(), path)
			print(f"\nModel saved at: {path}")
			best_metric = val_metric

	return best_metric