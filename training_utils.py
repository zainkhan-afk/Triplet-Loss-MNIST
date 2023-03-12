import torch
from tqdm import tqdm
import numpy as np

import config as cfg

def train_loop_triplet_loss(train_loader, optimizer, criterion, model, device):
	train_data_iter = iter(train_loader)
	model.train()
	running_loss = []
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

		if step%cfg.LOG_INTERVAL == 0:
			print(f"Step {step} - Loss: {loss}")
		
		running_loss.append(loss.cpu().detach().numpy())
		step += 1

	return running_loss


def validation_loop_triplet_loss(val_loader, criterion, model, device):
	val_data_iter = iter(val_loader)
	step = 0
	running_loss = []
	model.eval()
	with torch.no_grad():
		for anchor_img, positive_img, negative_img, anchor_label in tqdm(val_data_iter, desc="Validation", leave=False):
			anchor_img = anchor_img.to(device)
			positive_img = positive_img.to(device)
			negative_img = negative_img.to(device)
			
			anchor_out = model(anchor_img)
			positive_out = model(positive_img)
			negative_out = model(negative_img)
			
			loss = criterion(anchor_out, positive_out, negative_out)

			if step%cfg.LOG_INTERVAL == 0:
				print(f"Step {step} - Loss: {loss}")
			
			running_loss.append(loss.cpu().detach().numpy())
			step += 1

	return running_loss


def train_loop_classification(train_loader, optimizer, criterion, model, device):
	train_data_iter = iter(train_loader)
	model.train()
	running_loss = []
	step = 0
	for imgs, labels in tqdm(train_data_iter, desc="Training", leave=False):
		imgs = imgs.to(device)
		labels = labels.to(device)
		
		optimizer.zero_grad()
		pred = model(imgs)
		
		loss = criterion(pred, labels)
		loss.backward()
		optimizer.step()

		if step%cfg.LOG_INTERVAL == 0:
			print(f"Step {step} - Loss: {loss}")
		
		running_loss.append(loss.cpu().detach().numpy())
		step += 1

	return running_loss


def validation_loop_classifitaion(val_loader, criterion, model, device):
	val_data_iter = iter(val_loader)
	step = 0
	running_loss = []
	model.eval()
	with torch.no_grad():
		for imgs, labels in tqdm(val_data_iter, desc="Validation", leave=False):
			imgs = imgs.to(device)
			labels = labels.to(device)
			
			pred = model(imgs)
			
			loss = criterion(pred, labels)

			if step%cfg.LOG_INTERVAL == 0:
				print(f"Step {step} - Loss: {loss}")
			
			running_loss.append(loss.cpu().detach().numpy())
			step += 1

	return running_loss