import random
import os
import glob

def load_paths(data_dir, num_labels_to_load = 10):
	all_paths = []
	labels = []
	label_map = {}
	for i in range(10):
		label_map[chr(65+i)] = i
	for key in label_map:
		path = os.path.join(data_dir, key)
		path = path + f"/*.jpg"
		label_paths = glob.glob(path)

		all_paths += label_paths
		labels += [label_map[key]]*len(label_paths)

	return all_paths, labels, label_map


def split_dataset(all_paths, labels, val_ratio = 0.2):
	N = len(all_paths)
	val_N = int(val_ratio*N)

	val_paths = all_paths[:val_N]
	train_paths = all_paths[val_N:]

	val_labels = labels[:val_N]
	train_labels = labels[val_N:]

	return train_paths, train_labels, val_paths, val_labels