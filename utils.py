import random
import os
import glob

def load_paths(data_dir, val_ratio = 0.2, num_labels_to_load = 10):
	train_paths = []
	train_labels = []

	val_paths = []
	val_labels = []

	label_map = {}
	for i in range(10):
		label_map[chr(65+i)] = i

	for key in label_map:
		path = os.path.join(data_dir, key)
		path = path + f"/*.jpg"
		label_paths = glob.glob(path)

		N = len(label_paths)
		val_N = int(val_ratio*N)

		val_paths += label_paths[:val_N]
		train_paths += label_paths[val_N:]

		val_labels += [label_map[key]]*val_N
		train_labels += [label_map[key]]*(N-val_N)

	return train_paths, train_labels, val_paths, val_labels, label_map