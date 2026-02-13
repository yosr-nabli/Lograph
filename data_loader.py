import torch
import config
import torch as np
from torch.utils.data import Dataset, DataLoader

class LographDataset(Dataset):
	def __init__(self, samples, smpl_index, tmpl_list, template_map):
		self.dataset = []
		self.group_map = {}
		self.process(samples, smpl_index, tmpl_list, template_map)

	def process(self, samples, smpl_index, tmpl_list, template_map):
		for idx in smpl_index:
			indice = samples[idx][0]
			word = [template_map[tmpl_list[j]] for j in samples[idx][0]] #list of word IDs for that template.
			label = samples[idx][1]
			group = samples[idx][2]
			if group not in self.group_map:
				self.group_map[group] = len(self.group_map) #Map group string to integer
			group_id = self.group_map[group]
			self.dataset.append([word, label, group_id, indice])
		return self

	def __getitem__(self, idx):
		word = self.dataset[idx][0]
		label = self.dataset[idx][1]
		group = self.dataset[idx][2]
		indice = self.dataset[idx][3]
		return word, label, group, indice

	def __len__(self):
		return len(self.dataset)

def lograph_collate_fn(batch):
	batch_size = len(batch)
	words = [x[0] for x in batch]# list of words
	labels = [x[1] for x in batch] # list of labels
	groups = [x[2] for x in batch]# list of group_ids
	lengths = [len(x[0]) for x in batch] # sequence length per sample
	word_counts = [[len(v) for v in x[0]] for x in batch] # number of words per sample
	max_length = max(lengths) # longest sequence length in a bash
	max_word_count = max([max(x) for x in word_counts]) # longest word list in batch
	indices = [(x[3]+[-1]*max_length)[:max_length] for x in batch] # indices of words in the sequence
	word_tensor = torch.LongTensor(batch_size, max_length, max_word_count).fill_(0)
	group_tensor = torch.LongTensor(batch_size).fill_(0)
	label_tensor = torch.LongTensor(batch_size).fill_(0)
	mask_tensor = torch.ByteTensor(batch_size, max_length, max_word_count).fill_(0) # 1 where words exisst 0 where not
	for i, (word,label,group) in enumerate(zip(words, labels, groups)):
		for j,num in enumerate(word_counts[i]):
			word_tensor[i, j, :num] = torch.LongTensor(word[j])
			mask_tensor[i, j, :num] = torch.tensor([1]*num, dtype=torch.uint8)
		label_tensor[i] = label
		group_tensor[i] = group
	indice_array = np.array(indices)
	return word_tensor, label_tensor, group_tensor, mask_tensor, indice_array #indice_tensor

def convert_to_training_data_loader(samples, train_index, dev_index, tmpl_list, template_map, pseudo_label=None):
	train_dataset = LographDataset(samples, train_index, tmpl_list, template_map)
	dev_dataset = LographDataset(samples, dev_index, tmpl_list, template_map)
	train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=lograph_collate_fn)
	dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=lograph_collate_fn)
	return train_loader, dev_loader

def convert_to_testing_data_loader(samples, test_index, tmpl_list, template_map):
	test_dataset = LographDataset(samples, test_index, tmpl_list, template_map)
	test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=lograph_collate_fn)
	return test_loader
