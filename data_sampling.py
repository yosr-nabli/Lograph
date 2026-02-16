import config
import re
import numpy as np
from utils import *
from tqdm import tqdm

HOUR = 3600 # seconds

def convert_into_samples(log_data, entity=None, window_size=config.window_size, 
							log_name="bgl", max_time_interval=24*HOUR):
	samples = []
	temp_info = {}
	group_names = ['']
	min_window_length = 1
	sample_proba = 0.2
	collected_groups = set()
	def add_sequence_sample(group_id, tmp_id=None, use_sub_smpl=False):
		if group_id not in temp_info:
			return
		if tmp_id is not None:
			x = [tmp_id] + temp_info[group_id][0]
		else:
			x = [] + temp_info[group_id][0]
		if min_window_length <= len(x):
			y = 1 if temp_info[group_id][1]>0 else 0
			g = group_id
			first_sample = False
			if g not in collected_groups:
				collected_groups.add(g)
				first_sample = True
			if sample_proba==None or first_sample==True or np.random.random() < sample_proba:
				samples.append([x, y, g])
		return

	for log_id in tqdm(range(len(log_data))):
		log, label, timestamp, entities = log_data[log_id]
		if entity is not None and entity!='None':
			if entity not in entities: continue
			else: group_names = entities[entity]
		for group_id in group_names:
			if group_id not in temp_info or timestamp-temp_info[group_id][-1] > max_time_interval:
				add_sequence_sample(group_id)
				temp_info[group_id] = [[], 0, -1]
			temp_info[group_id][0].append(log_id)
			temp_info[group_id][1] += label
			temp_info[group_id][2] = timestamp
			if len(temp_info[group_id][0])==window_size:
				tmp_id = temp_info[group_id][0][0]
				temp_info[group_id][0] = temp_info[group_id][0][1:]
				add_sequence_sample(group_id, tmp_id, True)
				temp_info[group_id][1] -= log_data[tmp_id][1]

	for group_id in temp_info:
		if group_id=='': continue
		if min_window_length <= len(temp_info[group_id][0]) < window_size-1:
			add_sequence_sample(group_id)
	samples.sort(key=lambda x:x[0][-1])
	return samples
	
def train_test_split(samples, sample_ratio=0.05):
	all_index = list(range(len(samples)))
	np.random.shuffle(all_index)
	all_index = all_index[:int(len(all_index)*sample_ratio)]
	N = len(all_index)
	train_size = int(N*config.train_prop)
	train_and_dev_size = int(N*(1-config.test_prop))
	test_index = all_index[train_and_dev_size:]
	test_index.sort()
	train_dev_index = all_index[:train_and_dev_size]
	np.random.shuffle(train_dev_index)
	train_index = train_dev_index[:train_size]
	dev_index = train_dev_index[train_size:train_and_dev_size]
	# dev_index = np.random.choice(test_index, len(dev_index))
	return train_index, dev_index, test_index

def train_test_split_grouped(samples, sample_ratio=0.1):
	all_index = list(range(len(samples)))
	if config.sort_chronological==False:
		np.random.shuffle(all_index)
	all_group_map = {}
	for i, smpl in enumerate(samples):
		g = smpl[-1]
		if g not in all_group_map:
			all_group_map[g] = []
		all_group_map[g].append(i)
	all_groups = list(all_group_map.keys())
	np.random.shuffle(all_groups)
	N = len(all_groups)
	train_size = int(N*config.train_prop)
	train_and_dev_size = int(N*(1-config.test_prop))
	test_groups = all_groups[train_and_dev_size:] #last 20% of groups
	train_dev_groups = all_groups[:train_and_dev_size] # first 80% of groups
	np.random.shuffle(train_dev_groups)
	train_groups = train_dev_groups[:train_size] #first 70% of groups
	dev_groups = train_dev_groups[train_size:train_and_dev_size]
	#N = 100
	#train_size = 70
	#train_and_dev_size = 90
	#test_groups = all_groups[90:] → groups 90-99 (10 groups)
	#train_dev_groups = all_groups[:90] → groups 0-89 (90 groups)
	#Shuffle train_dev_groups
	#train_groups = train_dev_groups[:70] → 70 groups
	#dev_groups = train_dev_groups[70:90] → 20 groups

	train_groups = np.random.choice(train_groups, int(len(train_groups)*sample_ratio))
	test_groups = np.random.choice(test_groups, int(len(test_groups)*sample_ratio))
	dev_groups = np.random.choice(dev_groups, int(len(dev_groups)*sample_ratio))
	#For each selected group, collects all sample indices from that group.
	#Example: if train_groups = ["N1", "N5"] and all_group_map["N1"] = [0,1,2], all_group_map["N5"] = [10,11], then train_index = [0,1,2,10,11].

	train_index = [i for g in train_groups for i in all_group_map[g]]
	test_index = [i for g in test_groups for i in all_group_map[g]]
	dev_index = [i for g in dev_groups for i in all_group_map[g]]
	np.random.shuffle(train_index)
	np.random.shuffle(test_index)
	np.random.shuffle(dev_index)
	return train_index, dev_index, test_index

def simple_balance_sampling(samples, train_index, balance_coef=2):
	index_group_by_label = [[], []] # [normal_indices, anomalous_indices]
	for i in train_index:
		y = samples[i][1] #label of the sample
		index_group_by_label[y].append(i)
	num_positive = len(index_group_by_label[1]) #number of anomalous samples
	#Keeps only the first num_positive * 2 normal samples.
	#Example: if num_positive = 500, keeps 500 * 2 = 1000 normal samples.
	index_group_by_label[0] = index_group_by_label[0][:int(num_positive*balance_coef)]
	#combine and shuffle
	collect_index = index_group_by_label[0] + index_group_by_label[1]
	np.random.shuffle(collect_index)
	return collect_index
