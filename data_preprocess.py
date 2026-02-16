import config
import re
import numpy as np
from utils import *
from tqdm import tqdm

def load_bgl_dataset(log_name="bgl"):
	data_filename = os.path.join(config.cpu_data_path, "BGL.log")
	log_data = []
	with open(data_filename, 'r', encoding="utf-8") as f:
		for line in f.readlines():
			line_ = line.strip()
			pos = line_.find(" ")
			raw_label = line_[:pos]
			log = line_[pos+1:]
			header_items = line_.split(" ")
			timestamp = int(header_items[1])
			entities = {}
			if len(header_items[3])>=2: entities["node_id"] = [header_items[3]]
			label = 0 if raw_label=='-' else 1
			log_data.append([log, label, timestamp, entities])
	log_data.sort(key=lambda x:x[2])
	return log_data

def load_tdb_dataset(log_name="tdb"):
	data_filename = os.path.join(config.raw_dataset_path, "TDB.log")
	log_data = []
	with open(data_filename, 'r', encoding="utf-8") as f:
		for line in f.readlines():
			line_ = line.strip()
			pos = line_.find(" ")
			raw_label = line_[:pos]
			log = line_[pos+1:]
			header_items = line_.split(" ")
			timestamp = int(header_items[1])
			entities = {}
			if len(header_items[3])>=2: entities["user_id"] = [header_items[3]]
			if len(header_items)>8 and len(header_items[8])>=2:
				entities["compo_id"] = [header_items[8]] # component_id
			label = 0 if raw_label=='-' else 1
			log_data.append([log, label, timestamp, entities])
	log_data.sort(key=lambda x:x[2])
	return log_data

def load_hdfs_dataset(log_name="hdfs"):
	data_filename = os.path.join(config.raw_dataset_path, "HDFS.log")
	label_filename = os.path.join(config.raw_dataset_path, "HDFS_anomaly_label.csv")
	data = read_csv(label_filename, encoding="utf-8")
	label_map = {}
	for blk_id, label in data[1:]:
		label_map[blk_id] = 1 if label[0]=='A' else 0
	log_data = []
	with open(data_filename, 'r', encoding="utf-8") as f:
		for line in tqdm(f.readlines()):
			log = line.strip()
			blocks = re.findall(r'blk_-?\d+',log)
			label = 0
			entities = {}
			if len(blocks)>0: entities["block_id"] = []
			for b in blocks:
				label |= label_map.get(b, 0)
				entities["block_id"].append(b)
			timestamp = str2ts(log[:13],formats="%y%m%d %H%M%S")
			log_data.append([log, label, timestamp, entities])
	log_data.sort(key=lambda x:x[2])
	return log_data

log_dataset_processor = {
	"bgl": load_bgl_dataset,
	"tdb": load_tdb_dataset,
	"hdfs": load_hdfs_dataset,
}

def load_dataset(log_name="bgl"):
	if log_name not in log_dataset_processor:
		raise NotImplementedError(log_name)
		return None
	print("Loading %s dataset..."%(log_name))
	log_data = log_dataset_processor[log_name]()
	return log_data

def get_template_key_and_masked_log(log):
	masked_log = re.sub(r'\d+', ' ', log.lower())
	tmpl_key = int(hash_func(masked_log)[-8:], 16)
	return tmpl_key, masked_log

def tokenize(masked_log, vocab):
	word_list = [w for w in re.split(r'\W+', masked_log) if len(w)>=config.min_word_length]
	return word_list

def log_tokenization(log_data, vocab, template_map=dict()):
	tmpl_list = []
	print("Tokenize logs...")
	for line in tqdm(log_data):
		log = line[0]
		tmpl_key, masked_log = get_template_key_and_masked_log(log)
		tmpl_list.append(tmpl_key)
		if tmpl_key in template_map:
			continue
		word_list = tokenize(masked_log, vocab)
		template_map[tmpl_key] = [vocab.word2id(w) for w in word_list if w in vocab]
	return tmpl_list, template_map



