import config
import re
import os
import numpy as np
from utils import *
from tqdm import tqdm
from data_preprocess import load_dataset

HOUR = 3600 # seconds

class LogEntityGraph(object):
	FORMAT_SPACE = re.compile(r'\s+') #(multiple) spaces, new lines
	MAX_TIME_INTERVAL = int(config.max_time_interval*HOUR)
	THRESHOLD_TOKEN_COMPLEXITY = 2
	THRESHOLD_RECURRENCE_FREQUENCY = 2
	MIN_TOKEN_LENGTH = 2
	CASE_SENSITIVE = False
	ENTITY_FILTER = [
		re.compile(r'^\d+-\d+-\d+-\d+\.\d+\.\d+\.\d+$'), # (correlation / equivalent to timestamp), 4 groups seperated by hyphens and dots: timestamp bgl: 1117838570-2005-06-03-15.42.50.675872
	]

	def __init__(self, log_name="bgl", alias=""):
		self.entity_filter = []
		self.enable_temporal_link = True
		self.graph_file_name = os.path.join(config.data_package_path, "graph_struct_%s@%s.obj"%(log_name,alias))
		self.init_neighbor_map()

	def __str__(self):
		return "[log entity graph] number of nodes: %d , average degree: %.2lf , number of entities: %d \n" % (self.num_nodes, self.avg_degree, len(self.previous_entity))

	def init_neighbor_map(self):
		self.neighbor = {} 
		self.entity_index_map = {}
		return self

	def gen_statistics(self):
		self.num_nodes = len(self.neighbor)
		self.avg_degree = sum([len(v) for v in self.neighbor.values()]) / max(self.num_nodes, 1)

	def calc_token_complexity(self, token):
		token_complexity = 0
		last_token_type = -1
		for char in token:
			if char.isdigit(): token_type = 0
			elif char.isalpha():
				if self.CASE_SENSITIVE and char.isupper(): token_type = 3
				else: token_type = 1
			else: token_type = 2
			if token_type!=last_token_type:
				token_complexity += 1
			last_token_type = token_type
		return token_complexity

	def extract_entities(self, log):
		entity_set = set()
		if self.enable_temporal_link:
			entity_set.add("")
		all_tokens = re.split(self.FORMAT_SPACE, log) #split log into tokens(by spaces, new lines)
		for token in all_tokens:
			if len(token) < self.MIN_TOKEN_LENGTH:
				continue
			if token[-1].isalnum()==False: # remove the last punctuation, if it is not alphanumeric
				token = token[:-1]
			drop_entity = False
			for e in self.ENTITY_FILTER:
				if re.match(e, token): #check if token matches any of the entity filters: timestamp
					drop_entity = True
					break
			if drop_entity==True:
				continue
			token_complexity = self.calc_token_complexity(token)
			if token_complexity >= self.THRESHOLD_TOKEN_COMPLEXITY: # here tokens for logs like error, get dropped and only entites get added because they would have numers and chars, chars and special chars etc.. so complex >= 2
				entity_set.add(token)
		return entity_set #Return a set of distinct entities for this log.

	def add_edge(self, idx, prev_idx, ent_idx):
		if idx not in self.neighbor:
			self.neighbor[idx] = []
		self.neighbor[idx].append((ent_idx, prev_idx))
		return

	def build(self, log_data):
		self.init_neighbor_map() # neighbor = {}, entity_index_map = {}
		self.previous_entity = {} # entity_id -> last log index where it appeared
		self.tempo_map = {} # log_idx -> timestamp
		for idx, line in tqdm(enumerate(log_data)): #for each log index
			log, _, timestamp, _ = line
			self.tempo_map[idx] = timestamp
			entity_set = self.extract_entities(log) #get all entities for this log
			for entity in entity_set:
				if entity in self.entity_index_map:
					eid = self.entity_index_map.get(entity)
					prev_idx = self.previous_entity.get(eid, -1) #if not in previous_entity, set to -1
					if 0 <= self.tempo_map[idx] - self.tempo_map[prev_idx] <= self.MAX_TIME_INTERVAL:
						self.add_edge(idx, prev_idx, eid) #add edge between current log and previous log where this entity occurred.
				else:
					eid = len(self.entity_index_map)
					self.entity_index_map[entity] = eid
				self.previous_entity[eid] = idx #prev_idx = last log index where this entity occurred.
		self.gen_statistics()
		return self

	def fetch_subgraph(self, smpl_index, max_history=20):
		l2e_map, e2l_map = {}, {}
		entity_tmpl_map = {}

		def _collect_history_logs(cur_idx, ent_idx, prev_idx, frequency=0):
			if frequency >= max_history:
				entity_tmpl_map[ent_idx] = tmpl_list
				return True
			tmpl_list.append(self.tmpl_list[cur_idx])
			if prev_idx is None:
				if frequency >= self.THRESHOLD_RECURRENCE_FREQUENCY:
					entity_tmpl_map[ent_idx] = tmpl_list
					return True
				return False
			prev_idx2 = None
			if prev_idx in self.neighbor:
				for item in self.neighbor[prev_idx]:
					if item[0]==ent_idx:
						prev_idx2 = item[1]
						break
			return _collect_history_logs(prev_idx, ent_idx, prev_idx2, frequency+1)

		for i, log_idx in enumerate(smpl_index[::-1]):
			l2e_map[log_idx] = []
			if log_idx not in self.neighbor:
				continue
			for ent_idx, prev_idx in self.neighbor[log_idx]:
				tmpl_list = []
				valid_entity = _collect_history_logs(log_idx, ent_idx, prev_idx)
				if valid_entity==True:
					if ent_idx not in e2l_map:
						e2l_map[ent_idx] = []
					e2l_map[ent_idx].append(log_idx)
					l2e_map[log_idx].append(ent_idx)
		return l2e_map, e2l_map, entity_tmpl_map

	def apply_template_mapping(self, tmpl_list):
		self.tmpl_list = tmpl_list

	def build_index_entity_map(self):
		self.index_entity_map = {
			v:k for k,v in self.entity_index_map.items()
		}
		return self

	def get_entity_name(self, ent_id):
		return self.index_entity_map.get(ent_id, "")

	def save(self, file_name=None):
		if file_name is None:
			file_name = self.graph_file_name
		save_object([self.neighbor,self.entity_index_map],file_name)
		return

	def load(self, file_name=None):
		if file_name is None:
			file_name = self.graph_file_name
		self.neighbor,self.entity_index_map = load_object(file_name)
		self.gen_statistics()
		return self

	def show_log_entity_graph_statistics(self):
		group_by_entity = {}
		for log_idx, edge_list in self.neighbor.items():
			for ent_idx, prev_idx in edge_list:
				if ent_idx not in group_by_entity:
					group_by_entity[ent_idx] = 0
				group_by_entity[ent_idx] += 1
		freq_ent_set = set([k for k,v in group_by_entity.items() if v>=2])
		group_by_entity = {k:v for k,v in group_by_entity.items() if k in freq_ent_set}
		group_by_log = {}
		for log_idx, edge_list in self.neighbor.items():
			group_by_log[log_idx] = 0
			for ent_idx, prev_idx in edge_list:
				if ent_idx in freq_ent_set:
					group_by_log[log_idx] += 1

		print_function = print
		print_function("# Log Nodes:",len(group_by_log))
		print_function("# Entity Nodes:",len(group_by_entity))
		print_function("# Log->Entity Edges:",sum(group_by_log.values()))
		print_function("# Entity->Log Edges:",sum(group_by_entity.values()))
		print_function("# Log-Log Edges:",len(group_by_log)-1)
		print_function("# Total Edges:", (len(group_by_log)-1)+(sum(group_by_log.values())+sum(group_by_entity.values()))//2)
		print_function("Avg. Degree Log:",np.mean(list(group_by_log.values()))+2)
		print_function("Avg. Degree Entity:",np.mean(list(group_by_entity.values())))
		print_function("Max Degree Log:",np.max(list(group_by_log.values()))+2)
		print_function("Max Degree Entity:",np.max(list(group_by_entity.values())))
		return

if __name__=="__main__":
	log_data = load_dataset("bgl")
	log_entity_graph = LogEntityGraph("bgl")
	log_entity_graph.build(log_data)
	#log_entity_graph.save()
	#log_entity_graph.load()
	log_entity_graph.show_log_entity_graph_statistics()

