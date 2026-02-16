import config
import re
import os
import numpy as np
from utils import *
from tqdm import tqdm
from data_preprocess import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import json

HOUR = 3600 # seconds
def save_graph(graph, step):
	G = nx.Graph()
	debug_lines = []
	# Add log nodes
	for log_idx in graph.neighbor.keys():
		G.add_node(f"log_{log_idx}", type="log", label=f"L{log_idx}")
		debug_lines.append(f"Added log node: log_{log_idx}")
	# Add entity nodes
	if hasattr(graph, 'entity_index_map'):
		for entity, eid in graph.entity_index_map.items():
			G.add_node(f"entity_{eid}", type="entity", label=f"E{eid}")
			debug_lines.append(f"Added entity node: entity_{eid} ({entity})")
		# Track edge types for different styling
		log_entity_edges = []
		log_log_edges = []
	# Add edges (log <-> entity)
	for log_idx, neighbors in graph.neighbor.items():
		for ent_idx, prev_idx in neighbors:
			log_entity_edges.append((f"log_{log_idx}", f"entity_{ent_idx}"))
			debug_lines.append(f"Added edge: log_{log_idx} -- entity_{ent_idx}")
			if prev_idx != -1:
				G.add_node(f"log_{prev_idx}", type="log")
				log_log_edges.append((f"log_{log_idx}", f"log_{prev_idx}"))
				debug_lines.append(f"Added edge: log_{log_idx} -- log_{prev_idx}")
	G.add_edges_from(log_entity_edges)
	G.add_edges_from(log_log_edges)
	# Write debug info to file
	with open("graph_debug_step_%d.txt" % step, "w") as dbg:
		for line in debug_lines:
			dbg.write(line + "\n")
	# Prepare labels and colors
	log_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "log"]
	entity_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "entity"]
	# CHRONOLOGICAL LAYOUT: Logs on left (vertical), entities on right
	pos = {}
	log_y_spacing = 2.0
	entity_y_spacing = 2.0
	# Sort logs by index (chronological order)
	sorted_logs = sorted(log_nodes, key=lambda x: int(x.split('_')[1]))
	for i, log_node in enumerate(sorted_logs):
		pos[log_node] = (0, -i * log_y_spacing)  # Logs on the left
	# Sort entities by ID
	sorted_entities = sorted(entity_nodes, key=lambda x: int(x.split('_')[1]))
	for i, entity_node in enumerate(sorted_entities):
		pos[entity_node] = (5, -i * entity_y_spacing)  # Entities on the right
	# Create figure with better size and DPI
	plt.figure(figsize=(16, 12), dpi=150)
	# Draw entity nodes (circles) - coral/salmon color
	nx.draw_networkx_nodes(
        G, pos, 
        nodelist=entity_nodes, 
        node_color="#FF6B6B",  # Coral red
        node_shape='o', 
        node_size=3000,
        alpha=0.9,
        edgecolors='black',
        linewidths=2.5
    )
	# Draw log nodes (rectangles) - teal/cyan
	nx.draw_networkx_nodes(
        G, pos, 
        nodelist=log_nodes, 
        node_color="#4ECDC4",  # Teal/cyan
        node_shape='s', 
        node_size=3000,
        alpha=0.9,
        edgecolors='black',
        linewidths=2.5
    )
	# Draw log-entity edges (solid gray)
	nx.draw_networkx_edges(
        G, pos, 
        edgelist=log_entity_edges,
        edge_color='#757575',  # Dark gray
        width=2,
        alpha=0.6,
        style='solid'
    )
	# Draw log-log edges (dashed blue) - temporal links
	nx.draw_networkx_edges(
        G, pos, 
        edgelist=log_log_edges,
        edge_color='#1E88E5',  # Blue
        width=3,
        alpha=0.8,
        style='dashed'
    )
	# ADD LABELS - Extract from node data
	labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
	nx.draw_networkx_labels(
        G, pos, 
        labels, 
        font_size=14,  # Bigger font
        font_weight='bold',
        font_family='sans-serif',
        font_color='white'  # White text on colored nodes
    )
	plt.title(f"Log-Entity Graph at step {step} (Chronological Layout)", fontsize=18, fontweight='bold', pad=20)
	# Create legend
	legend_elements = [
        Patch(facecolor='#4ECDC4', edgecolor='black', label='Log Node', linewidth=2),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Entity Node', linewidth=2),
        Line2D([0], [0], color='#757575', linewidth=2, label='Log-Entity Edge'),
        Line2D([0], [0], color='#1E88E5', linewidth=3, linestyle='--', label='Temporal Link (Log-Log)')
    ]
	plt.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
	plt.tight_layout()
	plt.savefig(f"graph_step_{step}.png", dpi=150, bbox_inches='tight', facecolor='white')
	plt.close()

def save_graph_data(graph, step):
    with open(f"graph_step_{step}.json", "w") as f:
        json.dump(graph.neighbor, f)

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

	def add_edge(self, idx, prev_idx, ent_idx, save_visualization=False, max_visualizations=5):
		if idx not in self.neighbor:
			self.neighbor[idx] = []
		self.neighbor[idx].append((ent_idx, prev_idx))
		# Only save visualization for first N steps
		if save_visualization and idx < max_visualizations:
			save_graph(self, step=idx)
			save_graph_data(self, step=idx)
		return

	def build(self, log_data, visualize_first_n=0):
		self.init_neighbor_map() # neighbor = {}, entity_index_map = {}
		self.previous_entity = {} # entity_id -> last log index where it appeared
		self.tempo_map = {} # log_idx -> timestamp
		self.log_content = {}  # ADD THIS: Store actual log content
		for idx, line in tqdm(enumerate(log_data)): #for each log index
			log, _, timestamp, _ = line
			self.tempo_map[idx] = timestamp
			self.log_content[idx] = log  # Store the actual log content
			# ADD DEBUG PRINTING FOR FIRST N LOGS
			
			entity_set = self.extract_entities(log) #get all entities for this log
			if visualize_first_n > 0 and idx < visualize_first_n:
				print(f"\n{'='*80}")
				print(f"Processing Log {idx}:")
				print(f"Content: {log[:200]}...")  # First 200 chars
				print(f"Timestamp: {timestamp}")
				print(f"Extracted entities: {entity_set}")
			save_viz = visualize_first_n > 0
			for entity in entity_set:
				if entity in self.entity_index_map:
					eid = self.entity_index_map.get(entity)
					prev_idx = self.previous_entity.get(eid, -1) #if not in previous_entity, set to -1
					if 0 <= self.tempo_map[idx] - self.tempo_map[prev_idx] <= self.MAX_TIME_INTERVAL:
						self.add_edge(idx, prev_idx, eid, save_viz, visualize_first_n) #add edge between current log and previous log where this entity occurred.
				else:
					eid = len(self.entity_index_map)
					self.entity_index_map[entity] = eid
					if visualize_first_n > 0 and idx < visualize_first_n:
						print(f"  NEW Entity '{entity}' assigned eid={eid}")
					self.add_edge(idx, -1, eid, save_viz, visualize_first_n)
				self.previous_entity[eid] = idx #prev_idx = last log index where this entity occurred.
		self.gen_statistics()
		# ADD THIS: Save log content summary after building
		if visualize_first_n > 0:
			self.save_log_summary(visualize_first_n)
		return self
	
	def save_log_summary(self, n_logs):
		"""Save a summary of the first N logs and their connections"""
		with open("log_summary.txt", "w") as f:
			f.write("="*80 + "\n")
			f.write("LOG SUMMARY\n")
			f.write("="*80 + "\n\n")
			
			for idx in range(min(n_logs, len(self.log_content))):
				f.write(f"\n{'='*80}\n")
				f.write(f"LOG {idx}\n")
				f.write(f"{'='*80}\n")
				f.write(f"Content: {self.log_content.get(idx, 'N/A')}\n")
				f.write(f"Timestamp: {self.tempo_map.get(idx, 'N/A')}\n\n")
				
				if idx in self.neighbor:
					f.write(f"Connections ({len(self.neighbor[idx])} edges):\n")
					for ent_idx, prev_idx in self.neighbor[idx]:
						entity_name = self.get_entity_name_by_id(ent_idx)
						f.write(f"  - Entity {ent_idx} ('{entity_name}')\n")
						if prev_idx != -1:
							f.write(f"    -> Links to previous log {prev_idx}\n")
							if prev_idx in self.log_content:
								f.write(f"       Previous log content: {self.log_content[prev_idx][:100]}...\n")
				else:
					f.write("No connections (isolated node)\n")
		
		print(f"\nLog summary saved to log_summary.txt")

	def get_entity_name_by_id(self, ent_id):
		"""Helper to get entity name from entity ID without needing index_entity_map"""
		for entity, eid in self.entity_index_map.items():
			if eid == ent_id:
				return entity if entity else "[temporal_link]"
		return "unknown"

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

