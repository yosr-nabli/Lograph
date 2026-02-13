import config
from module import *
from torch.nn import CrossEntropyLoss, MSELoss

class LogEntityLogAggregationLayer(nn.Module):
	def __init__(self, embedding_dim=300, hidden_size=128, num_layers=2, atten_size=32, 
							bidrectional=True, multi_head=2, use_lstm=False, use_gat=True):
		super(LogEntityLogAggregationLayer, self).__init__()
		self.embedding_dim = embedding_dim
		self.graph_atten_layer = GraphAttentionLayer(embedding_dim, atten_size) if use_gat==True else EmptyLayer()
		self.model_type = nn.LSTM if use_lstm==True else nn.GRU
		self.multi_head = multi_head
		self.seq_model = self.model_type(self.embedding_dim,
					hidden_size,
					num_layers,
					bidirectional=bidrectional,
					batch_first=True)

	def forward(self, log_repr, log_adj_matrices):
		log_repr = self.graph_atten_layer.forward(log_repr, log_adj_matrices)
		output, _ = self.seq_model(log_repr)
		att_log_reprs = output[:, -1, :] #inal hidden state at the last position in the window, log-path representation” of the whole window.
		return att_log_reprs

class LogEntityLogTransformerLayer(nn.Module):
	def __init__(self, embedding_dim=300, hidden_size=128, num_layers=2, atten_size=32, 
							bidrectional=True, multi_head=2, use_lstm=False, use_gat=True):
		super(LogEntityLogTransformerLayer, self).__init__()
		self.embedding_dim = embedding_dim
		self.graph_atten_layer = GraphAttentionLayer(embedding_dim, atten_size) if use_gat==True else EmptyLayer()
		self.model_type = nn.LSTM if use_lstm==True else nn.GRU
		self.multi_head = multi_head
		self.bidrection_coef = 2 if bidrectional==True else 1
		self.transformer_block = nn.TransformerEncoderLayer(
					d_model=self.embedding_dim,
					nhead=multi_head,
					dim_feedforward=256,
					batch_first=True,
					dropout=0.1)
		self.dense = nn.Linear(self.embedding_dim, hidden_size*self.bidrection_coef)
		self.activate_func = nn.ELU()

	def forward(self, log_repr, log_adj_matrices):
		log_repr = self.graph_atten_layer.forward(log_repr, log_adj_matrices)
		output = self.transformer_block(log_repr).mean(dim=1)
		output = self.activate_func(output)
		att_log_reprs = self.activate_func(self.dense(output))
		return att_log_reprs

class EntityLogEntityAggregationLayer(nn.Module):
	def __init__(self, embedding_dim=300, hidden_size=128, atten_size=32, bidrection_coef=2, use_gat=True):
		super(EntityLogEntityAggregationLayer, self).__init__()
		self.atten_size = atten_size
		self.hidden_size = hidden_size
		self.bidrection_coef = bidrection_coef
		self.graph_atten_layer = GraphAttentionLayer(embedding_dim, atten_size) if use_gat==True else EmptyLayer()
		self.atten_layer = SelfAttentionLayer(embedding_dim, self.bidrection_coef*hidden_size)

	def forward(self, entity_reprs, adjacency_matrices):
		agg_entity_reprs = self.graph_atten_layer.forward(entity_reprs, adjacency_matrices, return_list=True)
		all_attenion_score = []
		attention_entity_reprs = torch.FloatTensor(len(entity_reprs), self.bidrection_coef*self.hidden_size).fill_(0)
		for i, (ent_repr, agg_ent_repr) in enumerate(zip(entity_reprs, agg_entity_reprs)):
			#If a sample has zero entities, we:
#Append an empty list of attention scores.
#Leave that row in attention_entity_reprs as zeros.
#Skip to the next sample.
			if ent_repr.shape[0]==0:
				all_attenion_score.append([])
				continue
			att_entity_repr_, attenion_score = self.atten_layer(agg_ent_repr)
			attention_entity_reprs[i] = att_entity_repr_
			all_attenion_score.append(attenion_score)
		return attention_entity_reprs, all_attenion_score

class SemanticAggregationLayer(nn.Module):
	def __init__(self, hidden_size, bidrection_coef, use_meta_path=[True, True]):
		super(SemanticAggregationLayer, self).__init__()
		self.num_meta_path = sum([1 if flag==True else 0 for flag in use_meta_path])
		self.output_dim = hidden_size*bidrection_coef*self.num_meta_path

	def forward(self, att_log_reprs=None, att_entity_reprs=None):
		if att_log_reprs is None:
			semantic_agg = att_entity_reprs
		elif att_entity_reprs is None:
			semantic_agg = att_log_reprs
		else:
			semantic_agg = torch.cat((att_log_reprs, att_entity_reprs), axis=1)
		return semantic_agg

class LographEmbeddingLayer(nn.Module):
	def __init__(self, log_entity_graph=None, embed_layer=None, template_cache=None, use_meta_path=[True,True]):
		super(LographEmbeddingLayer, self).__init__()
		self.log_entity_graph = log_entity_graph
		self.embed_layer = embed_layer
		self.template_cache = template_cache
		self.embedding_dim = embed_layer.embedding_dim
		self.use_log_repr = use_meta_path[0]
		self.use_ent_repr = use_meta_path[1]

	def generate_entity_embed_and_adj_matrix(self, indice):
		l2e_map, e2l_map, entity_tmpl_map = self.log_entity_graph.fetch_subgraph(indice)
		ent_list = list(entity_tmpl_map.keys())
		ent2id = {x:i for i,x in enumerate(ent_list)}
		num_ent = len(ent2id)
		adjacency_matrix = np.eye(num_ent)
		for i, u in enumerate(ent_list):
			for l in e2l_map[u]:
				for v in l2e_map[l]:
					j = ent2id[v]
					adjacency_matrix[i, j] += 1
		ent_repr = []
		for ent_id in entity_tmpl_map:
			avg_repr = np.zeros(self.embedding_dim, dtype=float)
			for k in entity_tmpl_map[ent_id]:
				tmpl_repr = self.template_cache.get_template_repr(k)
				avg_repr += tmpl_repr
			avg_repr /= max(len(entity_tmpl_map[ent_id]), 1)
			ent_repr.append(avg_repr.tolist())
		return ent_repr, adjacency_matrix, ent_list

	def generate_adjacency_matrix(self, indice):
		l2e_map, e2l_map, entity_tmpl_map = self.log_entity_graph.fetch_subgraph(indice)
		log2id = {x:i for i,x in enumerate(indice)}
		seq_len = len(indice)
		adjacency_matrix = np.eye(seq_len) #identity matrix: each log connected to itself
		for i, u in enumerate(indice):
			for e in l2e_map[u]: # entities of log u
				for v in e2l_map[e]: # logs connected to entity e
					j = log2id[v]
					adjacency_matrix[i, j] += 1
		return adjacency_matrix

	def forward(self, inputs):
		words, labels, groups, masks, indices = inputs[:5]
		if self.use_log_repr==True:
			log_repr, labels = self.embed_layer(inputs)
			log_adj_matrices = []
			for i, indice in enumerate(indices):
				adj_matrix = self.generate_adjacency_matrix(indice)
				log_adj_matrices.append(torch.FloatTensor(adj_matrix))
		else:
			log_repr, log_adj_matrices = None, None
		if self.use_ent_repr==True:
			entity_reprs, ent_adj_matrices, ent_indices = [], [], []
			for i, indice in enumerate(indices):
				ent_repr, adj_matrix, ent_index = self.generate_entity_embed_and_adj_matrix(indice)
				ent_adj_matrices.append(torch.FloatTensor(adj_matrix))
				ent_repr = torch.FloatTensor(ent_repr)
				entity_reprs.append(ent_repr)
				ent_indices.append(ent_index)
		else:
			entity_reprs, ent_adj_matrices, ent_indices = None, None, None
		return log_repr, log_adj_matrices, entity_reprs, ent_adj_matrices, ent_indices

	def fetch_entity_names(self, ent_indices):
		entity_names = []
		for ent_batch in ent_indices:
			entity_names.append([self.log_entity_graph.get_entity_name(e_id) for e_id in ent_batch])
		return entity_names

class Lograph(PyTorchModule):
	def __init__(self, log_entity_graph=None, embed_layer=None, template_cache=None, hidden_size=100, atten_size=16,
							num_layers=2, bidrectional=True, use_meta_path=[True,True], alias=""):
		super(Lograph, self).__init__()
		#number of neurons 128
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.bidrection_coef = 2 if bidrectional==True else 1
		self.embedding_dim = embed_layer.embedding_dim #dim 300
		self.lograph_embed_layer = LographEmbeddingLayer(log_entity_graph, embed_layer, template_cache, use_meta_path)
		LogLayerModel = LogEntityLogTransformerLayer if config.use_transformer==True else LogEntityLogAggregationLayer
		self.log_entity_log_layer = LogLayerModel(self.embedding_dim, hidden_size, num_layers, atten_size,
						bidrectional, use_lstm=config.use_lstm_layer, use_gat=config.use_gat_layer) if use_meta_path[0]==True else EmptyLayer()
		self.entity_log_entity_layer = EntityLogEntityAggregationLayer(self.embedding_dim, hidden_size, atten_size,
						self.bidrection_coef, use_gat=config.use_gat_layer) if use_meta_path[1]==True else EmptyLayer(1)
		
		self.semantic_agg_layer = SemanticAggregationLayer(hidden_size, self.bidrection_coef, use_meta_path)
		self.classifier = nn.Linear(self.semantic_agg_layer.output_dim, 2)
		self.set_model_name("Lograph@%s"%(alias))
		self.loss_func = CrossEntropyLoss(reduction="mean")
		self.task = "binary_class"

	def forward(self, inputs):
		log_repr, log_adjmtx, entity_repr, ent_adjmtx, ent_indices = self.lograph_embed_layer(inputs)
		att_log_reprs = self.log_entity_log_layer(log_repr, log_adjmtx)
		att_entity_reprs, all_att_score = self.entity_log_entity_layer(entity_repr, ent_adjmtx)
		agg_reprs = self.semantic_agg_layer(att_log_reprs, att_entity_reprs)
		output = self.classifier(agg_reprs)
		proba = softmax(output, dim=1)
		return proba

	def collect_atten_score(self, inputs):
		words, labels, groups, masks, indices = inputs[:5]
		log_repr, log_adjmtx, entity_repr, ent_adjmtx, ent_indices = self.lograph_embed_layer(inputs)
		att_log_reprs = self.log_entity_log_layer(log_repr, log_adjmtx)
		att_entity_reprs, all_att_score = self.entity_log_entity_layer(entity_repr, ent_adjmtx)
		entity_names = self.lograph_embed_layer.fetch_entity_names(ent_indices)
		agg_reprs = self.semantic_agg_layer(att_log_reprs, att_entity_reprs)
		output = self.classifier(agg_reprs)
		proba = softmax(output, dim=1)
		return proba, labels, entity_names, all_att_score


