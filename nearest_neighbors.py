import config
import torch
import numpy as np
from module import EmbeddingLayer
from sklearn.neighbors import KDTree

class LogTemplateReprCache():
	def __init__(self, vocab, embed_layer):
		self.template_list = []
		self.feature_list = []
		self.tmpl_index_map = {}
		self.embed_layer = embed_layer
		self.kd_tree = None

	def feed(self, template_map):
		self.template_list = []
		self.feature_list = []
		for tmpl_id, word_list in template_map.items():
			self.tmpl_index_map[tmpl_id] = len(self.template_list)
			self.template_list.append(tmpl_id)
			word_tensor = torch.LongTensor(word_list)
			agg_repr = self.embed_layer.forward_pretrain(word_tensor)
			agg_repr = agg_repr.detach().numpy().reshape(-1).tolist()
			self.feature_list.append(agg_repr)
		self.feature_list = np.array(self.feature_list)
		self.kd_tree =  KDTree(self.feature_list)
		return self

	def get_template_repr(self, tmpl_id):
		line_id = self.tmpl_index_map[tmpl_id]
		return self.feature_list[line_id]
