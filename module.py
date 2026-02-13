from utils import *
import config
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import embedding, softmax

class PyTorchModule(nn.Module):
	def __init__(self):
		super(PyTorchModule, self).__init__()
		self.model_path = "./model.pkl"
		self.training_params = {}
		self.evaluation_score = 0
		self.name = "Undefined"
		self.loss_func = None
		self.drop_positive_samples = False

	def record_training_params(self, param_dict):
		self.training_params = {
			k:v for k,v in param_dict.items()
		}

	def record_model_params(self, param_list):
		print("Model Params: ", param_list)
		self.model_params = [v for v in param_list]

	def record_evaluation_score(self, score):
		self.evaluation_score = score

	def set_model_name(self, model_name):
		self.name = model_name
		self.attach_to_file(os.path.join(config.torch_model_path, "%s.pkl"%(self.name)))

	def attach_to_file(self, file_path):
		self.model_path = file_path

	def auto_save(self, file_path=None):
		if file_path is None: file_path = self.model_path
		save_object([self.state_dict(), self.training_params, self.evaluation_score], file_path)

	def reload(self, file_path=None):
		if file_path is None: file_path = self.model_path
		data_package = load_object(file_path)
		loaded_state_dict, self.training_params, self.evaluation_score = load_object(file_path)
		self.load_state_dict(loaded_state_dict)

	def calculate_loss(self, inputs, proba):
		return self.loss_func(proba, inputs[1])

class EmbeddingLayer(nn.Module):
	def __init__(self, vocab, padding_idx=None):
		super(EmbeddingLayer, self).__init__()
		self.vocab = vocab
		self.vocab_size = vocab.vocab_size
		self.embedding_dim = vocab.word_dim
		self.padding_idx = padding_idx
		self.weight = Parameter(torch.Tensor(self.vocab_size, self.embedding_dim))
		self.weight.data.copy_(torch.from_numpy(vocab.embeddings))
		self.weight.requires_grad = False

	def forward(self, inputs):
		reprs = embedding(inputs, self.weight, self.padding_idx).to(config.device)
		return reprs

class WordAggregateLayer(nn.Module):
	def __init__(self, vocab, dimension = 2):
		super(WordAggregateLayer, self).__init__()
		self.embed_layer = EmbeddingLayer(vocab)
		self.embedding_dim = vocab.word_dim
		self.dimension = dimension

	def set_dimension(self, dim=2):
		self.dimension = dim
		return self

	def forward_pretrain(self, words):
		words = words.reshape([1,1,-1])
		reprs = self.embed_layer(words)
		column_sum = torch.sum(masks, axis=2)
		column_sum[column_sum==0] = 1
		agg_reprs = torch.sum(reprs, axis=2)/column_sum.reshape([masks.shape[0],-1,1])
		agg_reprs = torch.mean(agg_reprs, axis=1)
		return agg_reprs

	def forward(self, inputs):
		words, labels, groups, masks = inputs[:4]
		reprs = self.embed_layer(words)
		masks = torch.Tensor(words.shape).fill_(1)
		column_sum = torch.sum(masks, axis=2)
		column_sum[column_sum==0] = 1
		agg_reprs = torch.sum(reprs, axis=2)/column_sum.reshape([masks.shape[0],-1,1])
		if self.dimension==1:
			agg_reprs = torch.mean(agg_reprs, axis=1)
		return agg_reprs, labels

class WordAggregateTfIdfLayer(nn.Module):
	def __init__(self, vocab, dimension = 2):
		super(WordAggregateTfIdfLayer, self).__init__()
		self.embed_layer = EmbeddingLayer(vocab) # GloVe lookup
		self.embedding_dim = vocab.word_dim# e.g., 300
		self.dimension = dimension
		self.idf_counter = {}  #IDF values per word
		self.total_count = 0 # total number of logs
		self.oov_idf_value = 0 # IDF for unknown words

	def set_dimension(self, dim=2):
		self.dimension = dim
		return self

	def feed(self, data_loader):
		#Called once before training to compute IDF statistics:
		self.idf_counter = {}
		self.total_count = 0
		# print("Running TF-IDF algorithm...")
		for batch in data_loader:
			words, labels, groups, masks = batch[:4]
			batch_size, seq_len, num_word = words.shape
			word_list = words.detach().numpy().tolist()
			mask_list = masks.detach().numpy().tolist()
			for i in range(batch_size):
				for j in range(seq_len):
					word_set = set()
					for k,(w,m) in enumerate(zip(word_list[i][j], mask_list[i][j])):
						if m==0: break
						word_set.add(w)
					for w in word_set:
						self.idf_counter[w] = self.idf_counter.get(w, 0) + 1
			self.total_count += batch_size
		for w,c in self.idf_counter.items():
			self.idf_counter[w] = np.log((self.total_count+1)/(self.idf_counter[w]+1) + 1)
		self.oov_idf_value = np.log(max(self.total_count, 1))
		return self

	def calc_idf_matrix(self, words, masks):
		#It looks up IDF from self.idf_counter (already computed in feed())
		batch_size, seq_len, num_word = words.shape
		word_list = words.detach().numpy().tolist()
		mask_list = masks.detach().numpy().tolist()
		idf_matrix = torch.Tensor(batch_size, seq_len, num_word, 1).fill_(0)
		for i in range(batch_size):
			for j in range(seq_len):
				idf_values = []
				for k,(w,m) in enumerate(zip(word_list[i][j], mask_list[i][j])):
					if m==0: break
					idf_matrix[i, j, k, 0] = self.idf_counter.get(w, self.oov_idf_value)
					word = self.embed_layer.vocab.id2word(w)
		return idf_matrix

	def forward_pretrain(self, words):
		words = words.reshape([1,1,-1])
		reprs = self.embed_layer(words)
		masks = torch.Tensor(words.shape).fill_(1)
		idf_matrix = self.calc_idf_matrix(words, masks)
		column_sum = torch.sum(masks, axis=2)
		column_sum[column_sum==0] = 1
		agg_reprs = torch.sum(reprs*idf_matrix, axis=2)/column_sum.reshape([masks.shape[0],-1,1])
		agg_reprs = torch.mean(agg_reprs, axis=1)
		return agg_reprs

	def forward(self, inputs):
		words, labels, groups, masks = inputs[:4]
		reprs = self.embed_layer(words)
		idf_matrix = self.calc_idf_matrix(words, masks)
		column_sum = torch.sum(masks, axis=2)
		column_sum[column_sum==0] = 1
		agg_reprs = torch.sum(reprs*idf_matrix, axis=2)/column_sum.reshape([masks.shape[0],-1,1])
		if self.dimension==1:
			agg_reprs = torch.mean(agg_reprs, axis=1)
		return agg_reprs, labels

class EmptyLayer(nn.Module):
	def __init__(self, output_padding=0):
		super(EmptyLayer, self).__init__()
		self.output_padding = output_padding

	def forward(self, inputs, *args):
		if self.output_padding==0: return inputs
		else: return [inputs] + [None]*output_padding

class GraphAttentionLayer(nn.Module):
	def __init__(self, embedding_dim=300, atten_size=32):
		super(GraphAttentionLayer, self).__init__()
		self.embedding_dim = embedding_dim
		self.Wq = Parameter(torch.FloatTensor(embedding_dim, atten_size))
		self.Wk = Parameter(torch.FloatTensor(embedding_dim, atten_size))
		self.Wq.data.normal_(mean = 0, std = 0.01) #Initialize both matrices with small Gaussian noise (mean 0, std 0.01).
		self.Wk.data.normal_(mean = 0, std = 0.01) #Small random init helps training converge and avoids symmetry.

	def forward(self, node_reprs, adjacency_matrices, return_list=False):
		if return_list==True: agg_node_reprs = []
		else: agg_node_reprs = torch.FloatTensor(node_reprs.shape).fill_(0)
		for i, (node_repr, adj_matrix) in enumerate(zip(node_reprs, adjacency_matrices)):
			if len(node_repr)==0 and return_list==True:
				agg_node_reprs.append(torch.FloatTensor(self.embedding_dim).fill_(0))
				continue
			Q = torch.mm(node_repr, self.Wq)
			K = torch.mm(node_repr, self.Wk)
			QK = torch.mm(Q, K.transpose(0, 1)) * adj_matrix
			QK = softmax(QK, dim=1) 
			node_repr_ = torch.mm(QK, node_repr)
			if return_list==True: agg_node_reprs.append(node_repr_)
			else: agg_node_reprs[i] = node_repr_
		return agg_node_reprs

class SelfAttentionLayer(nn.Module):
	def __init__(self, embedding_dim=300, output_size=128):
		super(SelfAttentionLayer, self).__init__()
		self.Wv = Parameter(torch.FloatTensor(embedding_dim, output_size))
		self.Wk = Parameter(torch.FloatTensor(embedding_dim, 1))
		self.Wv.data.normal_(mean = 0, std = 0.01)
		self.Wk.data.normal_(mean = 0, std = 0.01)
		self.activate_func = nn.ELU()

	def forward(self, node_repr):
		attenion_score = torch.mm(node_repr, self.Wk).transpose(0, 1)
		attenion_score = softmax(attenion_score, dim=1)
		agg_repr_project = torch.mm(node_repr, self.Wv)
		att_node_repr_ = torch.mm(attenion_score, agg_repr_project)
		att_node_repr_ = self.activate_func(att_node_repr_)
		return att_node_repr_, attenion_score

