import numpy as np
import config
import os
from utils import load_object

def load_word_embedding():
	print("Start to load word embeddings...")
	word2vec = {}
	with open(embed_file, "r", encoding="utf-8") as f:
		for line in f.readlines():
			vec = line.split()
			word2vec[vec[0]] = np.array(list(map(float,vec[1:])))
	print("Loaded %d word embeddings."%(len(word2vec)))
	return word2vec

def load_word_embedding_model(model_name="bert"):
	word2vec_file = "%s_embedding.obj"%(model_name)
	word2vec = load_object(os.path.join(config.word_embed_path, word2vec_file))
	return word2vec

class Vocab():
	PAD, START, END, UNK = 0, 1, 2, 3
	def __init__(self):
		# Labels for classification (e.g., Normal and Anomalous logs)
		self._id2label = ["Normal", "Anomalous"]
		# Mapping from label to its ID
		self._label2id = {k: v for k, v in enumerate(self._id2label)}
		# List of words in the vocabulary (indexable by ID)
		self._id2word = []
		# Mapping from word to its ID
		self._word2id = {}
		# Dimensionality of word embeddings
		self._embed_dim = -1
		# Matrix of word embeddings (rows correspond to words in _id2word)
		self.embeddings = None

	def feed(self, word2vec, force_replace=False):
		if force_replace==True:
			self._id2word = []
			self._word2id = {}
			self._embed_dim = -1
		for special_word in ['<pad>', '<bos>', '<eos>', '<oov>']:
			if special_word not in self._word2id:
				self._word2id[special_word] = len(self._word2id)
				self._id2word.append(special_word)
		for word, embed in word2vec.items():
			if self._embed_dim == -1:
				self._embed_dim = embed.shape[0]
				config.embedding_dim = self._embed_dim
			if word not in self._word2id:
				self._word2id[word] = len(self._word2id)
				self._id2word.append(word)
		word_num = len(self._id2word)
		print('Number of words: %d, Dimension of embeddings: %d.'%(len(self._id2word), self._embed_dim))
		self.embeddings = np.zeros([word_num, self._embed_dim])
		for word, embed in word2vec.items():
			index = self._word2id.get(word)
			vector = np.array(embed, dtype=float)
			self.embeddings[index] = vector
			self.embeddings[self.UNK] += vector
		self.embeddings[self.UNK] /= max(word_num, 1)
		if config.use_normalization==True:
			avg_dim_std = 0
			for k in range(self._embed_dim):
				dim_std = np.std(self.embeddings[:, k]) #take column k each time and do std
				avg_dim_std += dim_std
			avg_dim_std = max(avg_dim_std/self._embed_dim, 1e-8)
			self.embeddings /= avg_dim_std
		return self

	def __contains__(self, x):
		return x in self._word2id

	def word2id(self, xs):
		if isinstance(xs, list):
			return [self._word2id.get(x, self.UNK) for x in xs]
		return self._word2id.get(xs, self.UNK)

	def id2word(self, xs):
		if isinstance(xs, list):
			return [self._id2word[x] for x in xs]
		return self._id2word[xs]

	def label2id(self, xs):
		if isinstance(xs, list):
			return [self._label2id.get(x) for x in xs]
		return self._label2id.get(xs)

	def id2label(self, xs):
		if isinstance(xs, list):
			return [self._id2label[x] for x in xs]
		return self._id2label[xs]

	@property
	def vocab_size(self):
		return len(self._id2word)

	@property
	def label_size(self):
		return len(self._id2label)

	@property
	def word_dim(self):
		return self._embed_dim

