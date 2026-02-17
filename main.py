import config
import os
import numpy as np
import torch
from utils import *

def init_experiment():
	check_folder_exists(config.output_path) #for logs and general output
	check_folder_exists(config.torch_model_path) #for model checkpoints
	check_folder_exists(config.data_package_path) #for cached processed data and graphs
	init_logger(os.path.join(config.output_path, config.log_file_name))
	printf("Initializing...")
	np.random.seed(config.random_seed)

init_experiment()

from model import *
from embedding import *
from data_loader import *
from data_sampling import *
from data_preprocess import *
from nearest_neighbors import *
from trainer import train, test
from graph_construction import LogEntityGraph

def prepare_log_dataset(log_name, vocab, group_type=None, window_size=20, force_replace=False):
	config.current_log_name = log_name
	local_data_file = os.path.join(config.data_package_path, "%s_%s_%d.obj"%(log_name, group_type, window_size))
	if os.path.exists(local_data_file)==False or force_replace==True:
		log_data = load_dataset(log_name)
		with open("dataset_preparation_log.txt", "a") as log_file:
			log_file.write(f"3. Loaded {log_name.upper()} dataset with each row corresponding to [log, label, timestamp, entities].\n")
			log_file.write(f"First 3 rows of the dataset: {log_data[:3]}\n")
		printf("Loaded %d log messages." % (len(log_data)))
		samples = convert_into_samples(log_data, group_type, window_size, log_name=log_name, 
										max_time_interval=int(config.max_time_interval*HOUR))
		with open("dataset_preparation_log.txt", "a") as log_file:
			log_file.write("4.Converted log data into samples.\n")
			log_file.write(f"First 2 samples: {samples[:2]}\n")
		printf("Got %d log samples." % (len(samples)))
		tmpl_list, template_map = log_tokenization(log_data, vocab)
		with open("dataset_preparation_log.txt", "a") as log_file:
			log_file.write("3.Performed log tokenization.\n")
			log_file.write(f"First 2 tmpl_list entries:tmpl_list[i] tells you which template key log i belongs to.: {tmpl_list[:2]}\n")
			log_file.write("First 2 template_map entries:gives you the sequence of word IDs that defines that template:\n")
			for tmpl_key, word_ids in list(template_map.items())[:2]:
				log_file.write(f"Template Key: {tmpl_key}, Word IDs: {word_ids}\n")
		save_object([samples, tmpl_list, template_map], local_data_file)
	else:
		samples, tmpl_list, template_map = load_object(local_data_file)
	return samples, tmpl_list, template_map

def prepare_entity_dataset(log_name, tmpl_list=None, force_replace=False):
	log_entity_graph = LogEntityGraph(log_name)
	set_clock("build_graph")
	local_data_file = log_entity_graph.graph_file_name
	if os.path.exists(local_data_file)==False or force_replace==True:
		log_data = load_dataset(log_name)
		printf("Loaded %d log messages." % (len(log_data)))
		print("Building Log Entity Graph...")
		log_entity_graph.build(log_data, visualize_first_n=5)
		log_entity_graph.save()
	else:
		log_entity_graph.load()
	log_entity_graph.apply_template_mapping(tmpl_list)
	print("Build Log Entity Graph Complete! Elapsed Time: %.3lfs"%(get_clock("build_graph")))
	return log_entity_graph

def prepare_dataset(log_name, group_type):
	if group_type is None or group_type=="None":
		config.use_group_evaluation = False

	# Load pretrained embeddings
	word2vec = load_word_embedding_model(config.word2vec_model)

	# Create vocabulary
	vocab = Vocab().feed(word2vec)

	#prepare log: samples[indices of logs length:32, label for those 32,node_id"user1"],template_map: with templ_key access the words of that template(masked_log), 
	# tmpl_list[i] tells us log i to which template_key it belongs
	samples, tmpl_list, template_map = prepare_log_dataset(log_name, vocab, group_type, config.window_size) 
	log_entity_graph = prepare_entity_dataset(log_name, tmpl_list) if config.use_log_entity_graph==True else None
	train_index, dev_index, test_index = train_test_split_grouped(samples)
	train_index = simple_balance_sampling(samples, train_index)
	train_loader, dev_loader = convert_to_training_data_loader(samples, train_index, dev_index, tmpl_list, template_map)
	test_loader = convert_to_testing_data_loader(samples, test_index, tmpl_list, template_map)
	embed_layer = WordAggregateTfIdfLayer(vocab).feed(train_loader) if config.use_tf_idf_aggregation==True else WordAggregateLayer(vocab)
	template_cache = LogTemplateReprCache(vocab, embed_layer).feed(template_map)
	return train_loader, dev_loader, test_loader, vocab, embed_layer, template_cache, log_entity_graph

def prepare_dataset_cpu(log_name, group_type):
	if group_type is None or group_type=="None":
		config.use_group_evaluation = False

	# Load pretrained embeddings
	word2vec = load_word_embedding_model(config.word2vec_model)
	with open("dataset_preparation_log.txt", "a") as log_file:
		log_file.write("1.Loaded GloVe word embedding.\n")

	# Create vocabulary
	vocab = Vocab().feed(word2vec)
	with open("dataset_preparation_log.txt", "a") as log_file:
		log_file.write("2.Created vocabulary.\n")
		log_file.write(f"entities in _id2label:Labels for classification: {vocab._id2label[:3]}\n")
		log_file.write(f"entities in _label2id:Mapping from label to its ID: {list(vocab._label2id.items())[:3]}\n")
		log_file.write(f"First 8 entities in _id2word:List of words in the vocabulary (indexable by ID): {vocab._id2word[:8]}\n")
		log_file.write(f"First 8 entities in _word2id:Mapping from word to its ID: {list(vocab._word2id.items())[:8]}\n")
		log_file.write(f"Embedding dimension:Dimensionality of word embeddings: {vocab._embed_dim}\n")


	#prepare log: samples[indices of logs length:32, label for those 32,node_id"user1"],template_map: with templ_key access the words of that template(masked_log), 
	# tmpl_list[i] tells us log i to which template_key it belongs
	samples, tmpl_list, template_map = prepare_log_dataset(log_name, vocab, group_type, config.window_size)
	log_entity_graph = prepare_entity_dataset(log_name, tmpl_list) if config.use_log_entity_graph==True else None 
	train_index, dev_index, test_index = train_test_split_grouped(samples, sample_ratio=1.0)
	train_index = simple_balance_sampling(samples, train_index, balance_coef=4)
	train_loader, dev_loader = convert_to_training_data_loader(samples, train_index, dev_index, tmpl_list, template_map)
	test_loader = convert_to_testing_data_loader(samples, test_index, tmpl_list, template_map)
	with open("dataset_preparation_log.txt", "a") as log_file:
		log_file.write("5.Splitted samples into train/dev/test sets.\n")
		log_file.write(f"First 3 indices in train_index: {train_index[:3]}\n")
		log_file.write(f"First 3 indices in dev_index: {dev_index[:3]}\n")
		log_file.write(f"First 3 indices in test_index: {test_index[:3]}\n")
		log_file.write("6.balanced samples with anomalous logs 2* normal logs\n")
		log_file.write("7.Converted into Dataloaders\n")
		# Show an example batch from train_loader
		train_iter = iter(train_loader)
		try:
			words, labels, groups, masks, indices = next(train_iter)
			log_file.write("Example train_loader batch (first sample):\n")
			word_ids = words[0].tolist() # shape: [seq_len, num_word]
			word_strs = [[vocab._id2word[w] if w < len(vocab._id2word) else "OOV" for w in word_list if w != 0] for word_list in word_ids]
			log_file.write(f"  Word IDs: {word_ids}\n")
			log_file.write(f"  Words: {word_strs}\n")
			log_file.write(f"  Label: {labels[0].item()}\n")
			log_file.write(f"  Group: {groups[0].item()}\n")
			log_file.write(f"  Indices: {indices[0].tolist()}\n")
		except Exception as e:
			log_file.write(f"  Could not fetch example train_loader batch: {e}\n")
		# Show an example batch from test_loader
		test_iter = iter(test_loader)
		try:
			words, labels, groups, masks, indices = next(test_iter)
			log_file.write("Example test_loader batch (first sample):\n")
			word_ids = words[0].tolist()
			word_strs = [[vocab._id2word[w] if w < len(vocab._id2word) else "OOV" for w in word_list if w != 0] for word_list in word_ids]
			log_file.write(f"  Word IDs: {word_ids}\n")
			log_file.write(f"  Words: {word_strs}\n")
			log_file.write(f"  Label: {labels[0].item()}\n")
			log_file.write(f"  Group: {groups[0].item()}\n")
			log_file.write(f"  Indices: {indices[0].tolist()}\n")
		except Exception as e:
			log_file.write(f"  Could not fetch example test_loader batch: {e}\n")
	embed_layer = WordAggregateTfIdfLayer(vocab).feed(train_loader) if config.use_tf_idf_aggregation==True else WordAggregateLayer(vocab)
	with open("dataset_preparation_log.txt", "a") as log_file:
		log_file.write("Finished computing TF-IDF embeddings.\n")
		if hasattr(embed_layer, 'idf_counter'):
			idf_items = list(embed_layer.idf_counter.items())[:2]
			for word_id, idf_value in idf_items:
				word = vocab._id2word[word_id] if word_id < len(vocab._id2word) else "OOV"
				log_file.write(f"Word_id : {word_id},Word: {word}, IDF: {idf_value}\n")
			log_file.write(f"OOV IDF Value: {embed_layer.oov_idf_value}\n")
	template_cache = LogTemplateReprCache(vocab, embed_layer).feed(template_map)
	with open("dataset_preparation_log.txt", "a") as log_file:
		log_file.write("Finished caching embedding vectors for templates.\n")
		tmpl_keys = list(template_cache.tmpl_index_map.keys())[:2]
		for tmpl_key in tmpl_keys:
			word_ids = template_map[tmpl_key]
			words = [vocab._id2word[w_id] for w_id in word_ids]
			log_file.write(f"Template ID: {tmpl_key}, Words: {words}\n")
			log_file.write(f"Embedding: {template_cache.get_template_repr(tmpl_key)}\n")

def run_experiment(log_name, group_type, model_name="Lograph", alias=""):
	train_loader, dev_loader, test_loader, vocab, embed_layer, template_cache, log_entity_graph = prepare_dataset_cpu(log_name, group_type)
	alias += "_"+config.word2vec_model
	if model_name=="Lograph":
		model = Lograph(log_entity_graph, embed_layer, template_cache, 
						hidden_size=config.hidden_size, atten_size=config.atten_size,
						use_meta_path=config.use_meta_path, alias=alias)
	else:
		raise NotImplementedError(model_name)
	train(model, train_loader, dev_loader)
	model.reload()
	printf("Testing model...")
	result = test(model, test_loader, alias="on Test")
	print(result)
	return result
        	
def run_experiment(log_name, group_type, model_name="Lograph", alias=""):
	train_loader, dev_loader, test_loader, vocab, embed_layer, template_cache, log_entity_graph = prepare_dataset(log_name, group_type)
	alias += "_"+config.word2vec_model
	if model_name=="Lograph":
		model = Lograph(log_entity_graph, embed_layer, template_cache, 
						hidden_size=config.hidden_size, atten_size=config.atten_size,
						use_meta_path=config.use_meta_path, alias=alias)
	else:
		raise NotImplementedError(model_name)
	train(model, train_loader, dev_loader)
	model.reload()
	printf("Testing model...")
	result = test(model, test_loader, alias="on Test")
	print(result)
	return result


if __name__=="__main__":
	#run_experiment("bgl", "node_id")
	# Run only the prepare_log_dataset function to generate the file
    prepare_dataset_cpu("bgl", "node_id")
    print("prepare_log_dataset completed. File generated.")