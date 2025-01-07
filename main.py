import config
import os
import numpy as np
import torch
from utils import *

def init_experiment():
	check_folder_exists(config.output_path)
	check_folder_exists(config.torch_model_path)
	check_folder_exists(config.data_package_path)
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
		printf("Loaded %d log messages." % (len(log_data)))
		samples = convert_into_samples(log_data, group_type, window_size, log_name=log_name, 
										max_time_interval=int(config.max_time_interval*HOUR))
		printf("Got %d log samples." % (len(samples)))
		tmpl_list, template_map = log_tokenization(log_data, vocab)
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
		log_entity_graph.build(log_data)
		log_entity_graph.save()
	else:
		log_entity_graph.load()
	log_entity_graph.apply_template_mapping(tmpl_list)
	print("Build Log Entity Graph Complete! Elapsed Time: %.3lfs"%(get_clock("build_graph")))
	return log_entity_graph

def prepare_dataset(log_name, group_type):
	if group_type is None or group_type=="None":
		config.use_group_evaluation = False
	word2vec = load_word_embedding_model(config.word2vec_model)
	vocab = Vocab().feed(word2vec)
	samples, tmpl_list, template_map = prepare_log_dataset(log_name, vocab, group_type, config.window_size)
	log_entity_graph = prepare_entity_dataset(log_name, tmpl_list) if config.use_log_entity_graph==True else None
	train_index, dev_index, test_index = train_test_split_grouped(samples)
	train_index = simple_balance_sampling(samples, train_index)
	train_loader, dev_loader = convert_to_training_data_loader(samples, train_index, dev_index, tmpl_list, template_map)
	test_loader = convert_to_testing_data_loader(samples, test_index, tmpl_list, template_map)
	embed_layer = WordAggregateTfIdfLayer(vocab).feed(train_loader) if config.use_tf_idf_aggregation==True else WordAggregateLayer(vocab)
	template_cache = LogTemplateReprCache(vocab, embed_layer).feed(template_map)
	return train_loader, dev_loader, test_loader, vocab, embed_layer, template_cache, log_entity_graph

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
	run_experiment("bgl", "node_id")
