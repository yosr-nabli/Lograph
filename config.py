# Paths
torch_model_path = "./model/"
data_package_path = "./data/"
word_embed_path = "./data/"
raw_dataset_path = "../LogHub/"
output_path = "./output/"
log_file_name = "lograph.log"
device = None

# Preprocessing
random_seed = 2023
min_word_length = 2
window_size = 32
word2vec_model = "glove"
sort_chronological = True
use_tf_idf_aggregation = True
use_normalization = True
max_time_interval = 0.5

# Training Scheme
train_prop = 0.7
dev_prop = 0.1
test_prop = 1.0 - train_prop - dev_prop
group_by_entities = True
use_log_entity_graph = True
use_meta_path = [True, True]
use_transformer = False
use_lstm_layer = True
use_gat_layer = True

# Network Parameters
batch_size = 4
hidden_size = 128
atten_size = 16
num_heads = 2
drop_out = 0.1
lr = 0.001
betas = (0.9, 0.999)
lr_step = 5
lr_gamma = 0.8
num_epoch = 12
min_num_epoch = 3
num_early_stop_steps = 3
improve_threshold = 1e-3

### GLOBAL VARIABLES ###
embedding_dim = None # 
current_log_name = None # 

