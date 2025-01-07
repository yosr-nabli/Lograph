import hashlib
import config
import logging
import os

def hash_func(x):
	return hashlib.md5(x.encode()).hexdigest()

def init_logger(log_path):
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		## To file
		file_handler = logging.FileHandler(log_path)
		file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
		logger.addHandler(file_handler)
		## To console
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter('%(message)s'))
		logger.addHandler(stream_handler)
	return

def printf(*args):
	message = " ".join(list(map(str,args)))
	logging.info(message)
	return

def check_folder_exists(folder_path):
	if os.path.exists(folder_path)==False:
		os.mkdir(folder_path)
	return

import time
def str2ts(time_str,formats="%Y-%m-%d %H:%M:%S"):
    return int(time.mktime(time.strptime(time_str,formats)))

LAST_TIMESTAMP = {}
def set_clock(key=""):
	LAST_TIMESTAMP[key] = time.time()

def get_clock(key=""):
	return time.time() - LAST_TIMESTAMP.get(key, 0)

import numpy as np
np.set_printoptions(threshold=np.inf)

def save_data(data,file_name):
	data = np.array(data, dtype=object)
	np.savez_compressed(file_name, data=data)

def load_data(file_name):
	data = np.load(file_name, allow_pickle=True)["data"].tolist()
	return data

import pickle
def save_object(obj,filename):
	with open(filename,"wb+") as f:
		pickle.dump(obj,f)
	return

def load_object(filename):
	with open(filename,"rb") as f:
		obj = pickle.load(f)
	return obj

import csv
def read_csv(filename,encoding=None):
	f = open(filename,"r",encoding=encoding)
	data = list(csv.reader(f))
	f.close()
	return data

import json
def read_json(filename,encoding=None):
	f = open(filename,"r",encoding=encoding)
	data = json.load(f)
	f.close()
	return data

def flatten(x):
	y = []
	for item in x:
		y += item
	return y

def multiply(x):
	y = 1
	for item in x:
		y *= item
	return y

if __name__=="__main__":
	print(hash_func("HelloWorld"))
