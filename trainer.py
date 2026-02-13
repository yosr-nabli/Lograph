import torch
import config
import metrics
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import *

def train_epoch(model, train_loader, optimizer, scheduler, epoch_id):
	#nables “training mode”
	model.train()
	#stores per-batch losses.
	loss_curve = []
	#Iterate over batches from the DataLoader
	for idx, batch_samples in enumerate(tqdm(train_loader)):
		#model.zero_grad() clears gradients from the previous batch
		model.zero_grad()
		#if enabled, keep only normal samples (label 0) in this batch
		if model.drop_positive_samples == True:
			labels = batch_samples[1]
			indices = [i for i,y in enumerate(labels) if y==0]
			if len(indices)==0: continue
			batch_samples = [smpl[indices] for smpl in batch_samples]
		#produces predicted probabilities for normal/anomaly for each sample in the batch.
		proba = model.forward(batch_samples)
		#Compute loss comparing predictions to true labels
		loss = model.calculate_loss(batch_samples, proba)
		#loss.item converts it to a Python number for logging.
		loss_curve.append(float(loss.item()))
		#Backpropagation:
		#compute gradients of loss w.r.t. all trainable parameters.
		loss.backward()
		#updates parameters using the gradients.
		optimizer.step()
		#clears gradients again :redunadant but harmless
		optimizer.zero_grad()
	#After finishing all batches in the epoch, update the learning rate schedule
	scheduler.step()
	#Log average training loss for the epoch.
	avg_train_loss = np.mean(loss_curve)
	printf("Epoch: %d, train loss: %.6lf"%(epoch_id, avg_train_loss))
	#Return list of batch losses.
	return loss_curve

def train(model, train_loader, dev_loader):
	best_score = [0.0] #stores the best dev F1-score seen so far. They use a list so they can mutate it inside the nested function.
	loss_curve = [] #track average training loss per epoch.
	loss_curve_dev = [] #track dev loss per epoch.
	early_stop_step_counter = [0] #how many epochs in a row we did not improve.

	set_clock("training") #start a timer (just for printing elapsed time).
	model.to(config.device) #move model parameters to CPU/GPU
	#model.parameters() = all trainable weights (GraphAttention matrices, LSTM weights, classifier weights, etc.).
	optimizer = Adam(model.parameters(), lr=config.lr, betas=config.betas) #the algorithm that updates weights to reduce loss.
	scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)#changes the learning rate over time. StepLR: every lr_step epochs multiply lr by lr_gamma.
	printf("Start to train model...")
	training_params = {
		"lr":config.lr,
		"min_num_epoch":(config.min_num_epoch,config.num_epoch,config.num_early_stop_steps),
		"hidden_size":config.hidden_size,
		"batch_size":config.batch_size,
		"drop_out":config.drop_out,
		"train_prop":config.train_prop,
		"improve_threshold":config.improve_threshold,
	}
	printf("Params:", training_params)
	model.record_training_params(training_params)
	model.auto_save() # saves an initial checkpoint to disk

	def _control_training_progress(model, cur_score): #early-stopping controller #current epoch’s dev F1.
		improve_score = cur_score - best_score[-1] #how much better we got.
		if improve_score > config.improve_threshold:
			best_score.append(cur_score) #update best score
			model.auto_save() #save model checkpoint
			early_stop_step_counter[0] = 0 #reset counter
		else:
			best_score.append(best_score[-1]) # keep best score unchanged
			early_stop_step_counter[0] += 1 #increment no-improvement counter
		#If after a few epochs the best F1 is still terrible (<0.01), stop early.
		
		if epoch >= config.num_early_stop_steps and best_score[-1] < 0.01:
			printf("Training failed! val f1-score: %.6lf"%(best_score[-1]))
			return True #true if training should end
		#and already trained at least min_num_epoch
		if (early_stop_step_counter[0] >= config.num_early_stop_steps and 
					epoch > config.min_num_epoch) or epoch == config.num_epoch:
			printf("Best val f1-score: %.6lf"%(best_score[-1]))
			return True
		return False

	for epoch in range(1, config.num_epoch + 1):
		#loss_curve for that epoch
		loss_ = train_epoch(model, train_loader, optimizer, scheduler, epoch)
		loss_curve.append(np.mean(loss_))
		#don’t store gradients (faster, less memory) because we’re evaluating only.
		with torch.no_grad():
			#Compute metrics on:
			#training set (for monitoring),
			result_train = test(model, train_loader, calc_loss=True, show_detail=False)
			#dev set (for early stopping / best model selection).
			result = test(model, dev_loader, calc_loss=True, show_detail=False)
			#Print train/dev F1 and dev loss.
			printf("Epoch: %d, train f1-score: %.6lf, dev f1-score: %.6lf, dev loss: %.6lf"%\
							(epoch, result_train['f1'], result['f1'], result['loss']))
			#Decide if we should stop and/or save based on dev F1.
			end_flag = _control_training_progress(model, result['f1'])
			loss_curve_dev.append(result['loss'])
			#stop condition is met, break the epoch loop.
			if end_flag==True:
				break
    #print time
	printf("Training complete!")
	printf("Elapsed Time: %.3lfs"%(get_clock("training")))
	#reloads the checkpoint saved by model.auto_save() (the best one according to dev F1).
	model.reload()
	#Evaluate on dev one final time (with detailed logging).
	result = test(model, dev_loader, calc_loss=True, alias="on Dev")
	#Store the final best dev F1 inside the model.
	model.record_evaluation_score(result['f1'])
	#Save the final checkpoint.
	model.auto_save()
	return result

def group_by_entities(y_true, y_proba, z_group, aggregator="max"):
	if aggregator=="avg": agg_func = np.mean
	elif aggregator=="max": agg_func = np.max
	else: agg_func = np.mean
	group_info = {}
	for yt,yp,g in zip(y_true, y_proba, z_group):
		if g not in group_info:
			group_info[g] = [[], []]
		group_info[g][0].append(yt)
		group_info[g][1].append(yp)
	y_true, y_proba = [], []
	for g, (yts, yps) in group_info.items():
		y_true.append(int(round(agg_func(yts), 0)))
		y_proba.append(round(agg_func(yps), 8))
	return y_true, y_proba

def test(model, data_loader, calc_loss=False, show_detail=True, alias=""):
	#puts layers like dropout, batchnorm into evaluation mode (no randomness, deterministic behavior).
	model.eval()
	#y_true: list of ground-truth labels (0/1).
	#y_proba: list of predicted probabilities for the positive class.
	#z_group: list of group IDs (e.g., node IDs) per sample.
	y_true, y_proba, z_group = [], [], []
	#accumulate loss if calc_loss=True.
	sum_dev_losses = 0.0
	#wraps loader in a progress bar if show_detail.
	target_data_loader = tqdm(data_loader) if show_detail==True else data_loader
	for idx, batch_samples in enumerate(target_data_loader):
		output = model.forward(batch_samples)
		#If we want loss: for dev
		if calc_loss==True and model.loss_func is not None:
			#Compute cross-entropy loss for this batch.
			loss = model.calculate_loss(batch_samples, output)
			#Add to sum_dev_losses.
			sum_dev_losses += float(loss.item())
		if model.task=="binary_class":
			#output[:, 1] = probability of class 1 (anomaly).
			if len(output.shape)==2: output = output[:,1]
			#convert to plain Python list of floats (no gradient tracking).
			proba = output.detach().numpy().tolist()
		else:
			raise NotImplementedError(model.task)
		#labels tensor from DataLoader.
		label = batch_samples[1]
		#Append ground-truth labels to y_true.
		y_true.extend(label.detach().numpy().tolist())
		#Append predicted probabilities to y_proba.
		y_proba.extend(proba)
		#batch_samples[2] = group IDs (integers for each sample; 0,1,2... mapped from group strings earlier) → extend z_group.
		z_group.extend(batch_samples[2].numpy().tolist())
	result = {}
	if calc_loss==True:
		#If we computed loss, average it over number of samples and store in result['loss'].
		result['loss'] = float(sum_dev_losses) / len(y_true)
	if config.group_by_entities==True and z_group[0]!='':
		#Aggregate predictions at group level instead of sample level.
		#group_by_entities will:
			#For each group ID:
			#Collect all y_true and y_proba for samples in that group.
			#Aggregate them (max or mean) to a single label/probability per group.
			#This is useful for entity-level detection (e.g. per node).
		y_true, y_proba = group_by_entities(y_true, y_proba, z_group)
		#metrics.evaluate computes F1 (and also precision/recall/AUC) using y_true:0/1, y_proba:[0..1] threshold: 0.5 by default, logs the metrics via printf if how_detail
	result['f1'] = metrics.evaluate(y_true, y_proba, show_detail=show_detail, alias=alias)
	return result
