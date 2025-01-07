import torch
import config
import metrics
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import *

def train_epoch(model, train_loader, optimizer, scheduler, epoch_id):
	model.train()
	loss_curve = []
	for idx, batch_samples in enumerate(tqdm(train_loader)):
		model.zero_grad()
		if model.drop_positive_samples == True:
			labels = batch_samples[1]
			indices = [i for i,y in enumerate(labels) if y==0]
			if len(indices)==0: continue
			batch_samples = [smpl[indices] for smpl in batch_samples]
		proba = model.forward(batch_samples)
		loss = model.calculate_loss(batch_samples, proba)
		loss_curve.append(float(loss.item()))
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
	scheduler.step()
	avg_train_loss = np.mean(loss_curve)
	printf("Epoch: %d, train loss: %.6lf"%(epoch_id, avg_train_loss))
	return loss_curve

def train(model, train_loader, dev_loader):
	best_score = [0.0]
	loss_curve = []
	loss_curve_dev = []
	early_stop_step_counter = [0]

	set_clock("training")
	model.to(config.device)
	optimizer = Adam(model.parameters(), lr=config.lr, betas=config.betas)
	scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
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
	model.auto_save()

	def _control_training_progress(model, cur_score):
		improve_score = cur_score - best_score[-1]
		if improve_score > config.improve_threshold:
			best_score.append(cur_score)
			model.auto_save()
			early_stop_step_counter[0] = 0
		else:
			best_score.append(best_score[-1])
			early_stop_step_counter[0] += 1
		if epoch >= config.num_early_stop_steps and best_score[-1] < 0.01:
			printf("Training failed! val f1-score: %.6lf"%(best_score[-1]))
			return True
		if (early_stop_step_counter[0] >= config.num_early_stop_steps and 
					epoch > config.min_num_epoch) or epoch == config.num_epoch:
			printf("Best val f1-score: %.6lf"%(best_score[-1]))
			return True
		return False

	for epoch in range(1, config.num_epoch + 1):
		loss_ = train_epoch(model, train_loader, optimizer, scheduler, epoch)
		loss_curve.append(np.mean(loss_))
		with torch.no_grad():
			result_train = test(model, train_loader, calc_loss=True, show_detail=False)
			result = test(model, dev_loader, calc_loss=True, show_detail=False)
			printf("Epoch: %d, train f1-score: %.6lf, dev f1-score: %.6lf, dev loss: %.6lf"%\
							(epoch, result_train['f1'], result['f1'], result['loss']))
			end_flag = _control_training_progress(model, result['f1'])
			loss_curve_dev.append(result['loss'])
			if end_flag==True:
				break

	printf("Training complete!")
	printf("Elapsed Time: %.3lfs"%(get_clock("training")))
	model.reload()
	result = test(model, dev_loader, calc_loss=True, alias="on Dev")
	model.record_evaluation_score(result['f1'])
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
	model.eval()
	y_true, y_proba, z_group = [], [], []
	sum_dev_losses = 0.0
	target_data_loader = tqdm(data_loader) if show_detail==True else data_loader
	for idx, batch_samples in enumerate(target_data_loader):
		output = model.forward(batch_samples)
		if calc_loss==True and model.loss_func is not None:
			loss = model.calculate_loss(batch_samples, output)
			sum_dev_losses += float(loss.item())
		if model.task=="binary_class":
			if len(output.shape)==2: output = output[:,1]
			proba = output.detach().numpy().tolist()
		else:
			raise NotImplementedError(model.task)
		label = batch_samples[1]
		y_true.extend(label.detach().numpy().tolist())
		y_proba.extend(proba)
		z_group.extend(batch_samples[2].numpy().tolist())
	result = {}
	if calc_loss==True:
		result['loss'] = float(sum_dev_losses) / len(y_true)
	if config.group_by_entities==True and z_group[0]!='':
		y_true, y_proba = group_by_entities(y_true, y_proba, z_group)
	result['f1'] = metrics.evaluate(y_true, y_proba, show_detail=show_detail, alias=alias)
	return result
