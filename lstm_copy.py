import torch
import torch.cuda
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# system utils
import os
import sys
from time import time
# data utils
import numpy as np
import h5py
import pickle
import json
# self-defined modules
import preprocess

# system
USE_GPU = torch.cuda.is_available()
# data
FEAT_DIM = 2048
WV_DIM = 300
dtype = torch.FloatTensor
# training
n_epoch = 150
print_every_train = 50
print_every_val = 500
batch_size = 64
lr = 5e-6
wd = 5e-3
# Exec related
LOAD_TRAIN = True
LOAD_VAL = True
LOAD_TEST = True
PERFORM_TRAIN = True


class VQADataset(Dataset):
	def __init__(self, img_features, qa_map):
		self.img_features = img_features
		self.qa_map = qa_map

	def __getitem__(self, index):
		qa_id = self.qa_map[index]['qa_id']
		img_id = str(self.qa_map[index]['img_id'])
		q_embed = self.qa_map[index]['q_embed']
		a_embeds = self.qa_map[index]['a_embeds']

		"""
		Shapes of return values:
		* q_embed: (n_q, 300)
		* a_embeds: list of 4 items, each of size (n_a, 300); the 1st item is GT
		* img_feats: (2048,) i.e. 1D vector
		"""
		# return np.stack([q_embed for i in range(4)]), np.stack([a_embed for a_embed in a_embeds]), np.tile(np.asarray(self.img_features[img_id]), [4,1]), np.asarray([1,0,0,0]).reshape(4,1)
		qa_embeds = []
		for a_embed in a_embeds:
			qa_embeds.append(np.vstack((q_embed, a_embed)))
		return qa_embeds, np.tile(np.asarray(self.img_features[img_id]), [4,1]), np.asarray([1,0,0,0]).reshape(4,1)
		# return np.tile(q_embed.mean(0), [4,1]), np.stack([a_embed.mean(0) for a_embed in a_embeds]), np.tile(np.asarray(self.img_features[img_id]), [4,1]), np.asarray([1,0,0,0]).reshape(4,1)

	def __len__(self):
		return len(self.qa_map)

def pad_collate_fn(batch):
	# max_len = max(batch, key=lambda tp: max(tp[0], key=lambda li: ))
	# max_len = max([max(tp[0], key=lambda a: a.shape[0]) for tp in batch])
	qa_embeds = [qa_embed for tp in batch for qa_embed in tp[0]]
	
	lens = [i.shape[0] for i in qa_embeds]

	indices = np.argsort(lens)[::-1]
	lens = lens[indices]
	max_len = lens[0]
	batch_size = len(batch)

	np.zeros((batch_size, lens))

	# for qa_embeds, img_feats, labels in batch:

	return batch

class LSTMModel(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, mlp_dims=[], n_layers=1, dropout=0):
		super(LSTMModel, self).__init__()
		#self.drop = nn.Dropout(dropout)

		self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers)
		self.hidden = self.init_hidden()

		layers = []
		dims = [hidden_dim] + mlp_dims + [out_dim]
		for i in range(len(dims)-1):
			layers += nn.Linear(dims[i], dims[i+1]),
			if i != len(dims)-2:
				# i.e. no ReLU at the last layer
				layers += nn.ReLU(),
		self.hidden2out = nn.Sequential(*layers)

	def init_hidden(self):
		#TODO batch size
		return Variable(torch.rand(self.n_layers, self.batch_size, self.hidden_dim))

	def forward(self, inputs):
		out, self.hidden = self.lstm(inputs, self.hidden)
		out = self.hidden2out(out)
		return out


def train(model, optim, loader):
	model.train()

	top1_cnt = 0
	total_cnt = 0
	t_start = time()
	for i,batch in enumerate(loader):
		if i and i%print_every_train==0:
			avg_loss = sum(stats['train_loss'][-print_every_train:])/print_every_train
			avg_time = (time()-t_start)/print_every_train
			cur_acc = top1_cnt/total_cnt if total_cnt else -1
			print('{:d}-{:d}: avg_loss={:f} / avg_time={:f}s / cur_acc={:f}'.format(i-print_every_train, i, avg_loss, avg_time, cur_acc))
			sys.stdout.flush()
			t_start = time()

		# unroll a batch
		q_embed, a_embeds, img_feats, gt = batch
		# Variable for autograd
		q_embed_var = Variable(q_embed.view(q_embed.size(0)*q_embed.size(1), q_embed.size(2), q_embed.size(3))).type(dtype) # Variable(q_embed.mean(1)).type(dtype)
		img_feats_var = Variable(img_feats.view(img_feats.size(0)*img_feats.size(1), img_feats.size(2), img_feats.size(3))).type(dtype)
		a_embeds_var = Variable(a_embeds.view(a_embeds.size(0)*a_embeds.size(1), a_embeds.size(2))).type(dtype) # [Variable(a_embed).type(dtype) for a_embed in a_embeds] # [Variable(a_embed.mean(1)).type(dtype) for a_embed in a_embeds]
		gt_var = Variable(gt.view(gt.size(0)*gt.size(1), gt.size(2))).type(dtype)
		# Concatenate features: question + img + answers
		concated_qa = torch.cat([q_embed_var, a_embeds_var], dim=2)
		if USE_GPU:
			concated = concated.cuda()
			gt_var = gt_var.cuda()

		#TODO padding
		
		# forward
		out = model(concated_qa)
		_, idx = out.view(q_embed.size(0), q_embed.size(1)).sort(dim=1, descending=True)
		loss = loss_fn(out, gt_var)
		# update stats
		top1_cnt += sum(idx[:, 0]==0).data[0]
		total_cnt += idx.size(0)
		stats['train_loss'].append(loss.data[0])
		# backward
		optim.zero_grad()
		loss.backward()
		optim.step()
	acc = top1_cnt/total_cnt if total_cnt else -1
	stats['train_acc'].append(acc)
	print("train top@1 accuracy:", acc)


def eval(model, loader, update_stats=False, save=''):
	model.eval()

	top1_cnt = 0
	total_cnt = 0
	top1_result = []
	t_start = time()
	for i,batch in enumerate(loader):
		if i and i%print_every_val==0:
			avg_time = (time()-t_start)/print_every_val
			cur_acc = top1_cnt/total_cnt if total_cnt else -1
			print('{:d}-{:d}: avg_time={:f}s / cur_acc={:f}'.format(i-print_every_val, i, avg_time, cur_acc))
			sys.stdout.flush()
			t_start = time()

		# unroll a batch
		q_embed, a_embeds, img_feats, gt = batch
		# Variable for autograd
		q_embed_var = Variable(q_embed.view(q_embed.size(0)*q_embed.size(1), q_embed.size(2))).type(dtype) # Variable(q_embed.mean(1)).type(dtype)
		img_feats_var = Variable(img_feats.view(img_feats.size(0)*img_feats.size(1), img_feats.size(2))).type(dtype)
		a_embeds_var = Variable(a_embeds.view(a_embeds.size(0)*a_embeds.size(1), a_embeds.size(2))).type(dtype) # [Variable(a_embed).type(dtype) for a_embed in a_embeds] # [Variable(a_embed.mean(1)).type(dtype) for a_embed in a_embeds]
		gt_var = Variable(gt.view(gt.size(0)*gt.size(1), gt.size(2))).type(dtype)
		# Concatenate features: question + img + answers
		concated = torch.cat([q_embed_var, img_feats_var, a_embeds_var], dim=1)
		if USE_GPU:
			concated = concated.cuda()
			gt_var = gt_var.cuda()

		# forward
		out = model(concated)
		_, idx = out.view(q_embed.size(0), q_embed.size(1)).sort(dim=1, descending=True)
		# update stats
		top1_cnt += sum(idx[:, 0]==0).data[0]
		total_cnt += idx.size(0)
		top1_result.append((idx[:, 0] == 0).cpu().data.numpy())
	acc = top1_cnt/total_cnt if total_cnt else -1
	if update_stats:
		stats['val_acc'].append(acc)
	print("val top@1 accuracy:", acc)
	if save:
		with open(save, 'wb') as fout:
			np.save(fout, np.concatenate(top1_result))
	return acc


if __name__ == '__main__':
	# input files
	json_filename_format = './data/visual7w-telling_{:s}.json'
	img_feats_fname = './data/resnet101_avgpool.h5'
	glove_p_filename = './data/word2vec_glove.6B.300d.pkl'
	saved_qa_map_format = './data/qa_cache_{:s}.pkl'
	pretrained_path = './checkpoints/lr0.000005_wd0.005000_bts64.pt'
	use_pretrain = False
	# output files
	out_dir = './checkpoints'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	file_format = os.path.join(out_dir, 'lr{:f}_wd{:f}_bts{:d}'.format(lr, wd, batch_size))
	log_file = file_format + '.json'
	checkpoint = file_format + '.pt'

	# img_feat # TODO: split img features to be efficient
        # NOTE: make a copy of the data in img_feats so that we can release the h5py file object
	with h5py.File(img_feats_fname) as handle: 
		img_feats = {key:np.asarray(handle[key]) for key in handle}
	print('Finish loading img_feats.') # NOTE: indicating the file has been released

	if LOAD_TRAIN:
		# train
		train_qa_map = pickle.load(open(saved_qa_map_format.format('train'), 'rb')) if os.path.exists(saved_qa_map_format.format('train')) \
			else preprocess.process_qas(json_filename_format.format('train'), glove_p_filename, save=saved_qa_map_format.format('train'))
		train_loader = DataLoader(VQADataset(img_feats, train_qa_map), batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
	if LOAD_VAL:
		# val
		val_qa_map = pickle.load(open(saved_qa_map_format.format('val'), 'rb')) if os.path.exists(saved_qa_map_format.format('val')) \
			else preprocess.process_qas(json_filename_format.format('val'), glove_p_filename, save=saved_qa_map_format.format('val'))
		val_loader = DataLoader(VQADataset(img_feats, val_qa_map), batch_size=1, shuffle=False)
	if LOAD_TEST:
		# test
		test_qa_map = pickle.load(open(saved_qa_map_format.format('test'), 'rb')) if os.path.exists(saved_qa_map_format.format('test')) \
			else preprocess.process_qas(json_filename_format.format('test'), glove_p_filename, save=saved_qa_map_format.format('test'))
		test_loader = DataLoader(VQADataset(img_feats, test_qa_map), batch_size=1, shuffle=False)


	model = LSTMModel(in_dim=2*WV_DIM+FEAT_DIM, out_dim=1, hidden_dims=[1024, 512, 512])
	loss_fn = torch.nn.BCEWithLogitsLoss()
	optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
	if use_pretrain and pretrained_path:
		pretrained = torch.load(pretrained_path)
		model.load_state_dict(pretrained['model'])
		optim.load_state_dict(pretrained['optim'])
	if USE_GPU:
		print("Use GPU")
		model = model.cuda()
		loss_fn = loss_fn.cuda()
	else:
		print("Use CPU")

	if PERFORM_TRAIN:
		best_acc = 0
		stats = {'train_loss':[], 'train_acc':[], 'val_acc':[]}
		for e in range(n_epoch):
			print("\n\n==== Epoch {:d} ====".format(e+1))
			train(model, optim, train_loader)
			val_acc = eval(model, val_loader, update_stats=True)
	
			with open(log_file, "w") as handle:
				json.dump(stats, handle)
			
			if val_acc > best_acc:
				best_acc = val_acc
				torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, checkpoint)

	# Evaluate on test set
	print("Evaluating on test set...")
	eval(model, test_loader, save='bow_test_top1.npy')
