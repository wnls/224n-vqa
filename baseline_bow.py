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


USE_GPU = torch.cuda.is_available()
FEAT_DIM = 2048
WV_DIM = 300
print_every = 50
dtype = torch.FloatTensor
# Hyperparameters
batch_size = 32
lr = 5e-4
wd = 5e-4


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
		return np.tile(q_embed.mean(0), [4,1]), np.stack([a_embed.mean(0) for a_embed in a_embeds]), np.tile(np.asarray(self.img_features[img_id]), [4,1]), np.asarray([1,0,0,0]).reshape(4,1)

	def __len__(self):
		return len(self.qa_map)


class BOWModel(nn.Module):
	def __init__(self, in_dim, out_dim=1, hidden_dims=[]):
		super(BOWModel, self).__init__()
		layers = []
		dims = [in_dim] + hidden_dims + [out_dim]
		for i in range(len(dims)-1):
			layers += nn.Linear(dims[i], dims[i+1]),
			if i != len(dims)-2:
				# i.e. no ReLU at the last layer
				layers += nn.ReLU(),
		self.net = nn.Sequential(*layers)

	def forward(self, input_var):
		return self.net(input_var)


def train(model, optim, loader):
	model.train()

	top1_cnt = 0
	total_cnt = 0
	stats = []
	t_start = time()
	for i,batch in enumerate(loader):
		if i and i%print_every==0:
			avg_loss = sum(stats[-print_every:])/print_every
			avg_time = (time()-t_start)/print_every
			cur_acc = top1_cnt/total_cnt if total_cnt else -1
			print('{:d}-{:d}: avg_loss={:f} / avg_time={:f}s / cur_acc={:f}'.format(i-print_every, i, avg_loss, avg_time, cur_acc))
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

		# forward
		out = model(concated)
		_, idx = out.view(q_embed.size(0), q_embed.size(1)).sort(dim=1, descending=True)
		loss = loss_fn(out, gt_var)
		# update stats
		top1_cnt += sum(idx[:, 0]==0).data[0]
		total_cnt += idx.size(0)
		stats.append(loss.data[0])
		# backward
		optim.zero_grad()
		loss.backward()
		optim.step()
	print("train top@1 accuracy:", top1_cnt/total_cnt if total_cnt else "N/A")
	return stats



def eval(model, loader):
	model.eval()

	top1_cnt = 0
	total_cnt = 0
	for i,batch in enumerate(loader):
		if i and i%print_every==0:
			print('{:d}-{:d}: avg_time={:f}s / cur_acc={:f}'.format(i-print_every, i,
				(time()-t_start)/print_every, top1_cnt/total_cnt if total_cnt else -1))
			sys.stdout.flush()
			t_start = time()

		q_embed, a_embeds, img_feats, gt = batch
		# convert to Variables
		q_embed_var = Variable(q_embed.mean(1)).type(dtype)
		img_feats_var = Variable(img_feats).type(dtype)
		a_embeds_var = [Variable(a_embed.mean(1)).type(dtype) for a_embed in a_embeds]
		gt_var = Variable(torch.transpose(gt, 0, 1)).type(dtype)

		concated = torch.stack([torch.cat([q_embed_var, img_feats_var, a_embed_var], dim=1) for a_embed_var in a_embeds_var])
		concated = concated.view(concated.size(0), concated.size(-1))

		out = model(concated)
		_, idx = out.sort()
		top1_cnt += sum(idx[:,0] == 0)
		total_cnt += idx
	acc = top1_cnt/total_cnt if total_cnt else -1
	print("train top@1 accuracy:", -1)
	return acc


if __name__ == '__main__':
	# input files
	json_filename_format = './data/visual7w-telling_{:s}.json'
	img_feats_fname = './data/resnet101_avgpool.h5'
	glove_p_filename = './data/word2vec_glove.6B.300d.pkl'
	saved_qa_map_format = './data/qa_cache_{:s}.pkl'
	# output files
	file_format = './checkpoints/lr{:f}_wd{:f}_bts{:d}'.format(lr, wd, batch_size)
	log_file = file_format + '.json'
	checkpoint = file_format + '.pt'

	# img_feat # TODO: split img features to be efficient
	img_feats = h5py.File(img_feats_fname)

	# train
	train_qa_map = pickle.load(open(saved_qa_map_format.format('train'), 'rb')) if os.path.exists(saved_qa_map_format.format('train')) \
		else preprocess.process_qas(json_filename_format.format('train'), glove_p_filename, save=saved_qa_map_format.format('train'))
	train_loader = DataLoader(VQADataset(img_feats, train_qa_map), batch_size=batch_size, shuffle=True)
	# val
	val_qa_map = pickle.load(open(saved_qa_map_format.format('val'), 'rb')) if os.path.exists(saved_qa_map_format.format('val')) \
		else preprocess.process_qas(json_filename_format.format('val'), glove_p_filename, save=saved_qa_map_format.format('val'))
	val_loader = DataLoader(VQADataset(img_feats, val_qa_map), batch_size=1, shuffle=False)
	# test
	test_qa_map = pickle.load(open(saved_qa_map_format.format('test'), 'rb')) if os.path.exists(saved_qa_map_format.format('test')) \
		else preprocess.process_qas(json_filename_format.format('test'), glove_p_filename, save=saved_qa_map_format.format('test'))
	test_loader = DataLoader(VQADataset(img_feats, test_qa_map), batch_size=1, shuffle=False)


	model = BOWModel(in_dim=2*WV_DIM+FEAT_DIM, out_dim=1, hidden_dims=[1024, 512, 512])
	loss_fn = torch.nn.BCEWithLogitsLoss()
	if USE_GPU:
		model = model.cuda()
		loss_fn = loss_fn.cuda()
	optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

	n_epoch = 100
	best_acc = 0
	stats = {'train_loss':[]}
	for e in range(n_epoch):
		stats['train_loss'] += train(model, optim, train_loader)
		val_acc = eval(model, val_loader)

		with open(log_file, "w") as handle:
			json.dump(stats, handle)
		
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, checkpoint)
