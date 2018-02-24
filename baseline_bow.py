import torch
import torch.cuda
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# system utils
import os
# data utils
import h5py
import pickle
import json
# self-defined modules
import preprocess


USE_GPU = torch.cuda.is_available()
FEAT_DIM = 2048
WV_DIM = 300
# Hyperparameters
lr = 5e-4
wd = 5e-4


class VQADataset(Dataset):
	def __init__(self, img_features, qa_map):
		self.img_features = img_features
		self.qa_map = qa_map

	def __getitem__(self, index):
		qa_id = self.qa_map[index]['qa_id']
		img_id = self.qa_map[index]['img_id']
		q_embed = self.qa_map[index]['q_embed']
		a_embeds = self.qa_map[index]['a_embeds']
		return q_embed, a_embeds, self.img_features[img_id], [1,0,0,0]

	def __len__(self):
		return len(self.qa_map)


class BOWModel(nn.Module):
	def __init__(self, in_dim, out_dim=1, hidden_dims=[]):
		super(BOWModel, self).__init__()
		layers = []
		dims = [in_dim] + hidden_dims + [out_dim]
		for i in range(len(dims)-1):
			layers += nn.Linear(dims[i], dims[i+1]),
			layers += nn.ReLU(),
		self.net = nn.Sequential(*layers)


def train(model, optim, loader):
	model.train()

	top1_cnt = 0
	total_cnt = 0
	for i,batch in enumerate(loader):
		# q_embed, a_embeds, self.img_features[img_id], [1,0,0,0]
		q_embed, a_embeds, img_feats, gt = batch
		concated = 2 # TODO

		# forward
		optim.zero_grad()
		out = model(concated)
		_, idx = out.sort()
		top1_cnt += sum(idx[:,0] == 0)
		total_cnt += idx

		loss = loss_fn(out, gt)
		# backward
		loss.backward()
		optim.step()

def eval(model, loader):
	model.eval()

	top1_cnt = 0
	for i,batch in enumerate(loader):
		q_embed, a_embeds, img_feats, gt = batch
		concated = 1 # TODO
		out = model(concated)

if __name__ == '__main__':
	json_filename_format = './data/visual7w-telling_{:s}.json'
	img_feats_fname = './data/resnet101_avgpool.h5'
	glove_p_filename = './data/word2vec_glove.6B.300d.pkl'
	saved_qa_map_format = './data/qa_cache_{:s}.pkl'

	# img_feat # TODO: split img features to be efficient
	img_feats = h5py.File(img_feats_fname)

	# train
	train_qa_map = pickle.load(open(saved_qa_map_format.format('train'), 'rb')) if os.path.exists(saved_qa_map_format.format('train')) \
		else preprocess.process_qas(json_filename_format.format('train'), glove_p_filename, save=saved_qa_map_format.format('train'))
	train_loader = DataLoader(VQADataset(img_feats, train_qa_map), batch_size=1, shuffle=True)
	# val
	val_qa_map = pickle.load(open(saved_qa_map_format.format('val'), 'rb')) if os.path.exists(saved_qa_map_format.format('val')) \
		else preprocess.process_qas(json_filename_format.format('val'), glove_p_filename, save=saved_qa_map_format.format('val'))
	val_loader = DataLoader(VQADataset(img_feats, val_qa_map), batch_size=1, shuffle=False)
	# test
	test_qa_map = pickle.load(open(saved_qa_map_format.format('test'), 'rb')) if os.path.exists(saved_qa_map_format.format('test')) \
		else preprocess.process_qas(json_filename_format.format('test'), glove_p_filename, save=saved_qa_map_format.format('test'))
	test_loader = DataLoader(VQADataset(img_feats, test_qa_map), batch_size=1, shuffle=False)


	model = BOWModel(in_dim=2*WV_DIM+FEAT_DIM, out_dim=1, hidden_dims=[1024, 512, 512])
	model.train()
	loss_fn = torch.nn.BCEWithLogitsLoss()
	if USE_GPU:
		model = model.cuda()
		loss_fn = loss_fn.cuda()
	optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

	n_epoch = 100
	for e in range(n_epoch):
		train(model, optim, train_loader)
		eval(model, val_loader)
