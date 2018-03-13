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
import time
from datetime import datetime
# data utils
import numpy as np
import h5py
import pickle
import json
# self-defined modules
import preprocess

t_start_total = time.time()

# system
USE_GPU = torch.cuda.is_available()
# data
FEAT_DIM = 2048
WV_DIM = 300
dtype = torch.FloatTensor
# training
n_epoch = 300
print_every_train = 50
print_every_val = 100
batch_size = 128
lr = 5e-7 #####
wd = 5e-4
bidir = False #####
img2seq = False #####
# Exec related
LOAD_TRAIN = True
LOAD_VAL = True
LOAD_TEST = True
PERFORM_TRAIN = True
use_pretrain = True ###
pretrained_path = './checkpoints/lr5e-06_wd0.0005_bts128_ep300_0311010557_continue5.pt' #####
# output files
out_dir = './checkpoints'
if not os.path.exists(out_dir):
	os.mkdir(out_dir)
now = datetime.now()
file_format = os.path.join(out_dir, '{}{}lr{}_wd{}_bts{:d}_ep{:d}_{}'
                           .format("bi_" if bidir else "", "img_" if img2seq else "",
								   lr, wd, batch_size, n_epoch, time.strftime("%m%d%H%M%S")))
log_file = file_format + '_continue6.json' #####
checkpoint = file_format + '_continue6.pt' #####


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
		qa_encode = []
		for a_embed in a_embeds:
			qa_encode.append(np.vstack((q_embed, a_embed)))
		img_encode = np.tile(np.asarray(self.img_features[img_id]), [4,1])
		sys.stdout.flush()
		return qa_encode, img_encode, np.asarray([1,0,0,0]).reshape(4,1)

	def __len__(self):
		return len(self.qa_map)

def pad_collate_fn(batch):
	"""
	Input: batch: list
	Output:
	"""
	qa_embeds = []
	img_feats = []
	labels = []
	for tp in batch:
		qa_embeds.extend(tp[0])
		img_feats.extend(tp[1])
		labels.extend(tp[2])

	img_feats = torch.FloatTensor(np.stack(img_feats))
	labels = torch.FloatTensor(np.stack(labels))

	lens = torch.IntTensor([i.shape[0] for i in qa_embeds])
	seq_lens, indices = lens.sort(descending=True)
	max_len = seq_lens[0]
	qa_embeds_padded = []

	for i in indices:
		qa_embed = qa_embeds[i]
		padded = np.concatenate([qa_embed, np.zeros([max_len - qa_embed.shape[0], WV_DIM])], axis=0)
		qa_embeds_padded.append(padded)
	qa_embeds_padded = torch.FloatTensor(np.stack(qa_embeds_padded, 0))

	if USE_GPU:
		seq_lens = seq_lens.cuda()
		indices = indices.cuda()
		qa_embeds_padded = qa_embeds_padded.cuda()
		img_feats = img_feats.cuda()
		labels = labels.cuda()

	return seq_lens, indices, qa_embeds_padded, img_feats[indices], labels[indices]

class LSTMModel(nn.Module):
	def __init__(self, visual_dim, lang_dim, hidden_dim, out_dim=1, mlp_dims=[], bidirectional=False, n_layers=1, dropout=0, img2seq=False):
		super(LSTMModel, self).__init__()
		#self.drop = nn.Dropout(dropout)

		self.lstm = nn.LSTM(lang_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
		self.visual_dim = visual_dim
		self.lang_dim = lang_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.bidirectional = bidirectional
		# if using img as the first input of word sequence
		self.img2seq = img2seq
        
        # self.batch_size = 4 * batch_size # TODO: hard coded
		# self.hidden = self.init_hidden()
		if img2seq:
			self.img2worddim = nn.Sequential(nn.Linear(visual_dim, lang_dim), nn.ReLU())

		layers = []
		dims = [hidden_dim+visual_dim] + mlp_dims + [out_dim]
		for i in range(len(dims)-1):
			layers += nn.Linear(dims[i], dims[i+1]),
			if i != len(dims)-2:
				# i.e. no ReLU at the last layer
				layers += nn.ReLU(),
		self.mlp = nn.Sequential(*layers)

	def init_hidden(self, batch_size):
		#TODO batch size
		if USE_GPU:
			return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda(),
					Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda())
		else:
			return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)),
					Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)))

	def forward(self, lang_input, img_input, seq_lens):
		'''
		lang_input: tensor - (batch, seq_len, word_dim)
		img_input: tensor - (batch_size, img_dim)
		seq_lens: tensor - list of seq lengtsh in decreasing size, for pack_padded_sequence
		'''
		if self.img2seq:
			img_token = self.img2worddim(img_input)
			img_token = torch.unsqueeze(img_token, 1)
			lang_input = torch.cat([img_token, lang_input], dim=1)
			seq_lens += 1

		lang_input_packed = nn.utils.rnn.pack_padded_sequence(lang_input, list(seq_lens), batch_first=True)

		# self.hidden = self.init_hidden(lang_input.size(0))
		# hidden size zeros by default
		out, (h_t, c_t) = self.lstm(lang_input_packed) # h_t: (num_layers * num_directions, batch_size, hidden_size)
		# h_t = h_t.squeeze(0) 
		h_t = torch.mean(h_t, 0)
		assert(h_t.size(0) == img_input.size(0)), "size mismatch: h_t: {} / img_input: {}".format(h_t.size(), img_input.size())
		mlp_input = torch.cat([h_t, img_input], 1)
		out = self.mlp(mlp_input) # batch * 1
		return out


def train(model, optim, loader):
	model.train()

	top1_cnt = 0
	total_cnt = 0
	t_start = time.time()
	for i,batch in enumerate(loader):
		if i and i%print_every_train==0:
			avg_loss = sum(stats['train_loss'][-print_every_train:])/print_every_train
			avg_time = (time.time()-t_start)/print_every_train
			cur_acc = top1_cnt/total_cnt if total_cnt else -1
			print('{:d}-{:d}: avg_loss={:f} / avg_time={:f}s / cur_acc={:f}'.format(i-print_every_train, i, avg_loss, avg_time, cur_acc))
			sys.stdout.flush()
			t_start = time.time()

		# unroll a batch
		seq_lens, indices, qa_embeds, img_feats, gt = batch
		this_batch_size = int(seq_lens.size(0)/4)
		# Variable for autograd
		qa_embeds_var = Variable(qa_embeds).type(dtype) # Variable(q_embed.mean(1)).type(dtype)
		img_feats_var = Variable(img_feats).type(dtype)
		gt_var = Variable(gt).type(dtype)
		if USE_GPU:
			qa_embeds_var = qa_embeds_var.cuda()
			img_feats_var = img_feats_var.cuda()
			gt_var = gt_var.cuda()

		# forward
		out = model(qa_embeds_var, img_feats_var, seq_lens)
		loss = loss_fn(out, gt_var)
		# unsort out
		_, unsort_ind = indices.sort(0)
		out = out[unsort_ind]
		# update stats
		_, idx = out.view(this_batch_size, -1).sort(dim=1, descending=True)
		top1_cnt += sum(idx[:, 0] == 0).data[0]
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
	t_start = time.time()
	for i,batch in enumerate(loader):
		if i and i%print_every_val==0:
			avg_time = (time.time()-t_start)/print_every_val
			cur_acc = top1_cnt/total_cnt if total_cnt else -1
			print('{:d}-{:d}: avg_time={:f}s / cur_acc={:f}'.format(i-print_every_val, i, avg_time, cur_acc))
			sys.stdout.flush()
			t_start = time.time()

		# unroll a batch
		seq_lens, indices, qa_embeds, img_feats, gt = batch
		this_batch_size = int(seq_lens.size(0)/4)
		# Variable for autograd
		qa_embeds_var = Variable(qa_embeds).type(dtype)  # Variable(q_embed.mean(1)).type(dtype)
		img_feats_var = Variable(img_feats).type(dtype)
		gt_var = Variable(gt).type(dtype)
		if USE_GPU:
			qa_embeds_var = qa_embeds_var.cuda()
			img_feats_var = img_feats_var.cuda()
			gt_var = gt_var.cuda()

		# forward
		out = model(qa_embeds_var, img_feats_var, seq_lens)
		# unsort out
		_, unsort_ind = indices.sort(0)
		out = out[unsort_ind]
		# update stats
		_, idx = out.view(this_batch_size, -1).sort(dim=1, descending=True)
		top1_cnt += sum(idx[:, 0] == 0).data[0]
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
		val_loader = DataLoader(VQADataset(img_feats, val_qa_map), batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
	if LOAD_TEST:
		# test
		test_qa_map = pickle.load(open(saved_qa_map_format.format('test'), 'rb')) if os.path.exists(saved_qa_map_format.format('test')) \
			else preprocess.process_qas(json_filename_format.format('test'), glove_p_filename, save=saved_qa_map_format.format('test'))
		test_loader = DataLoader(VQADataset(img_feats, test_qa_map), batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)


	model = LSTMModel(visual_dim=FEAT_DIM, lang_dim=WV_DIM, hidden_dim=WV_DIM, out_dim=1, mlp_dims=[1024, 512, 512], bidirectional=bidir, img2seq=img2seq)
	loss_fn = torch.nn.BCEWithLogitsLoss()
	optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
	if use_pretrain and pretrained_path:
		pretrained = torch.load(pretrained_path)
		model.load_state_dict(pretrained['model'])
		optim.load_state_dict(pretrained['optim'])
		# set model lr to new lr
		for param_group in optim.param_groups:
			param_group['lr'] = lr
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
			t_start_ep = time.time()
			print("\n\n==== Epoch {:d} ====".format(e+1))
			train(model, optim, train_loader)
			val_acc = eval(model, val_loader, update_stats=True)
	
			with open(log_file, "w") as handle:
				json.dump(stats, handle)
			
			if val_acc > best_acc:
				best_acc = val_acc
				torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, checkpoint)
			print("Time taken: {}m".format((time.time()-t_start_ep)/60))
			print("File: {}".format(log_file))

	# Evaluate on test set
	print("\nEvaluating on test set...")
	eval(model, test_loader, save='bow_test_top1.npy')

	print("\nTotal Time: {}h".format((time.time()-t_start_total)/60/60))
