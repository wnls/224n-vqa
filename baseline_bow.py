import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import preprocess
from torch.utils.data import DataLoader, Dataset

class VQADataset(Dataset):
	def __init__(self, img_features, img_map, qa_map):
		self.img_features = img_features
		self.img_map = img_map
		self.qa_map = qa_map

	def __getitem__(self, index):
		#TODO gpu?
		qa_id, img_id, q_embed, a_embeds = self.a_map[index]
		
		return q_embed, a_embeds, self.img_features[qa[img_id]], [1,0,0,0]

	def __len__(self):
		return len(self.qa_map)

if __name__ == '__main__':
	json_filename = './data/visual7w-telling.json'
	glove_p_filename = './data/word2vec_glove.6B.300d.pkl'
	qa_map = preprocess.process_qas(json_filename, glove_p_filename)

	# img_map:{img_id: img_feat_idx}

	model = LSTMModel()

	sents = ['What is the color of the cat red', 'what is the color of the cat blue', "What is the woman wearing"]
	feats = []
	glove_embed = # load embedding
	for sent in sents:
		feats += np.stack([glove_embed[word] for word in sent]),
	max_seq_len = max([len(each) for each in feats])
	# pad if same batch; reshape if single example
	feats = # from glove
	dataset = MyDataset(feats, labels)
	data_loader = some_loader(dataset, batch_size, shuffle=True) # torch builtin
	

	model.train()
	loss_fn = torch. # loss func
	optim = torch.optim.Adam(model.state())
	for batch in data_loader:
		feats, labels = batch
		out, h = model.lstm(feats)
		score = model.linear(out)
		loss = loss_fn(score, labels)
		loss.backward()
		model.linear.backward()
		model.lstm.backward()
		optim
