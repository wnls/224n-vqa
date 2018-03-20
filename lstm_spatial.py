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
import argparse
import numpy as np
import h5py
import pickle
import json
# self-defined modules
import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
# Model
parser.add_argument('--use_pretrain', default=0, type=int)
parser.add_argument('--pretrained_path', default='./checkpoints/bi_lr5e-07_wd0.0005_bts128_ep100_0311201826_continue5.pt', type=str)
parser.add_argument('--feat_dim', default=2048, type=int)
parser.add_argument('--wv_dim', default=300, type=int)
parser.add_argument('--bidir', default=False, type=bool)
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--img2seq', default=False, type=bool)
parser.add_argument('--lstm_hidden_dim', default=200, type=int)
parser.add_argument('--att_hidden_dim', default=200, type=int)
parser.add_argument('--n_levels', default=1, type=int)
# Training
parser.add_argument('--print_every_train', default=50, type=int)
parser.add_argument('--print_every_val', default=100, type=int)
# Optimization
parser.add_argument('--loss', default='BCE', type=str)
parser.add_argument('--margin', default=0.6, type=float)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--wd', default=5e-7, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--n_epoch', default=200, type=int)
parser.add_argument('--finetune_embeds', default=False, type=bool)
# Files
parser.add_argument('--outDir', default='./checkpoints', type=str)
parser.add_argument('--json_filename_format', default='./data/visual7w-telling_{:s}.json', type=str)
parser.add_argument('--img_feats_fname', default='./data/resnet101_layer4_flatten.h5', type=str)
parser.add_argument('--vocab_p_filename', default='./data/vqa_glove_vocab2ind.pkl', type=str)
parser.add_argument('--embeds_p_filename', default='./data/vqa_glove_embeddings.pkl', type=str)
parser.add_argument('--saved_qa_map_format', default='./data/qa_embeds_{:s}.pkl', type=str)

t_start_total = time.time()

# system
USE_GPU = torch.cuda.is_available()
# data
dtype = torch.FloatTensor
n_vocab = None

# Exec related
LOAD_TRAIN = True
LOAD_VAL = True
LOAD_TEST = True

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
    * q_embed: list (np array) of embedding indices for question words, shape (n_q,)
    * a_embeds: list of 4 ndarrays, each of size (n_a,); the 1st item is GT
    * img_feats: (2048,) i.e. 1D vector
    """
    qa_encode = []
    for a_embed in a_embeds:
      qa_encode.append(np.concatenate([q_embed, a_embed], 0))
    img_encode = np.tile(np.asarray(self.img_features[img_id]), [4,1,1]) # NOTE: change for [4,1] to [4,1,1] for spatial features

    sys.stdout.flush()
    return qa_encode, img_encode, np.asarray([1,0,0,0]).reshape(4,1)
    # return np.tile(q_embed.mean(0), [4,1]), np.stack([a_embed.mean(0) for a_embed in a_embeds]), np.tile(np.asarray(self.img_features[img_id]), [4,1]), np.asarray([1,0,0,0]).reshape(4,1)

  def __len__(self):
    return len(self.qa_map)

def pad_collate_fn(batch):
  """
  Input: batch: list
  Output:
  * seq_lens: IntTensor - sorted seq_lens in decreasing order
  * indices: LongTensor - indices of sorted seq_lens in decreasing order
  * qa_embeds_padded: LongTensor - qa embed lookup indices, padded to batch max length with zeros
  * img_feats[indices]: FloatTensor - img_feats sorted by indices
  * labels[indices]: FloatTensor - labels sorted by indices
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
    padded = np.concatenate([qa_embed, np.zeros([max_len - qa_embed.shape[0]])], axis=0)
    qa_embeds_padded.append(padded)
  qa_embeds_padded = torch.LongTensor(np.stack(qa_embeds_padded, 0))

  if USE_GPU:
    seq_lens = seq_lens.cuda()
    indices = indices.cuda()
    qa_embeds_padded = qa_embeds_padded.cuda()
    img_feats = img_feats.cuda()
    labels = labels.cuda()

  print('pad collate fn:')
  print('indices size:', indices.size())
  print('qa_embeds_padded size:', qa_embeds_padded.size())
  print('img_feats size:', img_feats.size())
  print('labels size:', labels.size())
  return seq_lens, indices, qa_embeds_padded, img_feats[indices], labels[indices]

class LSTMAttentionModel(nn.Module):
  def __init__(self, visual_dim, lang_dim, lstm_hidden_dim, att_hidden_dim, out_dim=1, mlp_dims=[], embed_weights=None, finetune_embeds=False,
    bidirectional=False, n_layers=1, dropout=0, img2seq=False, n_levels=1):
    super(LSTMAttentionModel, self).__init__()
    #self.drop = nn.Dropout(dropout)

    self.lstm = nn.LSTM(lang_dim, lstm_hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
    self.visual_dim = visual_dim
    self.lang_dim = lang_dim
    self.lstm_hidden_dim = lstm_hidden_dim
    self.embed_weights = embed_weights
    self.n_layers = n_layers
    self.bidirectional = bidirectional
    self.img2seq = img2seq
    self.n_levels = n_levels # number of levels for attention
    self.att_hidden_dim = att_hidden_dim
    self.function_modules = {}
    # lang embedding is first mapped to a common embedding space as visual
    self.lang_embed = nn.Linear(lstm_hidden_dim, visual_dim)
    # Q: do we need the activation layer?
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax()

    for l in range(n_levels):
      # mapping layer for language
      mod = nn.Linear(visual_dim, att_hidden_dim)
      self.add_module('wq{:d}'.format(l), mod)
      self.function_modules['wq{:d}'.format(l)] = mod
      # mapping layer for visual -- no bias
      mod = nn.Linear(visual_dim, att_hidden_dim, bias=False)
      self.add_module('wi{:d}'.format(l), mod)
      self.function_modules['wi{:d}'.format(l)] = mod
      # attention layer
      mod = nn.Linear(att_hidden_dim, out_dim)
      self.add_module('wp{:d}'.format(l), mod)
      self.function_modules['wp{:d}'.format(l)] = mod

    self.embeds = nn.Embedding(n_vocab, self.lang_dim)
    if embed_weights is not None:
      # embed_weights = torch.FloatTensor(embed_weights)
      # self.embeds = nn.Embedding.from_pretrained(embed_weights)
      self.embeds.weight.data.copy_(torch.from_numpy(embed_weights))
      # self.embeds.weight = nn.Parameter(embed_weights)
    if not finetune_embeds:
      self.embeds.weight.requires_grad = False

    # if using img as the first input of word sequence
    if img2seq:
      self.img2worddim = nn.Sequential(nn.Linear(visual_dim, lang_dim), nn.ReLU())

    layers = []
    # Current: concat qa & visual dim
    dims = [visual_dim] + mlp_dims + [out_dim]
    for i in range(len(dims)-1):
      layers += nn.Linear(dims[i], dims[i+1]),
      if i != len(dims)-2:
        # i.e. no ReLU at the last layer
        layers += nn.ReLU(),
      if args.dropout>0:
        layers += nn.Dropout(args.dropout),
    self.mlp = nn.Sequential(*layers)

  def init_hidden(self, batch_size):
    #TODO batch size
    if USE_GPU:
      return (Variable(torch.zeros(self.n_layers, batch_size, self.lstm_hidden_dim)).cuda(),
          Variable(torch.zeros(self.n_layers, batch_size, self.lstm_hidden_dim)).cuda())
    else:
      return (Variable(torch.zeros(self.n_layers, batch_size, self.lstm_hidden_dim)),
          Variable(torch.zeros(self.n_layers, batch_size, self.lstm_hidden_dim)))

  def forward(self, lang_indicies, img_input, seq_lens):
    '''
    lang_input: LongTensor - (batch, seq_len)
    img_input: tensor - (batch_size, img_dim)
    seq_lens: tensor - list of seq lengtsh in decreasing size, for pack_padded_sequence
    '''
    lang_input = self.embeds(lang_indicies)
    if self.img2seq:
      img_token = self.img2worddim(img_input)
      img_token = torch.unsqueeze(img_token, 1)
      lang_input = torch.cat([img_token, lang_input], dim=1)
      seq_lens += 1

    lang_input_packed = nn.utils.rnn.pack_padded_sequence(lang_input, list(seq_lens), batch_first=True)

    # self.hidden = self.init_hidden(lang_input.size(0))
    # hidden size zeros by default
    out, (h_t, c_t) = self.lstm(lang_input_packed) # h_t: (num_layers * num_directions, batch_size, hidden_size)
    print('h_t size:', h_t.size())
    h_t = torch.mean(h_t, 0)
    assert(h_t.size(0) == img_input.size(0)), "size mismatch: h_t: {} / img_input: {}".format(h_t.size(), img_input.size())
    # map hidden_lang to visual embedding space
    v_q = self.lang_embed(h_t)
    print('v_q size:', v_q.size())
    print('img_input size:', img_input.size())
    u_curr = v_q
    for l in range(self.n_levels):
      v_embed = self.function_modules['wi{:d}'.format(l)](img_input)
      l_embed = self.function_modules['wq{:d}'.format(l)](u_curr)
      if v_embed.dim() != l_embed.dim():
        l_embed = l_embed.unsqueeze(1)
      print('v_embed size:', v_embed.size())
      print('l_embed size:', l_embed.size())
      h_a = self.tanh(l_embed.expand_as(v_embed) + v_embed)
      print('h_a size:', h_a.size())
      p_i = self.softmax(self.function_modules['wp{:d}'.format(l)](h_a))
      print('p_i size:', p_i.size())
      v_i = (p_i.unsqueeze(1) * img_input).sum(-1)
      u_curr += v_i
    out = self.mlp(u_curr)
    return out


def train(args, model, optim, loader):
  model.train()

  top1_cnt = 0
  total_cnt = 0
  t_start = time.time()
  for i,batch in enumerate(loader):
    if i and i%args.print_every_train==0:
      avg_loss = sum(stats['train_loss'][-args.print_every_train:])/args.print_every_train
      avg_time = (time.time()-t_start)/args.print_every_train
      cur_acc = top1_cnt/total_cnt if total_cnt else -1
      print('{:d}-{:d}: avg_loss={:f} / avg_time={:f}s / cur_acc={:f}'.format(i-args.print_every_train, i, avg_loss, avg_time, cur_acc))
      sys.stdout.flush()
      t_start = time.time()

    # unroll a batch
    seq_lens, indices, qa_embeds, img_feats, gt = batch
    this_batch_size = int(seq_lens.size(0)/4)
    # Variable for autograd
    qa_embeds_var = Variable(qa_embeds).type(torch.LongTensor) # lookup indices to nn.Embeddings have to be LongTensor
    img_feats_var = Variable(img_feats).type(dtype)
    gt_var = Variable(gt).type(dtype)
    if USE_GPU:
      qa_embeds_var = qa_embeds_var.cuda()
      img_feats_var = img_feats_var.cuda()
      gt_var = gt_var.cuda()

    # forward
    out = model(qa_embeds_var, img_feats_var, seq_lens)
    if args.loss == 'BCE':
      loss = loss_fn(out, gt_var)
    elif args.loss == 'rank':
      # pos_col = out[indices[0::4]].expand(this_batch_size, 3).view(-1)
      _, unsort_ind = indices.sort()
      out_orig_order = out[unsort_ind].view(-1, 4)
      pos_col = out_orig_order[:,0]
      neg_col = out_orig_order[:,1:].mean(dim=1)
      flag_col = torch.ones_like(pos_col)
      if USE_GPU:
        pos_col = pos_col.cuda()
        neg_col = neg_col.cuda()
        flag_col = flag_col.cuda()
      loss = loss_fn(pos_col, neg_col, flag_col)
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


def eval(args, model, loader, update_stats=False, save=''):
  model.eval()

  top1_cnt = 0
  total_cnt = 0
  top1_result = []
  t_start = time.time()
  for i,batch in enumerate(loader):
    if i and i%args.print_every_val==0:
      avg_time = (time.time()-t_start)/args.print_every_val
      cur_acc = top1_cnt/total_cnt if total_cnt else -1
      print('{:d}-{:d}: avg_time={:f}s / cur_acc={:f}'.format(i-args.print_every_val, i, avg_time, cur_acc))
      sys.stdout.flush()
      t_start = time.time()

    # unroll a batch
    seq_lens, indices, qa_embeds, img_feats, gt = batch
    this_batch_size = int(seq_lens.size(0)/4)
    # Variable for autograd
    qa_embeds_var = Variable(qa_embeds).type(torch.LongTensor)
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
  args = parser.parse_args()

  # output files
  if not os.path.exists(args.outDir):
    os.mkdir(args.outDir)
  now = datetime.now()
  file_format = os.path.join(args.outDir, '{}{}{}{}hd{}_nl{}_dp{}_lr{}_wd{}_bts{:d}_ep{:d}_{}'
                             .format("bi_" if args.bidir else "",
                     "img_" if args.img2seq else "",
                     "emb_" if args.finetune_embeds else "",
                     "{:s}_".format(args.loss),
                     args.lstm_hidden_dim, args.n_layers, args.dropout, args.lr, args.wd, args.batch_size, args.n_epoch, time.strftime("%m%d%H%M%S")))
  log_file = file_format + '.json'
  checkpoint = file_format + '.pt'


  # img_feat # TODO: split img features to be efficient
        # NOTE: make a copy of the data in img_feats so that we can release the h5py file object
  with h5py.File(args.img_feats_fname) as handle: 
    img_feats = {key:np.asarray(handle[key]) for key in handle}
  print('Finish loading img_feats.') # NOTE: indicating the file has been released

  embeds = pickle.load(open(args.embeds_p_filename, 'rb'))
  n_vocab = len(embeds)

  if LOAD_TRAIN:
    # train
    train_qa_map = pickle.load(open(args.saved_qa_map_format.format('train'), 'rb')) if os.path.exists(args.saved_qa_map_format.format('train')) \
      else preprocess.process_qas_embeds(json_filename_format.format('train'), args.vocab_p_filename, save_file=args.saved_qa_map_format.format('train'))
    train_loader = DataLoader(VQADataset(img_feats, train_qa_map), batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
  if LOAD_VAL:
    # val
    val_qa_map = pickle.load(open(args.saved_qa_map_format.format('val'), 'rb')) if os.path.exists(args.saved_qa_map_format.format('val')) \
      else preprocess.process_qas_embeds(json_filename_format.format('val'), args.vocab_p_filename, save_file=args.saved_qa_map_format.format('val'))
    val_loader = DataLoader(VQADataset(img_feats, val_qa_map), batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)
  if LOAD_TEST:
    # test
    test_qa_map = pickle.load(open(args.saved_qa_map_format.format('test'), 'rb')) if os.path.exists(args.saved_qa_map_format.format('test')) \
      else preprocess.process_qas_embeds(json_filename_format.format('test'), args.vocab_p_filename, save_file=args.saved_qa_map_format.format('test'))
    test_loader = DataLoader(VQADataset(img_feats, test_qa_map), batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)


  model = LSTMAttentionModel(visual_dim=args.feat_dim, lang_dim=args.wv_dim, lstm_hidden_dim=args.lstm_hidden_dim, att_hidden_dim=args.att_hidden_dim, out_dim=1, mlp_dims=[1024, 512, 512], embed_weights=embeds, finetune_embeds=args.finetune_embeds, n_layers=args.n_layers, bidirectional=args.bidir, img2seq=args.img2seq, dropout=args.dropout, n_levels=args.n_levels)
  if args.loss == 'BCE':
    loss_fn = torch.nn.BCEWithLogitsLoss()
  elif args.loss == 'rank':
    loss_fn = torch.nn.MarginRankingLoss(margin=args.margin)
    # only pass in parameters that require grad
  optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)

  if args.use_pretrain and args.pretrained_path:
    print('Using pretrained model', args.pretrained_path)
    pretrained = torch.load(args.pretrained_path)
    model.load_state_dict(pretrained['model'])
    optim.load_state_dict(pretrained['optim'])
    # set model lr to new lr
    for param_group in optim.param_groups:
      before = param_group['lr']
      param_group['lr'] = args.lr
      print('optim lr: before={} / after={}'.format(before, args.lr))
  if USE_GPU:
    print("Use GPU")
    model = model.cuda()
    loss_fn = loss_fn.cuda()
  else:
    print("Use CPU")

  if args.mode == 'train':
    best_acc = 0
    stats = {'train_loss':[], 'train_acc':[], 'val_acc':[]}
    for e in range(args.n_epoch):
      t_start_ep = time.time()
      print("\n\n==== Epoch {:d} ====".format(e+1))
      train(args, model, optim, train_loader)
      val_acc = eval(args, model, val_loader, update_stats=True)
  
      with open(log_file, "w") as handle:
        json.dump(stats, handle)
      
      if val_acc > best_acc:
        best_acc = val_acc
        torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, checkpoint)
      print("Time taken: {}m".format((time.time()-t_start_ep)/60))
      print("File: {}".format(log_file))

  # Evaluate on test set
  print("\nEvaluating on test set...")
  eval(args, model, test_loader, save='bow_test_top1.npy')

  print("\nTotal Time: {}h".format((time.time()-t_start_total)/60/60))
