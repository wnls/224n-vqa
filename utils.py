import json
import numpy as np
from glob import glob
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='best', type=str)
parser.add_argument('--attribute', default='val_acc', type=str)

def get_qtype_matrix(json_name, save=''):
  data = json.load(open(json_name, 'r'))
  result_mtrx = []
  for img in data['images']:
    for qapair in img['qa_pairs']:
      q = qapair['question']
      result_onehot = [0]*6
      if 'What' in q:
        result_onehot[0] = 1
      elif 'Where' in q:
        result_onehot[1] = 1
      elif 'When' in q:
        result_onehot[2] = 1
      elif 'Who' in q:
        result_onehot[3] = 1
      elif 'Why' in q:
        result_onehot[4] = 1
      elif 'How' in q:
        result_onehot[5] = 1
      else:
        print('Unclassified question:', q)
      result_mtrx.append(result_onehot)
  ret_mtrx = np.asarray(result_mtrx) # size: n * 6
  if save:
    with open(save, 'w') as handle:
      np.save(handle, ret_mtrx)
  return ret_mtrx

def eval_by_type(result_vec, type_mtrx):
  if result_vec.shape[0] != type_mtrx.shape[0]:
    print('result_vec:', result_vec.shape)
    print('type_mtrx:', type_mtrx.shape)
    raise ValueError('Size mismatch.')

  name_dict = {0:'What', 1:'Where', 2:'When', 3:'Who', 4:'Why', 5:'How'}
  for t in range(6):
    type_acc = type_mtrx[:,t].dot(result_vec) / type_mtrx[:, t].sum()
    print('{:s}: {:f}'.format(name_dict[t], type_acc))
  print('Overall: {:f}'.format(result_vec.sum() / result_vec.shape[0]))


def check_best(attribute='val_acc'):
  best_vals = []
  for fname in glob('checkpoints/*.json'):
    data = json.load(open(fname, 'r'))
    if attribute in data:
      best_vals += (max(data[attribute]), fname),
    else:
      print('{:s} not in {:s}.\nKeys available: {}'.format(attribute, fname, data.keys()))
  sorted_best = sorted(best_vals, key=lambda x:x[0], reverse=True)
  if best_vals:
    print('Best vals for', attribute)
    for i,(val,fname) in enumerate(sorted_best):
      print('({:d})'.format(i), val, fname)

def flat_spatial(fname_in='resnet101_layer4.h5'):
  fin = h5py.File(fname_in)
  keys = list(fin.keys())
  fname_out = fname_in.replace('.h5', '_flatten.h5')
  print('Flattened features saving at', fname_out)
  fout = h5py.File(fname_out)
  for key in keys:
    feat = fin[key]
    assert(feat.ndim == 3) # (2048, 7, 7)
    flat = np.reshape(fin[key], (feat.shape[0], feat.shape[1]*feat.shape[2]))
    flat = flat.transpose(1,0)
    assert(flat.shape[0]==49)
    fout.create_dataset(key, data=flat)
  fout.close()



if __name__ == '__main__':
  args = parser.parse_args()
  # result_vec = np.load('bow_test_top1.npy')
  # type_mtrx = get_qtype_matrix(json_name='data/visual7w-telling_test.json')
  # eval_by_type(result_vec, type_mtrx)
  if args.mode == 'best':
    check_best(args.attribute)
  elif args.mode == 'flatten':
    flat_spatial('data/resnet101_layer4.h5')
