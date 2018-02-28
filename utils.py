import json
import numpy as np

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

if __name__ == '__main__':
  result_vec = np.load('bow_test_top1.npy')
  type_mtrx = get_qtype_matrix(json_name='data/visual7w-telling_test.json')
  eval_by_type(result_vec, type_mtrx)
