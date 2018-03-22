import numpy as np
import json
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imresize

FILTER_BETTER = True # only look at examples where stacked attention outperforms plain LSTM
if FILTER_BETTER:
  better_qids = list(np.load('results/nLv2_better_than_lstm.npy'))

n_level = 2

lv1 = np.load('results/att_49_lv1.npy')
lv2 = np.load('results/att_49_lv2.npy')
top1 = np.load('results/spatial_test_top1.npy')

img_dir = '/home/grapefruit/vqa/dataset/images/'
hm_dir = '/home/grapefruit/224n-vqa/heatmaps_4_better_than_LSTM/'
if not os.path.exists(hm_dir):
  os.mkdir(hm_dir)

def display_helper(ax, attn, title='', interp='lanczos'):
    ''' Convenience function to plot an attention on a set of axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The set of axes on which to display the attention.
        
    attn : numpy.ndarray, shape (r, c)
        The attention mask to display.
        
    title : str, optional (default='')
        The title to display over the plot.
    
    interp : str, optional (default='lanczos')
        The type of interpolation to use. In our paper, we use lanczos interpolation
        for the smoothed images and nearest for the rest of our visualizations.
        Legal options are: (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’, ‘cubic’).
    '''
    interp = interp if interp else 'nearest'
    ax.imshow(imresize(attn, (224, 224), interp))
    ax.axis('off')
    ax.set_title(title);


gt = json.load(open('data/visual7w-telling_test.json', 'r'))
qa_pairs = [pair for image in gt['images'] for pair in image['qa_pairs']]

if n_level == 2:
  cnt = 0
  q_cnt = 0
  for each in gt['images']:
    # if '2346306' not in each['filename']:
    #   cnt += len(4*each['qa_pairs'])
    #   continue
    for pair in each['qa_pairs']:
      if FILTER_BETTER and q_cnt not in better_qids:
        q_cnt += 1
        continue
      print(q_cnt)
      img = cv2.imread(os.path.join(img_dir, each['filename']))
      # NOTE: the following lines are for temporary checking
      # fig, axes = plt.subplots(ncols=5, figsize=(50,9))
      # display_helper(axes[0], img)
      # for i in range(1,5):
      #   att_lv1 = lv1[cnt].reshape(7,7)
      #   att_lv2 = lv2[cnt].reshape(7,7)
      #   combined = att_lv1 + att_lv2
      #   display_helper(axes[i], att_lv1)
      #   cnt += 1
      fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 10))
      answers = [pair['answer']] + pair['multiple_choices']
      for i in range(4):
        att_lv1 = lv1[cnt].reshape(7,7)
        att_lv2 = lv2[cnt].reshape(7,7)
        display_helper(axes[0, i], img, 'Input Image')
        display_helper(axes[1, i], att_lv1, 'Lv1 Attention')
        display_helper(axes[2, i], att_lv2, 'Lv2 Attention')
        display_helper(axes[3, i], att_lv1+att_lv2, 'Combined Attention')
        axes[0,i].set_title(answers[i])
        cnt += 1
      fig.text(0.1, 0.9, pair['question'])
      save_name = each['filename'].replace('.jpg', '_att4.jpg')
      if top1[cnt] == 1:
        save_name = '0_' + save_name
      plt.savefig(os.path.join(hm_dir, save_name))
      plt.close()
      q_cnt += 1
elif n_level == 1:
  cnt = 0
  for each in gt['images']:
    for pair in each['qa_pairs']:
      img = cv2.imread(os.path.join(img_dir, each['filename']))
      fig, axes = plt.subplots(ncols=5, figsize=(18, 10))
      answers = [pair['answer']] + pair['multiple_choices']
      display_helper(axes[0], img)
      for i in range(1,5):
        att_lv1 = lv1[cnt].reshape(7,7)
        display_helper(axes[i], att_lv1)
        axes[i].set_title(answers[i])
        cnt += 1
      fig.text(0.1, 0.9, pair['question'])
      save_name = each['filename'].replace('.jpg', '_att4.jpg')
      if top1[cnt] == 1:
        save_name = '0_' + save_name
      plt.savefig(os.path.join(hm_dir, save_name))
      plt.close()
else:
  raise NotImplementedError("Currently only have 1 or 2 attention levels.")
