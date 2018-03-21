import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from glob import glob
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='best', type=str, help="'best' for outputing best results from all matching files; 'plot' for plotting curve for selected files.")
parser.add_argument('--dir', default='checkpoints/')
parser.add_argument('--format', default='*.json', type=str)
parser.add_argument('--fig_format', default='', type=str)
parser.add_argument('--attribute', default='val_acc', type=str)
parser.add_argument('--multi', default='sep', type=str)
# args for smoothing
parser.add_argument('--window', default=20, type=int)
parser.add_argument('--step', default=10, type=int)

args = parser.parse_args()

# files = ['bi_lr2e-06_wd0.0005_bts128_m3d10h9m47s31_continue.json','bi_lr5e-06_wd0.0005_bts128_ep300_0310202638_continue2.json']

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot(train_loss, train_acc, val_acc, figname='results.png'):
	s = smooth(train_loss[::args.step], args.window)

	fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
	ax0.plot(s, 'g-', lw=1)
	ax0.set_title("Train Loss")
	# fig.autofmt_xdate()

	print("Best train_acc: {}".format(max(train_acc)))
	ax1.plot(train_acc)
	ax1.set_title("Train Acc")
	# fig.autofmt_xdate()

	print("Best val_acc: {}".format(max(val_acc)))
	ax2.plot(val_acc)
	ax2.set_title("Val Acc")

	plt.savefig(figname)


files = sorted(glob(os.path.join(args.dir, args.format)))

if args.multi == 'sep':
# one plot per file
	for fname in files:
		with open(fname) as f:
			data = json.load(f)
			plot(data['train_loss'], data['train_acc'], data['val_acc'])

elif args.multi == 'stitch':
# stitch results from multiple files
	train_loss = []
	train_acc = []
	val_acc = []

	for fname in files:
		with open(fname) as f:
			data = json.load(f)
			train_loss.extend(data['train_loss'])
			train_acc.extend(data['train_acc'])
			val_acc.extend(data['val_acc'])
	plot(train_loss, train_acc, val_acc)




