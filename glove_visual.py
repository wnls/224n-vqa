import os
import pickle
import numpy as np
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

def get_embed_func(mapping, embeds):
    return lambda x: embeds[mapping[x]]

def tsne_plot(vocab, embed_func, save=''):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in vocab:
        tokens.append(embed_func(word))
        labels.append(word)
        # print(word)
    
    tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    if save:
        plt.savefig(save)

def tsne(vocab, embed_func):
    "Creates and TSNE model"
    labels = []
    tokens = []

    for word in vocab:
        tokens.append(embed_func(word))
        labels.append(word)
        # print(word)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    return x, y, labels

def plot_single(glove_x, glove_y, glove_labels, tuned_x, tuned_y, tuned_labels, save=''):
    fig, axes = plt.subplots(ncols=2, figsize=(20, 9))
        
    # plt.figure(figsize=(16, 16)) 
    for i in range(len(glove_x)):
        axes[0].scatter(glove_x[i], glove_y[i])
        if glove_labels[i] in ['what', 'why', 'how', 'where', 'when', 'who']:
            axes[0].annotate(glove_labels[i],
                         xy=(glove_x[i], glove_y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    for i in range(len(tuned_x)):
        axes[1].scatter(tuned_x[i], tuned_y[i])
        if tuned_labels[i] in ['what', 'why', 'how', 'where', 'when', 'who']:
            axes[1].annotate(tuned_labels[i],
                         xy=(tuned_x[i], tuned_y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    if save:
        plt.savefig(save)

if __name__ == '__main__':
    random.seed(0)
    vocab = pickle.load(open('data/vocab.pkl', 'rb'))
    vocab_selected = ['what', 'why', 'how', 'where', 'when', 'who']
    vocab_selected += random.sample(vocab, 500)
    mapping = pickle.load(open('data/vqa_glove_vocab2ind.pkl', 'rb'))
    glove = pickle.load(open('data/vqa_glove_embeddings.pkl', 'rb'))
    tuned = np.load(open('data/learned_embeds.npy', 'rb'))
    # tsne_plot(vocab_selected, get_embed_func(mapping, glove), save='tsne_glove_selected200_random0_tsne3.png')
    # tsne_plot(vocab_selected, get_embed_func(mapping, tuned), save='tsne_tuned_selected200_random0_tsne3.png')
    glove_x, glove_y, glove_labels = tsne(vocab_selected, get_embed_func(mapping, glove))
    tuned_x, tuned_y, tuned_labels = tsne(vocab_selected, get_embed_func(mapping, tuned))
    plot_single(glove_x, glove_y, glove_labels, tuned_x, tuned_y, tuned_labels, save='tsne_selected500_random0_tsne2.png')


# best rank:
#   LSTM: checkpoints/rank_hd400_nl1_dp0.0_lr5e-05_wd0.0_bts128_ep300_0321001739.json
#   BOW: 
# best BCE:



