import json, string, pickle
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
# from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
import numpy as np

WORD_DIM = 300

# glove2word2vec(glove_input_file='/Users/Weini/Dev/GloVe/glove.6B/glove.6B.300d.txt', word2vec_output_file='/Users/Weini/Dev/GloVe/glove.6B/word2vec_glove.6B.300d.txt')
# glove_model = KeyedVectors.load_word2vec_format("/Users/Weini/Dev/GloVe/glove.6B/word2vec_glove.6B.300d.txt", binary=False)
# with open('/Users/Weini/Dev/GloVe/glove.6B/word2vec_glove.6B.300d.pkl', 'wb') as f:
#     pickle.dump(glove_model, f)




def build_vocab(json_filename, glove_p_filename, save_vocab='./data/vqa_glove_vocab2ind.pkl', save_embeddings='./data/vqa_glove_embeddings.pkl'):
	vocab = {'NULL': 0}
	vecs = np.zeros(WORD_DIM)

	with open(glove_p_filename,'rb') as f:
		glove_embed = pickle.load(f)

	with open(json_filename) as file:
		json_data = json.load(file)
		for img in json_data['images']:
			for qapair in img['qa_pairs']:
				# qa_id, image_id
				li = []
				li.extend(word_tokenize(qapair['question'].strip(string.punctuation).lower()))
				# q = word_tokenize(qapair['question'].strip(string.punctuation).lower())
				li.extend(word_tokenize(qapair['answer'].strip(string.punctuation).lower()))
				# answers = [word_tokenize(qapair['answer'].strip(string.punctuation).lower())]
				for choice in qapair['multiple_choices']:
					# answers.append(word_tokenize(choice.strip(string.punctuation).lower()))
					li.extend(word_tokenize(choice.strip(string.punctuation).lower()))

				for t in li:
					if t not in vocab:
						if t in glove_embed:
							# vecs.append(glove_embed[t])
							vecs = np.vstack((vecs, glove_embed[t]))
							vocab[t] = vecs.shape[0]-1

	with open(save_vocab, 'wb') as handle:
		pickle.dump(vocab, handle)
	with open(save_embeddings, 'wb') as handle:
		pickle.dump(vecs, handle)

def process_qas_embeds(json_filename, vocab_p_filename, save_file=''):
	with open(vocab_p_filename,'rb') as f:
		vocab = pickle.load(f)

	qa_map = []
	with open(json_filename) as file:
		json_data = json.load(file)
		for img in json_data['images']:
			for qapair in img['qa_pairs']:
				# qa_id, image_id
				q = word_tokenize(qapair['question'].strip(string.punctuation).lower())
				answers = [word_tokenize(qapair['answer'].strip(string.punctuation).lower())]
				for choice in qapair['multiple_choices']:
					answers.append(word_tokenize(choice.strip(string.punctuation).lower()))

				q_ids = []
				as_ids = []
				for t in q:
					id = vocab[t] if t in vocab else 0
					q_ids.append(id)
				q_ids = np.asarray(q_ids, dtype=np.int32)
				for a in answers:
					a_ids = []
					for t in a:
						id = vocab[t] if t in vocab else 0
						a_ids.append(id)
					as_ids.append(np.asarray(a_ids, dtype=np.int32))

				qa = {'qa_id': qapair['qa_id'],
					  'img_id': img["image_id"],
					  'q_embed': q_ids, # nparray of indices
					  'a_embeds': as_ids} # list of 4 nparray of indices
				qa_map.append(qa)
				# print()
		if save_file:
			with open(save_file, 'wb') as handle:
				pickle.dump(qa_map, handle)
		return qa_map

def sent2vec(glove_embed, sentence, word_dim):
	v = np.array([]).reshape(0, word_dim)
	for w in sentence:
		if w in glove_embed:
			v = np.vstack((v, glove_embed[w]))
	if v.shape[0] == 0:
		v = np.vstack((v, np.zeros_like(glove_embed['this'])))
	return v

def process_qas(json_filename, glove_p_filename, save=''):
	with open(glove_p_filename,'rb') as f:
		glove_embed = pickle.load(f)

	qa_map = []
	with open(json_filename) as file:
		json_data = json.load(file)
		for img in json_data['images']:
			for qapair in img['qa_pairs']:
				# qa_id, image_id
				# qa_id_list.append(qapair['qa_id'])
				q = word_tokenize(qapair['question'].strip(string.punctuation).lower())
				answers = [word_tokenize(qapair['answer'].strip(string.punctuation).lower())]
				for choice in qapair['multiple_choices']:
					answers.append(word_tokenize(choice.strip(string.punctuation).lower()))

				# qv = np.array([]).reshape(WORD_DIM,)
				qv = sent2vec(glove_embed, q, WORD_DIM)
				# print("qv", qv.shape)
				# qa_m = np.array([]).reshape(0, WORD_DIM)
				av_list = []
				for ans in answers:
					av = sent2vec(glove_embed, ans, WORD_DIM)
					av_list.append(av)
				qa = {'qa_id': qapair['qa_id'],
						'img_id': img["image_id"],
						'q_embed': qv,
						'a_embeds': av_list}
				qa_map.append(qa)
	if save:
		with open(save, 'wb') as handle:
			pickle.dump(qa_map, handle)
	return qa_map

if __name__ == '__main__':
	json_filename = './data/visual7w-telling.json'
	glove_p_filename = './data/word2vec_glove.6B.300d.pkl'
	# print(process_qas(json_filename, glove_p_filename)[0])

	build_vocab(json_filename, glove_p_filename)

	process_qas_embeds('./data/visual7w-telling_train.json', './data/vqa_glove_vocab2ind.pkl', './data/qa_embeds_train.pkl')
	process_qas_embeds('./data/visual7w-telling_val.json', './data/vqa_glove_vocab2ind.pkl', './data/qa_embeds_val.pkl')
	process_qas_embeds('./data/visual7w-telling_test.json', './data/vqa_glove_vocab2ind.pkl', './data/qa_embeds_test.pkl')

	# print('image_id_list len', len(image_id_list))
	# print('image_id_list min', min(image_id_list))
	# print('image_id_list max', max(image_id_list))
	# print('qa_id_list len', len(qa_id_list))
	# print('qa_id_list min', min(qa_id_list))
	# print('qa_id_list max', max(qa_id_list))

