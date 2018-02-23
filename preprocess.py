import json, string, pickle
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import WordPunctTokenizer
import numpy as np

WORD_DIM = 300

# glove2word2vec(glove_input_file='/Users/Weini/Dev/GloVe/glove.6B/glove.6B.300d.txt', word2vec_output_file='/Users/Weini/Dev/GloVe/glove.6B/word2vec_glove.6B.300d.txt')
# glove_model = KeyedVectors.load_word2vec_format("/Users/Weini/Dev/GloVe/glove.6B/word2vec_glove.6B.300d.txt", binary=False)
# with open('/Users/Weini/Dev/GloVe/glove.6B/word2vec_glove.6B.300d.pkl', 'wb') as f:
#     pickle.dump(glove_model, f)

def sent2vec(glove_embed, sentence, word_dim):
	# v = []
	v = np.array([]).reshape(0, word_dim)
	for w in sentence:
		try:
			v = np.vstack((v, glove_embed[w]))
			# v.append(glove_embed[w])
		except KeyError as e:
			# print(e)
			# ise = True
			# qv.append(np.random.rand(WORD_DIM,))
			pass
	return v

def process_qas(json_filename, glove_p_filename):
	with open(glove_p_filename,'rb') as f:
		glove_embed = pickle.load(f)

	qa_map = []
	with open(json_filename) as file:
		json_data = json.load(file)
		# qa_data = []
		tokenizer = WordPunctTokenizer()
		# qa_id_list = []
		# image_id_list = []
		for img in json_data['images']:
			# image_id_list.append(img["image_id"])
			# if img['split'] == 'train':

			# elif img['split'] == 'val':

			for qapair in img['qa_pairs']:
				# qa_id, image_id
				# qa_id_list.append(qapair['qa_id'])
				q = tokenizer.tokenize(qapair['question'].strip(string.punctuation).lower())
				answers = [tokenizer.tokenize(qapair['answer'].strip(string.punctuation).lower())]
				for choice in qapair['multiple_choices']:
					answers.append(tokenizer.tokenize(choice.strip(string.punctuation).lower()))

				# qv = np.array([]).reshape(WORD_DIM,)
				qv = sent2vec(glove_embed, q, WORD_DIM)
				# print("qv", qv.shape)
				# qa_m = np.array([]).reshape(0, WORD_DIM)
				av_list = []
				for ans in answers:
					av = sent2vec(glove_embed, ans, WORD_DIM)
					av_list.append(av)
					# together = np.vstack((qv, ansv))
					# print("together", together.shape)
					# qa_m = np.hstack((qa_m, together))	
					# try:
					# 	ansv = sent2vec(glove_embed, ans, WORD_DIM)
					# 	together = np.vstack((qv, ansv))
					# 	print("together", together.shape)
					# 	qa_m = np.hstack((qa_m, together))
					# except:
					# 	print("error")
					# 	print(ansv.shape)
					# 	print(np.vstack((qv, ansv)).shape)
					# 	print(ans)
					# 	exit()
				qa = {'qa_id': qapair['qa_id'],
						'img_id': img["image_id"],
						'q_embed': qv,
						'a_embeds': av_list}
				qa_map.append(qa)
	return qa_map

if __name__ == '__main__':
	json_filename = './data/visual7w-telling.json'
	glove_p_filename = './data/word2vec_glove.6B.300d.pkl'
	print(process_qas(json_filename, glove_p_filename)[0])

	# print('image_id_list len', len(image_id_list))
	# print('image_id_list min', min(image_id_list))
	# print('image_id_list max', max(image_id_list))
	# print('qa_id_list len', len(qa_id_list))
	# print('qa_id_list min', min(qa_id_list))
	# print('qa_id_list max', max(qa_id_list))

