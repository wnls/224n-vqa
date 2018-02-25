## File list:
* `baseline_bow.py`: baseline model using image features and bad-of-word representation of text.
  * To train the model, run `python baseline_bow.py`. Models and log files (containing training loss only) are placed under `./checkpoints/`, with file extensions `.pt` and `.json` respectively.
* `preprocess.py`: for processing `qa_map`
* `preprocess_feat.py`: for extracting features from resnet101 (2048-dim vector from avgpool)
* `lstm_example.py`: an example of LSTM
* `README_GloVe.md`: how to use GloVe
* `./data/`: cache files for word embedding, `qa_map` and image features. Ignored by `.gitignore`.
* `./checkpoints/`: model checkpoints and training log files. Ignored by `.gitignore`.
