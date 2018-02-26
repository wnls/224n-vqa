## Train the BOW baseline
Command: `python baseline_bow.py`
* Hyperparameters: listed at the beginning of the file, after the import statements. `lr`, `wd` and `batch_size` are shown in the checkpoint path.
* Load pretrained model: to use a pretrained model, set the variable `pretrained_path` under `__main__` as the pretrained model path. You may need to check your `checkpoint` variable to avoid overwriting the pretrained file, since the checkpoint does not reflect the number of epochs.

## File list:
* `baseline_bow.py`: baseline model using image features and bad-of-word representation of text.
  * To train the model, run `python baseline_bow.py`. Models and log files (containing training loss only) are placed under `./checkpoints/`, with file extensions `.pt` and `.json` respectively.
* `preprocess.py`: for processing `qa_map`
* `preprocess_feat.py`: for extracting features from resnet101 (2048-dim vector from avgpool)
* `lstm_example.py`: an example of LSTM
* `README_GloVe.md`: how to use GloVe
* `./data/`: cache files for word embedding, `qa_map` and image features. Ignored by `.gitignore`.
* `./checkpoints/`: model checkpoints and training log files. Ignored by `.gitignore`.
