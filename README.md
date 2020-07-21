# CS-Embed at SemEval-2020 Task 9: The effectiveness of code-switched word embeddings for sentiment analysis
Code and specs for CS-Embed's contribution to SemEval-2020 Task 9.

* tweet_ids.zip : contains the tweet-id's of the tweets used to create the code-switched embeddings
* cs_kw.txt: Spanish and Spanglish Keywords used to extract tweets from twitter
* tweet_collect.py: code used to collect tweets from twitter using Tweepy and cs_kw.txt as keyword list
* cs_model.py: code used to train bilstm model
* cs_embeddings.tar.gz: word2vec code-switched embeddings with dimension 100. These are the main contribution for SemEval2020: Task 9


### Code-Switch BiLSTM Model Summary
_________________________________________________________________
|Layer (type)|Output Shape|Param No.|   
|-----------------------|-------------------------|--------------|
|embedding (Embedding)|(None, 12, 100)|21592000|
|bidirectional (Bidirectional)|(None, 12, 256)|234496|
|bidirectional_1 (Bidirectional)|(None, 256)|394240|
|dropout (Dropout)|(None, 256)|0|
|dense (Dense)|(None, 100)|25700|
|dropout_1 (Dropout)|(None, 100)|0|
|dense_1 (Dense)|(None, 100)|10100|
|dropout_2 (Dropout)|(None, 100)|0|
|dense_2 (Dense)|(None, 3)|303|
_________________________________________________________________
Total params: 22,256,839
Trainable params: 22,256,839
Non-trainable params: 0
_________________________________________________________________

**Hyperparameters of BiLSTM Model**
* Optimiser: Adamax
* Learning rate:0.0002
* EarlyStopping: min_delta=0.0001, patience=5


If any code or models are used please cite:

@InProceedings{Leon2020,
  author    = {Frances A. Laureano De Leon and Florimond Guéniat and Harish Tayyar Madabushi},
  title     = {CS-Embed at SemEval-2020 Task 9: The effectiveness of code-switched word embeddings for sentiment analysis},
  booktitle = {Proceedings of the 14th International Workshop on Semantic Evaluation ({S}em{E}val-2020)},
  year      = {2020},
  address   = {Barcelona, Spain},
  month     = {December},
  publisher = {Association for Computational Linguistics},
}

