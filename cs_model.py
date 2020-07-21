from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, Bidirectional, Embedding
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
import pickle
import numpy as np
import gensim
import keras.backend as K
from keras.utils import plot_model 
import matplotlib.pyplot as plt
import random
###############################
'''
Preparation of datasets and script where cs model trains
'''
#importing cs labels and tweets- This will be test set

cs_import = open("/home/francesita/SemEval2020/prepros_train.pkl", "rb")
cs_labels_import = open("/home/francesita/SemEval2020/labels_train.pkl", "rb")
cs_tweets = list(pickle.load(cs_import))
cs_labels = list(pickle.load(cs_labels_import))


# combine train tweets lists and train labels list
tweets = cs_tweets
labels = cs_labels

# randomize list of tweets and coreesponding labels for training
data = list(zip(tweets, labels)) #combine labels and tweets

random.shuffle(data)

tweets, labels = zip(*data)


#dividing data between training, test and development
train_size = int(0.8*(len(tweets)))
#dev_size = int(0.1(len(tweets)))
test_size = (len(tweets) - train_size)

# no of tweets can be param bc political tweets may be more verbose, also using no. of words as extra input to see if political or not
max_no_tokens = 12  
label_size = 3

indexes = set(np.random.choice(len(tweets), train_size + test_size, replace = False))

x_train = np.zeros((train_size, max_no_tokens), dtype=K.floatx())
y_train = np.zeros((train_size, label_size), dtype=np.int32)

x_test= np.zeros((test_size, max_no_tokens), dtype=K.floatx())
y_test= np.zeros((test_size, label_size), dtype=np.int32)


embed_model = gensim.models.Word2Vec.load("/home/francesita/SemEval2020/embedding_11_D100.model")
embedding_size = embed_model.wv.vector_size # I think this refers to the size of the word embeddings
embed_vocab_size = len(embed_model.wv.vocab) + 1 

#filling numpy arrays with encodings and labels for cs dataset when it is train and test

for i, index in enumerate(indexes):        
    if i < train_size:
        if int(labels[index]) == 1:
            y_train[i,:] = [1.,0.,0.]
        elif int(labels[index]) == 0:
            y_train[i,:]=[0.,1.,0.]
        else:
            y_train[i,:]=[0.,0.,1.]
    else: 
        if int(labels[index]) == 1:
            y_test[i-train_size,:] = [1.,0.,0.]
        elif int(labels[index]) == 0:
            y_test[i-train_size,:]=[0.,1.,0.]
        else:
            y_test[i-train_size,:]=[0.,0.,1.]

length_dic = len(embed_model.wv.index2word) 



for i_t, tweet in enumerate(tweets):
    twt=tweet
    for i_word in range(max_no_tokens):
        try:
            word = twt[i_word]
            if i_t < train_size:
                x_train[i_t,i_word] = embed_model.wv.index2word.index(word)
            else:
                x_test[i_t - train_size, i_word] = embed_model.wv.index2word.index(word)
        except:
             if i_t < train_size:
                x_train[i_t,i_word] = length_dic
             else:
                x_test[i_t - train_size, i_word] = length_dic
               


#preparing embeddings
# doing this so I get my +1 missing vocab. Cannot get it with just embed_model.wv.vectors
embedding_matrix = np.zeros((embed_vocab_size,embedding_size))
for i, vector in enumerate(embed_model.wv.vectors):
    embedding_matrix[i] = vector

global_dropout = 0.3
learning_rates = []
epochs = 25

model = Sequential()
model.add(Embedding(embed_vocab_size,embedding_size,weights=[embedding_matrix],input_length=max_no_tokens,trainable=True))
model.add(Bidirectional(LSTM(128, activation='relu', dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=False)))
model.add(Dropout( global_dropout ))

model.add(Dense(100, activation='relu'))
model.add(Dropout(global_dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout( global_dropout ))
model.add(Dense(label_size, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.0002), metrics=['accuracy', fbeta_score, recall, precision])

tensorboard= TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True)

#sklearn stuff
 
y_test_pred=model.predict_classes(x_test)
 
#print(precision_score(y_test,y_test_pred,average=None))

#print(recall_score(y_val,y_val_pred,average=None))
 
#print(classification_report(y_val, y_val_pred))
#### End sklearn stuff 
history = model.fit(x_train, y_train, batch_size=8, validation_data=(x_test, y_test), shuffle= False, epochs= epochs ,verbose=1,callbacks=[tensorboard, EarlyStopping(min_delta=0.0001, patience=5)])
model.summary()

y_pred = model.evaluate(x=x_test, y=y_test, batch_size=8, verbose=1)
#print(model.evaluate(x=x_train, y=y_train, batch_size=32, verbose=1))
print(model.evaluate(x=x_test, y=y_test, batch_size=8, verbose=1))
print(metrics.precisions)
model.save('semeval_model.h5')

