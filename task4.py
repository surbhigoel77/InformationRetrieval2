import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from numpy import dot
from numpy.linalg import norm
import nltk
import csv
import statistics
from sklearn.metrics import f1_score, accuracy_score    

from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, Embedding, Conv2D


from gensim.models import Word2Vec

# set up multiprocessing
import logging
import multiprocessing
import os
import sys
cores = multiprocessing.cpu_count()

import xgboost as xgb
from sklearn.model_selection import train_test_split 

#Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

#Lemmitizer
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


from num2words import num2words

def lower_case(text):
    return text.lower()

def punctuation_removal(text):
    punctuations=['!','@','#','$','%','^','&','*','(',')','-','_','`','~','+','=','[',']','{','}','|',';',':','<','>','?','/',',','.','"','<<','>>']
    for character in text:
        if(character in punctuations) or (character in ['—','\n',"\\"]) :  #including em-dash, forward slash and enter seperately
            text = text.replace(character," ")
    return text 

def remove_apostrophe(text):
    text = str(np.char.replace(text, "'", " "))
    return text

def num_to_words(text):
    if text.isdigit()==True:
        text = num2words(text)
    else:
        text = text
    return text

def remove_URLs(text):
    text = ' '.join(word for word in text.split() if word[:4] not in('www:','http'))
    return text

def remove_short_words(text):
    text = ' '.join(word for word in text.split() if len(word)>2)
    return text

def remove_long_words(text):
    text = ' '.join(word for word in text.split() if len(word)<15)
    return text

def remove_white_space(text):
    text = text.strip()
    return text


def lemmetizing(text):
    lemmatiser = WordNetLemmatizer()
    lemmetized_word = lemmatiser.lemmatize(text)
    #lemmetized_word = Lemmatizer.lemmatize
    return lemmetized_word

def stemming(term):
    suffixes = ['ed', 'ing' , 's','es']#,'ers', 'ion', 'ize', 'ise', 'ive', 'en', 'ly', 'ish', 'ian','ese']
    for suffix in suffixes:
        if term.endswith(suffix):
            term =  term[:-len(suffix)]
        else:
            term = term
    return term

def stop_words_removal(vocabulary): 
    FilteredVocabulary = []
    for term in vocabulary:
        if term not in stops:
            FilteredVocabulary.append(term)
    return FilteredVocabulary

def preprocessing(text):
    text = lower_case(text)
    text = punctuation_removal(text)
    text = remove_apostrophe(text)
    text = remove_URLs(text)
    # text = remove_short_words(text) 
    # text = remove_long_words(text)
    text = remove_white_space(text)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = stop_words_removal(tokens)
    return tokens

start = time.time()

validation_file = "validation_data.tsv"
valid_data = pd.read_csv(validation_file, sep='\t')

train_file = "train_data.tsv"
train_data = pd.read_csv(train_file, sep='\t')

st = time.time()
valid_data['passage'] = valid_data['passage'].apply(preprocessing)
print('Time taken to preprocess passage: ', time.time() - st)
st = time.time()
valid_data['queries'] = valid_data['queries'].apply(preprocessing)
print('Time taken to preprocess queries: ', time.time() - st)

def negative_sampling(df, postiveSamples, negativeSamples):
    seed = 10
    # Get a list of unique qid values
    qid_list = np.unique(df['qid'])
    samples_list = []
    for qid in qid_list:
        # Get rows with relevancy=1 and qid=current qid
        pos_rows = df[(df['qid'] == qid) & (df['relevancy'] == 1)]
        # Get rows with relevancy=0 and qid=current qid
        neg_rows = df[(df['qid'] == qid) & (df['relevancy'] == 0)]
        if len(pos_rows) < postiveSamples:
            samples_list.append(pos_rows)
        else:
            samples_list.append(pos_rows.sample(n=postiveSamples, random_state=np.random.RandomState(seed)))
        # # Append a single randomly selected positive row to samples
        # samples_list.append(pos_temp.sample(n=1, random_state=np.random.RandomState(seed)))
        # If there are less than k negative rows, append all of them to samples
        if len(neg_rows) < negativeSamples:
            samples_list.append(neg_rows)
        # Otherwise, append k randomly selected negative rows to samples
        else:
            samples_list.append(neg_rows.sample(n=negativeSamples, random_state=np.random.RandomState(seed)))
    # Concatenate all samples into a single dataframe and reset the index
    new_data = pd.concat(samples_list)
    return new_data.reset_index(drop=True)

st = time.time()
train_data_sampled = negative_sampling(train_data, 10, 20)
train_data_sampled = train_data_sampled.reset_index(drop=True)
print("Time taken to sample training data: ", time.time() - st)

print("Size of sampled data: ", len(train_data_sampled))

st = time.time()
train_data_sampled['passage'] = train_data_sampled['passage'].apply(preprocessing)
print('Time taken to preprocess passage: ', time.time() - st)

st = time.time()
train_data_sampled['queries'] = train_data_sampled['queries'].apply(preprocessing)
print('Time taken to preprocess queries: ', time.time() - st)

st = time.time()
# train Word2Vec model on train queries and passages
model = Word2Vec(sentences=train_data_sampled['queries'].append(train_data_sampled['passage']), min_count=1)

print('Time taken to calculate inverted index dict: ', time.time() - st)

def average_vector(vector_list):
    sum_vec = np.add.reduce(vector_list)
    return sum_vec/len(vector_list)

def cosine_similarity(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return 0
    return dot(a, b)/(norm(a)*norm(b))

def verify_data_correctness(df):
    return df.isnull().values.any()

st = time.time()
train_data_sampled['queries_vv'] = train_data_sampled['queries'].apply(lambda x: [model.wv[word] for word in x if word in model.wv.key_to_index ]) 
train_data_sampled['queries_vector_avg'] = train_data_sampled['queries_vv'].apply(lambda x: average_vector(x))

train_data_sampled['passage_vv'] = train_data_sampled['passage'].apply(lambda x: [model.wv[word] for word in x if word in model.wv.key_to_index ]) 
train_data_sampled['passage_vector_avg'] = train_data_sampled['passage_vv'].apply(lambda x: average_vector(x))
print("Time taken to calculate vector embedding for train data: ", time.time()-st)

st = time.time()
train_data_sampled['cosine_similarity'] = train_data_sampled.apply(lambda row: cosine_similarity(row['queries_vector_avg'], row['passage_vector_avg']), axis=1)
print("Time taken to calculate cosine similarity: ", time.time() - st)

st = time.time()
train_data_sampled['doc_len'] = train_data_sampled['passage'].apply(lambda x: float(len(x)))
train_data_sampled['query_len'] = train_data_sampled['queries'].apply(lambda x: float(len(x)))
print("Time taken to calculate len features: ", time.time() - st)

verify_data_correctness(train_data_sampled)

print(train_data_sampled.head())

st = time.time()
valid_data['queries_vv'] = valid_data['queries'].apply(lambda x: [model.wv[word] for word in x if word in model.wv.key_to_index ])
valid_data['queries_vector_avg'] = valid_data['queries_vv'].apply(lambda x: average_vector(x))
valid_data['queries_vector_avg'] = valid_data['queries_vector_avg'].fillna(0)
print("Time taken to calculate vector embedding for queries in validation data: ", time.time()-st)

st = time.time()
valid_data['passage_vv'] = valid_data['passage'].apply(lambda x: [model.wv[word] for word in x if word in model.wv.key_to_index])
valid_data['passage_vector_avg'] = valid_data['passage_vv'].apply(lambda x: average_vector(x) )
print("Time taken to calculate vector embedding for passages in validation data: ", time.time()-st)

st = time.time()
valid_data['cosine_similarity'] = valid_data.apply(lambda row: cosine_similarity(row['queries_vector_avg'], row['passage_vector_avg']), axis=1)
print("Time taken to calculate cosine similarity: ", time.time() - st)

st = time.time()
valid_data['doc_len'] = valid_data['passage'].apply(lambda x: float(len(x)))
valid_data['query_len'] = valid_data['queries'].apply(lambda x: float(len(x)))
print("Time taken to len features: ", time.time() - st)

print(valid_data.head())

verify_data_correctness(valid_data)

st = time.time()

X_train = train_data_sampled[['cosine_similarity','doc_len', 'query_len']]
y_train = train_data_sampled['relevancy']
# Assume `X_train` and `y_train` are the preprocessed and tokenized training data

# Define model hyperparameters
max_len = 1000  # Maximum length of a passage
num_filters = 64  # Number of filters in each convolutional layer
filter_sizes = [3, 4, 5]  # Filter sizes for the convolutional layers
dropout_rate = 0.5  # Dropout rate for regularization
num_classes = 2  # Number of classes (relevant or not relevant)

# Define the model architecture
# We are not using an embeddign layer here, because we have already performed embedding in the earlier section.
#model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
# Define the model architecture
tfmodel = Sequential()
tfmodel.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
tfmodel.add(Dropout(0.2))
tfmodel.add(Dense(64, activation='relu'))
tfmodel.add(Dropout(0.2))
tfmodel.add(Dense(32, activation='relu'))
tfmodel.add(Dense(1, activation='sigmoid'))

# Compile the model
tfmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision()])
tfmodel.summary()

# Train the model
tfmodel.fit(X_train.values, y_train.values, batch_size=32, epochs=10, validation_split=0.2)
tfmodel.evaluate(X_train.values, y_train.values)

def average_precision(sorted_df):    
    num_relevant = 0
    precisions = []
    ap = 0
    for i in range(len(sorted_df)):
        if sorted_df.relevancy[i] == 1:
            num_relevant += 1  #No. of documents that are relevant
            precision = num_relevant/ (i + 1)  #No. of relevant docs divided by total number of documents retrieved
            precisions.append(precision)
    if num_relevant==0:
        return 0
    else:
        return  (sum(precisions)/num_relevant)
 
    
def norm_disc_cum_gain(dcg_df, idcg_df):
    dcg = 0
    idcg = 0
    
    for i,irow in idcg_df.iterrows():
        rel = irow.relevancy
        idcg += (2**rel - 1) / np.log2(i + 2)
    for j,jrow in dcg_df.iterrows():
        rel = jrow.relevancy
        dcg += (2**rel - 1) / np.log2(i + 2)
    
    if idcg > 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0
        
    return ndcg

st = time.time()
querylist = set(list(valid_data.qid))

valid_data['relevancy_prediction'] = tfmodel.predict(valid_data[['cosine_similarity','doc_len', 'query_len']])
valid_data['prediction_rank'] = valid_data.groupby('qid')['relevancy_prediction'].rank(method='first',ascending=False)

MAP_querylist=[]
NDCG_querylist=[]

for q in querylist:
    sorted_df = valid_data[(valid_data.qid==q)].sort_values(by=['prediction_rank'],ascending=False)[:100]   # Extract top k passages for each query based on predicted relevancy
    sorted_df = sorted_df.reset_index(drop=True, inplace=False)
    average_prec = average_precision(sorted_df)     
    MAP_querylist.append(average_prec)

    # Compute ndcg
    dcg_df = sorted_df
    idcg_df = valid_data[(valid_data.qid==q)].sort_values(by=['relevancy'],ascending=False)[:100]
    ndcg = norm_disc_cum_gain(dcg_df,idcg_df)
    NDCG_querylist.append(ndcg)

print("Taken taken: ", time.time() -st)
Map_final = sum(MAP_querylist)/len(querylist)
print("Final MAP: ", Map_final)
Ndcg_final = sum(NDCG_querylist)/len(querylist)
print("Final NDCG: ", Ndcg_final)

# Calculate score for candidate passage
test_file = "candidate_passages_top1000.tsv"
test_data = pd.read_csv(test_file, sep='\t', names=['qid','pid','queries','passage'])

st = time.time()
test_data['passage'] = test_data['passage'].apply(preprocessing)
print('Time taken to preprocess passage: ', time.time() - st)

st = time.time()
test_data['queries'] = test_data['queries'].apply(preprocessing)
print('Time taken to preprocess queries: ', time.time() - st)

st = time.time()
test_data['queries_vv'] = test_data['queries'].apply(lambda x: [model.wv[word] for word in x if word in model.wv.key_to_index ]) 
test_data['queries_vector_avg'] = test_data['queries_vv'].apply(lambda x: average_vector(x))
test_data['queries_vector_avg'] = test_data['queries_vector_avg'].fillna(0)

test_data['passage_vv'] = test_data['passage'].apply(lambda x: [model.wv[word] for word in x if word in model.wv.key_to_index ]) 
test_data['passage_vector_avg'] = test_data['passage_vv'].apply(lambda x: average_vector(x))
test_data['passage_vector_avg'] = test_data['passage_vector_avg'].fillna(0)
print("Time taken to calculate vector embedding for test data: ", time.time()-st)

st = time.time()
test_data['cosine_similarity'] = test_data.apply(lambda row: cosine_similarity(row['queries_vector_avg'], row['passage_vector_avg']), axis=1)
print("Time taken to calculate cosine similarity: ", time.time() - st)

st = time.time()
test_data['doc_len'] = test_data['passage'].apply(lambda x: float(len(x)))
test_data['query_len'] = test_data['queries'].apply(lambda x: float(len(x)))
print("Time taken to len features: ", time.time() - st)

verify_data_correctness(test_data)

X_test = np.array(test_data[['cosine_similarity','doc_len', 'query_len']]).reshape(-1,3)

tensorflow_predictions = tfmodel.predict(X_test)
test_data['NN_score'] = tensorflow_predictions
test_data['NN_Rank'] = test_data.groupby('qid')['NN_score'].rank(method='first',ascending=False)

test_data.sort_values(by=['qid','NN_Rank'])

def write_results(model, df):
    querylist = test_data.qid.unique()
    filename = '%s.txt' % model
    col = '%s_Rank' % model
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for q in querylist:
            sorted_df = df[(df.qid==q)].sort_values(by=[col],ascending=True)
            if len(sorted_df)>=100:
                limit=100
            else:
                limit=len(sorted_df)
            for i, row in sorted_df.iloc[:limit].iterrows():
                writer.writerow((int(row.qid),'A2',int(row.pid),row[col],round(row.NN_score,8),'NN'))

write_results('NN', test_data)
print("Total time taken in task4 : ", time.time() - start)