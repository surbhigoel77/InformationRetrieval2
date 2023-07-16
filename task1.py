import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

import nltk

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

def word_counter(vocabulary):
    TermCount = {}  
    for term in vocabulary:
        if (term in TermCount):  #Count term frequency for each term in one_grams list 
            TermCount[term] = TermCount[term]+1
        else:
            TermCount[term]=1
    return TermCount


valid_data = pd.read_csv("validation_data.tsv", sep='\t')

print("Validation Data size: ", len(valid_data))

st = time.time()

valid_data_preprocessed = valid_data
valid_data_preprocessed['queries'] = valid_data_preprocessed['queries'].apply(preprocessing)
valid_data_preprocessed['passage'] = valid_data_preprocessed['passage'].apply(preprocessing)

print('Time taken to calculate inverted index dict: ', time.time() - st)

passage_data = valid_data_preprocessed[['pid','passage']]
passage_data= passage_data.drop_duplicates(subset=['pid'])
print(passage_data.head())




st2 = time.time()
inverted_index_dict = {}

for idx, row in passage_data.iterrows():
    passage_tokens = row['passage']
    passage_token_freq = nltk.FreqDist(passage_tokens)
    passage_length = len(passage_tokens)
    for word, freq in passage_token_freq.items():
        if word not in inverted_index_dict:
            inverted_index_dict[word] = [(int(row['pid']), freq, passage_length)] 
        else:
            inverted_index_dict[word].append((int(row['pid']), freq, passage_length))
print('Time taken to calculate inverted index dict: ', time.time() - st2)

N = len(passage_data)
words = 0
passage_data = passage_data.reset_index()
for i in range(len(passage_data)):
    words += len(passage_data.passage[i])

avg_doc_len = words/N
print("Avg Doc Len: ", avg_doc_len)

R = 0
r = 0
k1 = 1.2
k2 = 100
b = 0.75

def bm25_score(query, passage):
    passage_freq_distribution = nltk.FreqDist(passage)
    query_freq_distribution = nltk.FreqDist(query)
    len_doc = len(passage)
    K = k1*((1-b) + b *(float(len_doc)/float(avg_doc_len)))
    
    score = 0
    for token in query:
        try:
            n = len(inverted_index_dict[token])
        except:
            n = 0
        f = passage_freq_distribution[token]
        qf = query_freq_distribution[token]
        inter = np.log(((r + 0.5)/(R - r + 0.5))/((n-r+0.5)/(N-n-R+r+0.5)))
        score += inter * ((k1 + 1) * f)/(K+f) * ((k2+1) * qf)/(k2+qf)
    return score

st3 = time.time()
bm25=[]
for idx, row in valid_data_preprocessed.iterrows():
    passage = row['passage']
    query = row['queries']
    bm25.append(bm25_score(query,passage))
print('Time taken to calculate BM25 scores: ', time.time() - st3)

valid_data_preprocessed['BM25_score'] = bm25

print(valid_data_preprocessed.head())

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
 
    
def norm_disc_cum_gain(dcg_df, idcg_df, k):
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

querylist = set(list(valid_data_preprocessed.qid))

st4 = time.time()
res = []
for k in [3,10,100]:
    MAP=[]
    NDCG=[]
    
    for q in querylist:
        # Compute average Precision
        sorted_df = valid_data_preprocessed[(valid_data_preprocessed.qid==q)].sort_values(by=['BM25_score'],ascending=False)[:k]   # Extract top k passages for each query based on bm25 score
        sorted_df = sorted_df.reset_index(drop=True, inplace=False)
       
        average_prec = average_precision(sorted_df)    # Calculate precision value for each query-passage combination 
        MAP.append(average_prec)
        
        # Compute ndcg
        dcg_df = sorted_df
        idcg_df = valid_data[(valid_data_preprocessed.qid==q)].sort_values(by=['relevancy'],ascending=False)[:k]
        
        ndcg = norm_disc_cum_gain(dcg_df,idcg_df,k)
        NDCG.append(ndcg)
    print("Time Taken: ", time.time()-st4)
    Map_mean = sum(MAP)/len(querylist)
    Ndcg_mean = sum(NDCG)/len(querylist)
    res.append({'k':k,'Average Precision':Map_mean,'Average NDCG':Ndcg_mean})
    print("For K=",k, " Average Precision is: ", Map_mean, " and Average Normalized Discounted Cumulative Gain(NDCG) is: ",Ndcg_mean)

result = pd.DataFrame(res)
print(result)

print("Total time taken in task1: ", time.time()-st)