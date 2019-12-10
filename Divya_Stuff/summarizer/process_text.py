import re
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

MAX_INPUT_SEQ_LENGTH = 100
MAX_TARGET_SEQ_LENGTH = 20

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

def text_cleaner(text,num):
    stop_words = set(stopwords.words('english'))
    newString = text.lower()
    newString = BeautifulSoup(newString, "html.parser").text ##ME - had to change from lxml to html.parser
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z0-9]", " ", newString) 
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:                                                 
            long_words.append(i)   
    return (" ".join(long_words)).strip()

def fit_text(csv_file, max_text_len=None, max_summary_len=None):
    if max_text_len is None:
        max_text_len = MAX_INPUT_SEQ_LENGTH
    if max_summary_len is None:
        max_summary_len = MAX_TARGET_SEQ_LENGTH
        
    data=pd.read_csv(csv_file)
    data.drop_duplicates(subset=['Text'],inplace=True)
    data.dropna(inplace=True)
    data = data[data['Summary'].apply(lambda x: not(len(x.lower().split(' ')) <= 2 and ('star' in x.lower().split(' ') or 'stars' in x.lower().split(' ')) ))]
    data = data[~data.Text.str.match('<.*>')]
     
    cleaned_text = []
    for t in data['Text']:
        cleaned_text.append(text_cleaner(t,0)) 
    
    cleaned_summary = []
    for t in data['Summary']:
        cleaned_summary.append(text_cleaner(t,1))

    data['cleaned_text']=cleaned_text
    data['cleaned_summary']=cleaned_summary
    
    data.replace('', np.nan, inplace=True)
    data.dropna(axis=0,inplace=True)
    data = data[data['cleaned_text'].apply(lambda x: len(x.split(' ')) > 3)]
    
    cleaned_text =np.array(data['cleaned_text'])
    cleaned_summary=np.array(data['cleaned_summary'])

    short_text=[]
    short_summary=[]

    for i in range(len(cleaned_text)):
        if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])

    df=pd.DataFrame({'text':short_text,'summary':short_summary})
    df['summary'] = df['summary'].apply(lambda x : 'sostok '+ x + ' eostok')
    
    source_tokenizer = Tokenizer() 
    source_tokenizer.fit_on_texts(list(df['text']))
    
    
    reverse_source_word_index=source_tokenizer.index_word
    source_word_index = source_tokenizer.word_index
    
    target_tokenizer = Tokenizer()   
    target_tokenizer.fit_on_texts(list(df['summary']))
    
    #size of vocabulary
    reverse_target_word_index=target_tokenizer.index_word
    target_word_index=target_tokenizer.word_index

    config = dict()
    config['source_tokenizer'] = source_tokenizer
    config['target_tokenizer'] = target_tokenizer
    config['reverse_target_word_index'] = reverse_target_word_index
    config['reverse_target_word_index'] = reverse_target_word_index
    config['reverse_source_word_index'] = reverse_source_word_index
    config['target_word_index'] = target_word_index
    config['source_word_index'] = source_word_index
    config['max_text_len'] = max_text_len
    config['max_summary_len'] = max_summary_len
    config['reviews_dataframe'] = df

    return config