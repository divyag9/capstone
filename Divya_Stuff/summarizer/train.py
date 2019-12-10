import process_text
import seq2seq
import numpy as np
from sklearn.model_selection import train_test_split
from importlib import reload

config = process_text.fit_text("phone_reviews.csv")

summarizer = seq2seq.Seq2SeqSummarizer(config)

df = config['reviews_dataframe']

x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['text']),np.array(df['summary']),test_size=0.1,random_state=0,shuffle=True) 

summarizer.fit(x_tr, y_tr, x_val, y_val, epochs=20, model_dir_path='./models')