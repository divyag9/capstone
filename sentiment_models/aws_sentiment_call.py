import boto3
import json
from nltk.tokenize import sent_tokenize
import pandas as pd

def getSentiment(text):

    sentence_tokens = sent_tokenize(text)
    sent_list = []
    pos_list = []
    neg_list = []
    neu_list = []
    mix_list = []

    def sentenceSentiment(text):
        comprehend = boto3.client(service_name='comprehend', region_name='us-west-2')
        sentiment_json = comprehend.detect_sentiment(Text=text, LanguageCode='en')
        sent = sentiment_json['Sentiment']
        sent_pos = sentiment_json['SentimentScore']['Positive']
        sent_neg = sentiment_json['SentimentScore']['Negative']
        sent_neu = sentiment_json['SentimentScore']['Neutral']
        sent_mix = sentiment_json['SentimentScore']['Mixed']

        return sent, sent_pos, sent_neg, sent_neu, sent_mix

    for s in sentence_tokens:
        a, b, c, d, e = sentenceSentiment(s)
        sent_list.append(a)
        pos_list.append(b)
        neg_list.append(c)
        neu_list.append(d)
        mix_list.append(e)

    df = pd.DataFrame({'sentence': sentence_tokens, 'sentiment': sent_list, 'pos_prob': pos_list, 
                       'neg_prob': neg_list, 'neutral_prob':neu_list, 'mixed_prob': mix_list })

    return df
