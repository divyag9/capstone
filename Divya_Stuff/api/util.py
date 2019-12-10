import re
import boto3
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def sentenceSentiment(text):
    """
    Takes in a line of text and uses amazon's comprehend to obtain the sentiment of the text.
    Returns the text and the score, as well as the sentiment of the text
    """
    comprehend = boto3.client(service_name='comprehend', region_name='us-west-2')
    sentiment_json = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    sent = sentiment_json['Sentiment']
    sent_pos = sentiment_json['SentimentScore']['Positive']
    sent_neg = sentiment_json['SentimentScore']['Negative']
    sent_neu = sentiment_json['SentimentScore']['Neutral']
    sent_mix = sentiment_json['SentimentScore']['Mixed']
    
    #return sent, sent_pos, sent_neg, sent_neu, sent_mix
    if sent == "POSITIVE":
        return (text,sent_pos), sent
    elif sent == "NEGATIVE":
        return (text,sent_neg), sent
    elif sent == "NEUTRAL":
        return (text, sent_neu), sent
    elif sent == "MIXED":
        return (text, sent_mix), sent
        
def orderSentiment(statements):
    """
    Takes in a list of statements and outputs a dictionary of lists of phrases ordered by score
    """
    positive = []
    negative = []
    neutral = []
    mixed = []
    for statement in statements:
        text, sentiment = sentenceSentiment(statement)
        if sentiment == "POSITIVE":
            positive.append(text)
        elif sentiment == "NEGATIVE":
            negative.append(text)
        elif sentiment == "NEUTRAL":
            neutral.append(text)
        elif sentiment == "MIXED":
            mixed.append(text)
            
    total_count = len(positive)+len(negative)+len(neutral)+len(mixed)
    pcnt_pos = int(100*len(positive)/total_count)
    pcnt_neg = int(100*len(negative)/total_count)
    pcnt_neu = int(100*len(neutral)/total_count)
    pcnt_mixed = int(100*len(mixed)/total_count)
    # Return a dict in the format of {pos:[list of positive phrases sorted by score], etc.}
    output = {"positive":[[positive[j][0] for j in np.argsort([i[1] for i in positive])][::-1],pcnt_pos],
              "negative":[[negative[j][0] for j in np.argsort([i[1] for i in negative])][::-1],pcnt_neg],
              "neutral":[[neutral[j][0] for j in np.argsort([i[1] for i in neutral])][::-1],pcnt_neu],
              "mixed":[[mixed[j][0] for j in np.argsort([i[1] for i in mixed])][::-1],pcnt_mixed]
             }
    return output


def text_cleaner(text,num):
    """
    Cleans the text for use in the summarizer supervised model
    """
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
    
    stop_words = set(stopwords.words('english'))
    newString = text.lower_
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