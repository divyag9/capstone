import pandas as pd
import boto3
import spacy
import nltk
!python -m spacy download en_core_web_sm
from nltk.tokenize.treebank import TreebankWordDetokenizer

class SentimentFilter(object):
    
    def __init__(self, text):
        self.text = text
        self.sent_df = None

    def getSentiment(self):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(self.text)
        sentence_tokens = [sents.text for sents in doc.sents]
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
        df = df.round(2)
        self.sent_df = df

    def filterNeutrals(self):
        """
        Filters out all neutral sentences.
        """
        self.sent_df = self.sent_df[self.sent_df['sentiment'] != 'NEUTRAL']
        self.sent_df.reset_index(inplace=True, drop=True)
        
        
    def filterPositives(self, threshold=0):
        """
        Filters out positive sentiment sentences with score below the threshold.
        """
        self.sent_df = self.sent_df[~((self.sent_df['sentiment'] == 'POSITIVE') &
                                      (self.sent_df['pos_prob'] < threshold))]
        self.sent_df.reset_index(inplace=True, drop=True)
    
    def filterNegatives(self, threshold=0):
        self.sent_df = self.sent_df[~((self.sent_df['sentiment'] == 'NEGATIVE') &
                                      (self.sent_df['neg_prob'] < threshold))]
        self.sent_df.reset_index(inplace=True, drop=True)
    
    def filterMixed(self, threshold=0):
        """
        Filters out mixed sentiment sentences with score below the threshold.
        """
        self.sent_df = self.sent_df[~((self.sent_df['sentiment'] == 'MIXED') &
                                      (self.sent_df['mix_prob'] < threshold))]
        self.sent_df.reset_index(inplace=True, drop=True)
    
    def getDataFrame(self):
        """
        Get the sentiment dataframe.
        """
        return self.sent_df
    
    def getFilteredTokens(self):
        """
        Get the sentence tokens from the text.
        """
        self.filtered_tokens = list(self.sent_df['sentence'])
        return self.filtered_tokens
    
    def getFilteredText(self):
        """
        Get the text that has had sentences filtered out.
        """
        self.filtered_text = TreebankWordDetokenizer().detokenize(list(self.sent_df['sentence']))
        return self.filtered_text
