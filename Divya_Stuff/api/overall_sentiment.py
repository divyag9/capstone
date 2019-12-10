import boto3
import spacy
spacy_nlp = spacy.load("en_core_web_sm")
import random

class OverallSentiment(object):
    """
    Simple object to return the overall sentiment of an object
    """
    def __init__(self, text):
        self.text = text
        self.sentences = set()
        
    def add_periods(self, text):
        """
        Takes in a string of text with no punctuation, uses Spacy's method of breaking up sentences to 
        add periods to the end of these sentences. Returns text with periods.
        """
        doc = spacy_nlp(text)
        sentence_tokens = [sents.text for sents in doc.sents]
        new_sentence_tokens = []
        add_on = None
        for i in range(len(sentence_tokens)-1,-1,-1):
            if " " in sentence_tokens[i]:
                if add_on == None:
                    # Add a period to the end of the sentence
                    new_sentence_tokens.append(sentence_tokens[i]+'.')
                else:
                    # Add 1 word and period to the end of the sentence.
                    new_sentence_tokens.append(sentence_tokens[i]+' '+add_on+'.')
                    add_on = None
            else:
                # If there is a sentence which is just one word, add it to the end of the previous sentence.
                add_on = sentence_tokens[i]
        new_text = " ".join(new_sentence_tokens[::-1])
        return new_text
        
    def getSentiment(self,faster = False):
        if ", " not in self.text:
            self.text = self.add_periods(self.text)
        else:
            pass
            
        doc = spacy_nlp(self.text)
        for token in doc:
            if token.pos_ == "ADJ":
                #print(" ".join(word.text for word in token.sent))
                self.sentences.add(" ".join(word.text for word in token.sent))
        if faster == True and len(self.sentences) > 100:
            randomsample = random.choices(list(self.sentences), k=100)
            output = self.calculateSentiment(randomsample)
        else:
            output = self.calculateSentiment(self.sentences)
        return output
    
    
    def sentenceSentiment(self,text):
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
        return sent, sent_pos, sent_neg, sent_neu, sent_mix

    def calculateSentiment(self,sentences):
        '''
        Take a list of sentences and give the overall sentiment
        '''
        numPos = 0
        numNeg = 0
        for sentence in sentences:
            sent = self.sentenceSentiment(sentence)
            if sent[0] == "POSITIVE":
                numPos+=1
            elif sent[0] == "NEGATIVE":
                numNeg+=1
        pctPos = 100*numPos/(numPos + numNeg)
        pctNeg = 100*numNeg/(numPos + numNeg)
        return (int(pctPos), int(pctNeg))