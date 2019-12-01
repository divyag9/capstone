import nltk
import numpy as np
from itertools import groupby, chain
from collections import Counter, defaultdict
import spacy
spacy_nlp = spacy.load("en_core_web_sm")
import re
import boto3

class StatementExtracter(object):
    """
    As part of key word extraction and sentiment summarization, this code will be
    able to extract certain types of statements from a block of text.
    """
    
    def __init__(self, stopwords = None, punctuations = None):
        self.stopwords = stopwords
        self.punctuations = punctuations
        if self.stopwords == None:
            self.stopwords = []
        if self.punctuations == None:
            self.punctuations = list('!"#%&\'()*+,./:;<=>?@[\\]^_`{|}~♪')
        self.phrase_breaks = set(self.stopwords + self.punctuations)
        
    def replace_contraction(self, text):
        """
        Takes in text and replaces certain contractions
        """
        contraction_patterns = [(r'won\'t', 'will not'), (r'can\'t', 'can not'), (r'i\'m', 'i am'), 
                                (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                                (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'),
                                (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not'),
                                (r'they\'re', 'they are'), (r'They\'re', 'They are'), (r'it\'s', 'it is'), (r'It\'s', 'It is')]
        patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
        for (pattern, repl) in patterns:
            (text, count) = re.subn(pattern, repl, text)
        return text
    
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
        
    def spacy_POS_phrase_breaks(self, doc, POS_we_want, tag_we_want, dep_we_want):
        """
        Inputs a string of text, a list of POS, and a list of Spacy Tags that we want to keep in.
        This method parses the text and adds words that do not fall into POS and TAGS to phrase breaks list.
        """
        # Initialize the set with our existing phrase breaks
        temp_phrase_breaks = self.phrase_breaks.copy()

        for token in doc:
            if token.pos_ not in POS_we_want and token.tag_ not in tag_we_want and token.dep_ not in dep_we_want:
                temp_phrase_breaks.add(token.text.lower())
        return temp_phrase_breaks
    
    def visualize_POS(self, text, punctuation = False):
        """Visualize the POS of each word in the text"""
        if punctuation == False:
            text = self.add_periods(text)
            text = self.replace_contraction(text)
        else:
            text = self.replace_contraction(text)
        doc = spacy_nlp(text)
        for token in doc:
            print("{0}/{1}/{2} <--{3}-- {4}/{5}/{6}".format(
                    token.text,token.pos_,token.tag_,token.dep_,token.head.text,token.head.pos_,token.head.tag_))        
    
    def is_statements(self, text):
        # Situation where text contains sentences/punctuation
        if ", " in text:
            text = self.replace_contraction(text)
            statements = self.statementExtraction(text, withPunctuation = True)
            output = self.orderSentiment(statements)
            return output
        # Situation where text does not contain sentences/punctuation
        else:
            text = self.add_periods(text)
            text = self.replace_contraction(text)
            statements = self.statementExtraction(text, withPunctuation = False)
            output = self.orderSentiment(statements)
            return output
        
    def statementExtraction(self, text, withPunctuation = False):
        doc = spacy_nlp(text)
        statements = list()
        for token in doc:
            # If the token is "is" or "are", we want to look at the subtree
            if token.text in ['is','are']:
                # Pull the subtree as token objects and as pure text
                subtree = [i for i in token.subtree]

                # Split the subtree up into left and right groups
                left_subtree = [word for word in subtree if word.i < token.i]
                right_subtree = [word for word in subtree if word.i > token.i]

                if withPunctuation == True:
                    # Create a temporary set of break words based on the Part of Speech
                    left_POS = ['ADJ','DET','NOUN','NUM','PART','PROPN','PUNCT','ADP','PRON','AUX','SYM','SCONJ']
                    left_tags = []
                    left_dep = []
                    right_POS = ['ADJ','DET','NOUN','NUM','PART','PROPN','ADV','PUNCT','ADP','AUX','SYM','SCONJ','CCONJ','PRON']
                    right_tags = ['VBG','VBZ','VBP','VB','VBD','VBN']
                    right_dep = []
                if withPunctuation == False:
                    left_POS = ['NOUN']
                    left_tags = ['NNP','PRP','HYPH','JJ']
                    left_dep = ['advmod','amod','neg', 'nsubj']
                    right_POS = ['ADJ','DET','NOUN','NUM','PART','PROPN','ADV','ADP']
                    right_tags = ['VB','VBG','HYPH','VBN','UH']
                    right_dep = ['intj','prep','neg','pobj']
                left_phrase_breaks = self.spacy_POS_phrase_breaks(subtree, left_POS, left_tags, left_dep)
                right_phrase_breaks = self.spacy_POS_phrase_breaks(subtree, right_POS, right_tags, right_dep)

                # group words together using phrase breaks and a separator 
                left_phrase_groups = groupby(left_subtree, lambda word: word.text.lower() not in left_phrase_breaks)
                # Pull out the groups of words that do not include any of the phrase breaks   
                left_phrase_tuples = [tuple(group[1]) for group in left_phrase_groups if group[0] == True]

                # group words together using phrase breaks and a separator 
                right_phrase_groups = groupby(right_subtree, lambda word: word.text.lower() not in right_phrase_breaks)
                # Pull out the groups of words that do not include any of the phrase breaks   
                right_phrase_tuples = [tuple(group[1]) for group in right_phrase_groups if group[0] == True]

                subject = []
                description = []
                
                if withPunctuation == True:
                    # For the subjects on the left side of "is/are"
                    subject_word = None
                    for tuple_ in left_phrase_tuples:
                        for word in tuple_:
                            # if any of the words inside the tuple is a 'nsubj', it is potentially what we want
                            if word.dep_ in ['nsubj', 'expl']:
                                # check if we already have a good candidate for subject
                                if len(subject) < 1:
                                    subject_word = word
                                    subject = tuple_
                                    break
                                elif subject_word.pos_ in ['NOUN'] and word.pos_ not in ['NOUN']:
                                    pass
                                # if the existing candidate is not a noun, then we can update it.
                                else:
                                    subject_word = word
                                    subject = tuple_
                                    break
                    # Look at right text, if it includes adj, keep it + or just take the first phrase in the tuple.
                    if len(right_phrase_tuples) > 0:
                        description = right_phrase_tuples[0]
                if withPunctuation == False:
                    # For the subjects on the left side of "is/are"
                    subject_word = None
                    for tuple_ in left_phrase_tuples:
                        for word in tuple_:
                            # if any of the words inside the tuple is a 'nsubj', it is potentially what we want
                            if word.dep_ in ['nsubj', 'expl'] and word.pos_ in ['NOUN','DET','PRON']:
                                # check if we already have a good candidate for subject
                                if len(subject) < 1:
                                    subject_word = word
                                    subject = tuple_
                                    break
                                elif subject_word.pos_ in ['NOUN','DET'] and word.pos_ not in ['NOUN','DET']:
                                    pass
                                # if the existing candidate is not a noun, then we can update it.
                                else:
                                    subject_word = word
                                    subject = tuple_
                                    break
                    # Look at right text to get a description.
                    if len(right_phrase_tuples) > 0  and len(subtree) < 30:
                        for word in right_phrase_tuples[0]:
                            if word.pos_ == "ADJ":
                                description = right_phrase_tuples[0]

                    
                # Create the statements
#                 print(" ".join([word.text for word in subtree]))
#                 print([(word.text,word.pos_,word.tag_,word.dep_) for word in left_subtree])
#                 print([(word.text,word.pos_,word.tag_,word.dep_) for word in right_subtree])
#                 print('\n')
                #print(" ".join([word.text for word in subtree]))
                if len(subject) > 0 and len(description) > 0:
                    output = " ".join([word.text for word in subject]) + " " + token.text + " " + " ".join([word.text for word in description])
                    #print(output+"\n")
                    if output not in statements:
                        statements.append(output)
        return statements
    
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
        
        #return sent, sent_pos, sent_neg, sent_neu, sent_mix
        if sent == "POSITIVE":
            return (text,sent_pos), sent
        elif sent == "NEGATIVE":
            return (text,sent_neg), sent
        elif sent == "NEUTRAL":
            return (text, sent_neu), sent
        elif sent == "MIXED":
            return (text, sent_mix), sent
        
    def orderSentiment(self, statements):
        """
        Takes in a list of statements and outputs a dictionary of lists of phrases ordered by score
        """
        positive = []
        negative = []
        neutral = []
        mixed = []
        for statement in statements:
            text, sentiment = self.sentenceSentiment(statement)
            if sentiment == "POSITIVE":
                positive.append(text)
            elif sentiment == "NEGATIVE":
                negative.append(text)
            elif sentiment == "NEUTRAL":
                neutral.append(text)
            elif sentiment == "MIXED":
                mixed.append(text)
            
        # Return a dict in the format of {pos:[list of positive phrases sorted by score], etc.}
        output = {"pos":[positive[j][0] for j in np.argsort([i[1] for i in positive])[::-1]],
                  "neg":[negative[j][0] for j in np.argsort([i[1] for i in negative])[::-1]],
                  "neu":[neutral[j][0] for j in np.argsort([i[1] for i in neutral])[::-1]],
                  "mixed":[mixed[j][0] for j in np.argsort([i[1] for i in mixed])[::-1]]
                 }
        return output