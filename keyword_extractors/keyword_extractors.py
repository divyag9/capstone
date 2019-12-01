import pandas as pd
import numpy as np
from itertools import groupby, chain
from collections import Counter, defaultdict
import boto3
import spacy
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pke

#!python -m nltk.downloader stopwords
#!python -m nltk.downloader universal_tagset
#!python -m spacy download en
#!python -m spacy download en_core_web_sm

spacy_nlp = spacy.load("en_core_web_sm")

class SentimentFilter(object):
    
    def __init__(self, text):
        self.text = text
        self.sent_df = None

    def getSentiment(self):
        """Get sentiment for each sentence in the text"""
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


class myRake(object):
    """Rapid Automatic Keyword Extraction Algorithm customized for 
    key-word extraction on video text w/ or w/o punctuation.
    
    RAKE algorithm based off of implementation from rake-nltk by Vishwas B Sharma
    https://github.com/csurfer/rake-nltk with changes to suit personal needs.
    """
    
    def __init__(self, stopwords = None, punctuations = None, num_words = 100000,
                 use_POS = True, known_words = []):
        # Initialize the stopwords and punctuations used to break text into phrases
        self.stopwords = stopwords
        self.punctuations = punctuations
        if self.stopwords == None:
            self.stopwords =  nltk.corpus.stopwords.words('english')
        if self.punctuations == None:
            self.punctuations = list('!"#$%&\'()*+,./:;<=>?@[-\\]^_`{|}~♪')
        # This is the set of words that determines breaks between phrases
        self.phrase_breaks = set(self.stopwords + self.punctuations)
        
        # This variable determines how many words long our key-words can be
        self.num_words = num_words
        
        # This variable lets us know if we want to use regular stopwords, or incorporate POS
        self.use_POS = use_POS
        # This variable stores a list of words that we want to have more impact in terms of score
        self.known_words = known_words
        
        # Variables to calcuate RAKE score
        self.frequencies = None
        self.degrees = None
        self.key_words = None
        
    def extract_keywords(self, text):
        # Situation where text contains sentences/punctuation
        if ", " in text:
            text_list = nltk.tokenize.sent_tokenize(text)
            phrase_tuples = self.key_word_candidates(text_list)
            self.RAKE_score(phrase_tuples)
            
        # Situation where text does not contain sentences/punctuation
        else:
            text_list = nltk.tokenize.sent_tokenize(text)
            phrase_tuples = self.key_word_candidates(text_list)
            self.RAKE_score(phrase_tuples)
            # TO DO: add some sort of method to split the text up into multiple sentences
            # Convert string to list of words. After x number of words, if the word and next word do not fall in
            # ['ADJ','DET','NOUN','NUM','PART','PROPN'] category, then add a . Then convert back to string
        
    def spacy_POS_phrase_breaks(self, text):
        """
        Inputs a string of text, find the Part of Speech for each word and add words that are not
        ['ADJ','DET','NOUN','NUM','PART','PROPN'] into a set of phrase break words to ignore.
        """
        # These are POS tags that we want in our keywords.
        # Try removing ADJ, DET 
        POS_we_want = ['ADJ','DET','NOUN','NUM','PART','PROPN']
        # Initialize the set with our existing phrase breaks
        temp_phrase_breaks = self.phrase_breaks
        
        # Use spacy to tag POS and then only keep words with the POS that we want
        doc = spacy_nlp(text)
        for token in doc:
            if token.pos_ not in POS_we_want:
                temp_phrase_breaks.add(token.text.lower())
        return temp_phrase_breaks
                
        
    def key_word_candidates(self, text_list):
        """
        Input a list of text segments and generates a set of possible key-word candidates.
        """
        candidates = set()
        for text in text_list:
            # Extract all words and punctuation from text into a list
            words = [word.lower() for word in nltk.wordpunct_tokenize(text)]
            
            if self.use_POS:
                # Create a temporary set of break words based on the Part of Speech
                temp_phrase_breaks = self.spacy_POS_phrase_breaks(text)
                # group words together using phrase breaks and a separator 
                phrase_groups = groupby(words, lambda word: word not in temp_phrase_breaks)
                
            else:
                # if we don't want to use POS, just use the stopwords + punct to break phrases
                phrase_groups = groupby(words, lambda word: word not in self.phrase_breaks)
                
            # Pull out the groups of words that do not include any of the phrase breaks   
            phrase_tuples = [tuple(group[1]) for group in phrase_groups if group[0] == True]
            # Add these groups to the output set
            candidates.update(phrase_tuples)
        # make sure the number of words in each of the tuples does not go over our limit
        return set(filter(lambda x: len(x) <= self.num_words, candidates))
        
    def RAKE_score(self, phrase_tuples):
        """
        Frequency part: chain up the phrase tuples and use the counter to tally up how often each word occurs.
                        Saves a dictionary of word:count pairs in self.frequencies
        Degree part: create a default dict to keep track of how many words each word co-occurs with in 
                     the phrase tuples. There is another way that keeps track of a co-occurence graph which
                     might be useful but I didn't implement for the sake of simplicity.
        Scoring part: Calculate the RAKE score for each phrase. The RAKE score for each  word is degree/frequency
                      and the RAKE score for each phrase is the sum of each word's RAKE score.
        """
        # Frequency part
        self.frequencies = Counter(chain.from_iterable(phrase_tuples))
        
        # Degree part
        self.degrees = defaultdict(int)
        for phrase in phrase_tuples:
            for word in phrase:
                self.degrees[word] += len(phrase)
        
        # Scoring part
        self.key_words = defaultdict(float)
        phrases = list()
        scores = list()
        for phrase in phrase_tuples:
            score = 0.0
            for word in phrase:
                score += float(self.degrees[word])/float(self.frequencies[word])
                # This is to give words that we know should be keywords a boost in score
                if word in self.known_words:
                    score += 10
            phrases.append(" ".join(phrase))
            scores.append(score)
        phrases = np.array(phrases)
        scores = np.array(scores)
        # Store the phrase:score pairs in descending order into self.key_words
        for i in np.argsort(scores)[::-1]:
            self.key_words[phrases[i]] = scores[i]
    
    def get_key_words(self, n = None):
        """
        get command to return a list of keywords ordered by their RAKE score
        n is the number of words to output
        """
        if n == None:
            return list(self.key_words.keys())
        else:
            return list(self.key_words.keys())[:n]
    
    def get_key_words_scores(self):
        """
        get command to return a list of keywords and their RAKE scores
        """
        return [(key,self.key_words[key]) for key in self.key_words]


class KeyWordExtractor(object):
    def __init__(self, text):
        self.text = text
        self.sent_df = None
        self.kw_tfidf = []
        self.kw_kpminer = []
        self.kw_yake = []
        self.kw_rake = []
        self.kw_textrank = []
        self.kw_singlerank = []
        self.kw_topicrank = []
        self.kw_tprank = []
        self.kw_positionrank = []
        self.kw_mprank = []
        
    def tfidf(self, n=20):
        try:
            extractor = pke.unsupervised.TfIdf()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_tfidf = extractor.get_n_best(n=n)
        except:
            print('Failed to extract keywords using KP-Miner')       
                  
    def kpMiner(self, n=20):
        try:
            extractor = pke.unsupervised.KPMiner()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_kpminer = extractor.get_n_best(n=n)
        except:
            print('Failed to extract keywords using KP-Miner')
        
    def yake(self, n=20):
        try:
            extractor = pke.unsupervised.YAKE()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_yake = extractor.get_n_best(n=n)
        except:
            print('Failed to extract keywords using YAKE')
    
    def rake(self, n=20):
        try:
            extractor = myRake(use_POS=True)
            extractor.extract_keywords(self.text)
            self.kw_rake = extractor.get_key_words_scores()[:n]
        except:
            print('Failed to extract keywords using KP-Miner')

    def textRank(self, n=20):
        try:
            extractor = pke.unsupervised.TextRank()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_textrank = extractor.get_n_best(n=n)
        except:
            print('Failed to extract keywords using TextRank')
                  
    def singleRank(self, n=20):
        try:
            extractor = pke.unsupervised.SingleRank()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_singlerank = extractor.get_n_best(n=n)
        except:
            print('Failed to extract keywords using SingleRank')
       
    def topicRank(self, n=20):
        try:
            extractor = pke.unsupervised.TopicRank()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_topicrank = extractor.get_n_best(n=n)
        except:
            print('Failed to extract keywords using TopicRank')
    
    def topicalPageRank(self, n=20):
        try:
            extractor = pke.unsupervised.TopicalPageRank()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_tprank = extractor.get_n_best(n=n)
        except:
            print('Failed to extract keywords using Topical PageRank')
        
    def positionRank(self, n=20):
        try:
            extractor = pke.unsupervised.PositionRank()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_positionrank = extractor.get_n_best(n=n)
        except:
            print('Failed to exract keywords using PositionRank')

    def multiPartiteRank(self, n=20):
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(self.text, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            self.kw_mprank = extractor.get_n_best(n=n)
        except:
            print('Failed to extract keywords using Multi-Partite Rank')


    def allExtractors(self, n=20, pos=None, window=10, normalized=False):
        self.tfidf(n=n)
        self.kpMiner(n=n)
        self.yake(n=n)
        self.rake(n=n)
        self.textRank(n=n)
        self.singleRank(n=n)
        self.topicRank(n=n)
        self.topicalPageRank(n=n)
        self.positionRank(n=n)
        self.multiPartiteRank(n=n)
        
    def getAllKeyWords(self, include_score=False, sort=False):
        """
        Get all keywords for all models
        """
        if include_score == True:
            all_keywords = {'tfidf_keywords': self.kw_tfidf, 
                            'kpminer_keywords': self.kw_kpminer, 
                            'yake_keywords': self.kw_yake,
                            'rake_keywords': self.kw_rake, 
                            'textrank_keywords': self.kw_textrank,
                            'singlerank_keywords':self.kw_singlerank,
                            'topicrank_keywords': self.kw_topicrank, 
                            'topicalpagerank_keywords': self.kw_tprank, 
                            'position_keywords': self.kw_positionrank,
                            'multipartiterank_keywords': self.kw_mprank}
        else:
            if sort == False:
                all_keywords = {'tfidf_keywords': [i[0] for i in self.kw_tfidf],
                                'kpminer_keywords': [i[0] for i in self.kw_kpminer], 
                                'yake_keywords': [i[0] for i in self.kw_yake],
                                'rake_keywords': [i[0] for i in self.kw_rake],
                                'textrank_keywords': [i[0] for i in self.kw_textrank],
                                'singlerank_keywords': [i[0] for i in self.kw_singlerank],
                                'topicrank_keywords': [i[0] for i in self.kw_topicrank], 
                                'topicalpagerank_keywords': [i[0] for i in self.kw_tprank],
                                'position_keywords': [i[0] for i in self.kw_positionrank],
                                'multipartiterank_keywords': [i[0] for i in self.kw_mprank]}
            else:
                all_keywords = {'tfidf_keywords': sorted([i[0] for i in self.kw_tfidf]),
                                'kpminer_keywords': sorted([i[0] for i in self.kw_kpminer]), 
                                'yake_keywords': sorted([i[0] for i in self.kw_yake]),
                                'rake_keywords': sorted([i[0] for i in self.kw_rake]),
                                'textrank_keywords': sorted([i[0] for i in self.kw_textrank]),
                                'singlerank_keywords': sorted([i[0] for i in self.kw_singlerank]),
                                'topicrank_keywords': sorted([i[0] for i in self.kw_topicrank]), 
                                'topicalpagerank_keywords': sorted([i[0] for i in self.kw_tprank]),
                                'position_keywords': sorted([i[0] for i in self.kw_positionrank]),
                                'multipartiterank_keywords': sorted([i[0] for i in self.kw_mprank])}

        return all_keywords

class keywordFilter(object):
    
    def __init__(self, keywords, sentiment_df):
        self.keywords = keywords
        self.sentiment_df = sentiment_df

        # Get the sentences tied to all the keywords
        kw_dict = {}
        for kw in self.keywords:
            kw_dict[kw] = list(self.sentiment_df[self.sentiment_df['sentence']
                                                     .str.lower()
                                                     .str.contains(kw)]['sentence'])

        # Create dataframe out of the dictionary
        keyword_df = pd.DataFrame.from_dict(kw_dict, orient='index')
        keyword_df = keyword_df.stack().to_frame('sentence').reset_index()
        keyword_df.drop('level_1', axis=1, inplace=True)
        keyword_df.columns = ['keyword', 'sentence']

        # join in the sentiment for each sentence
        keyword_df = keyword_df.set_index('sentence').join(sentiment_df.set_index('sentence'))
        keyword_df.reset_index(inplace=True)

        # filter down to the necessary columns
        keyword_df = keyword_df[['keyword', 'sentence', 'sentiment']]
        self.keyword_df = keyword_df
    
    def sentenceCountFilter(self, n=1):
        count_df = self.keyword_df['keyword'].value_counts().to_frame('sentence_count')
        count_df = count_df[count_df['sentence_count'] <= n] # Put variable here for filtering
        count_df.reset_index(inplace=True)
        self.keyword_df = self.keyword_df[self.keyword_df['keyword'].isin(count_df['index'])].copy()
        self.keyword_df.reset_index(inplace=True, drop=True)
        
    def duplicateFilter(self):
        self.keyword_df['keyword_num'] = self.keyword_df.groupby('sentence')['keyword']\
                                                        .expanding()\
                                                        .count()\
                                                        .to_frame()\
                                                        .reset_index()['keyword']

        self.keyword_df = self.keyword_df[self.keyword_df['keyword_num'] == 1].copy()
        self.keyword_df.drop('keyword_num', axis=1, inplace=True)
        self.keyword_df.reset_index(inplace=True, drop=True)
        
    def getKeywordDataFrame(self):
        return self.keyword_df

def outputKeywords(text):

    # Filter out neutral sentences and sentences that are only slightly positive/negative
    filtered_text = SentimentFilter(text)
    filtered_text.getSentiment()
    filtered_text.filterPositives(0.75)
    filtered_text.filterNegatives(0.75)
    filtered_text.filterNeutrals()
    sentiment_df = filtered_text.getDataFrame()

    # Extract keywords
    model_keywords = KeyWordExtractor(filtered_text.getFilteredText())
    model_keywords.singleRank(n=20)
    model_keywords_df = pd.DataFrame(model_keywords.kw_singlerank,columns=['keyword', 'score'])

    # Filter out duplicated keywords and keywords that appear in multiple sentences
    dupe_filter = keywordFilter(list(model_keywords_df['keyword']), sentiment_df)
    dupe_filter.sentenceCountFilter()
    dupe_filter.duplicateFilter()

    output = dupe_filter.getKeywordDataFrame()[['keyword', 'sentiment']]
    output = list(output.itertuples(index=False, name=None))

    return output
