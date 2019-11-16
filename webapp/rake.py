import nltk
import numpy as np
from itertools import groupby, chain
from collections import Counter, defaultdict
import spacy
spacy_nlp = spacy.load("en_core_web_sm")

class Rake(object):
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