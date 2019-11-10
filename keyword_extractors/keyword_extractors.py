import pke
#!python -m nltk.downloader stopwords
#!python -m nltk.downloader universal_tagset
#!python -m spacy download en

class KeyWordExtractor(object):
    def __init__(self, text):
        self.text = text
        self.sent_df = None
        self.kw_tfidf = None
        self.kw_kpminer = None
        self.kw_yake = None
        self.kw_textrank = None
        self.kw_singlerank = None
        self.kw_topicrank = None
        self.kw_tprank = None
        self.kw_positionrank = None
        self.kw_mprank = None
        
    def tfidf(self, n=20):
        extractor = pke.unsupervised.TfIdf()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_tfidf = extractor.get_n_best(n=n)
        
    def kpMiner(self, n=20):
        extractor = pke.unsupervised.KPMiner()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_kpminer = extractor.get_n_best(n=n)
        
    def yake(self, n=20):
        extractor = pke.unsupervised.YAKE()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_yake = extractor.get_n_best(n=n)
    
    def textRank(self, n=20):
        extractor = pke.unsupervised.TextRank()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_textrank = extractor.get_n_best(n=n)
    
    def singleRank(self, n=20):
        extractor = pke.unsupervised.SingleRank()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_singlerank = extractor.get_n_best(n=n)
        
    def textRank(self, n=20):
        extractor = pke.unsupervised.TextRank()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_textrank = extractor.get_n_best(n=n)
       
    def topicRank(self, n=20):
        extractor = pke.unsupervised.TopicRank()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_topicrank = extractor.get_n_best(n=n)
    
    def topicalPageRank(self, n=20):
        extractor = pke.unsupervised.TopicalPageRank()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_tprank = extractor.get_n_best(n=n)
        
    def positionRank(self, n=20):
        extractor = pke.unsupervised.PositionRank()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_positionrank = extractor.get_n_best(n=n)

    def multiPartiteRank(self, n=20):
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(self.text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        self.kw_mprank = extractor.get_n_best(n=n)

    def allExtractors(self, n=20, pos=None, window=10, normalized=False):
        self.tfidf(n=n)
        self.kpMiner(n=n)
        self.yake(n=n)
        self.textRank(n=n)
        self.singleRank(n=n)
        self.topicRank(n=n)
        self.topicalPageRank(n=n)
        self.positionRank(n=n)
        self.multiPartiteRank(n=n)
        
    def getAllKeyWords(self, include_score=False, sort=False):
        if include_score == True:
            all_keywords = {'tfidf_keywords': self.kw_tfidf, 'kpminer_keywords': self.kw_kpminer, 
                            'yake_keywords': self.kw_yake, 'textrank_keywords': self.kw_textrank,
                            'singlerank_keywords':self.kw_singlerank, 'topicrank_keywords': self.kw_topicrank, 
                            'topicalpagerank_keywords': self.kw_tprank, 'position_keywords': self.kw_positionrank,
                            'multipartiterank_keywords': self.kw_mprank}
        else:
            if sort == False:
                all_keywords = {'tfidf_keywords': [i[0] for i in self.kw_tfidf],
                                'kpminer_keywords': [i[0] for i in self.kw_kpminer], 
                                'yake_keywords': [i[0] for i in self.kw_yake],
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
                                'textrank_keywords': sorted([i[0] for i in self.kw_textrank]),
                                'singlerank_keywords': sorted([i[0] for i in self.kw_singlerank]),
                                'topicrank_keywords': sorted([i[0] for i in self.kw_topicrank]), 
                                'topicalpagerank_keywords': sorted([i[0] for i in self.kw_tprank]),
                                'position_keywords': sorted([i[0] for i in self.kw_positionrank]),
                                'multipartiterank_keywords': sorted([i[0] for i in self.kw_mprank])}

        return all_keywords
