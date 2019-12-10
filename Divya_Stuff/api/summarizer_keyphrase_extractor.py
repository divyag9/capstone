import re
import spacy
import seq2seq
import numpy as np
import language_check
import tensorflow as tf
from util import text_cleaner
from util import orderSentiment
from subject_extractor import clean_document
from subject_extractor import extract_subject
from rouge import Rouge
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

class SummarizerExtractor():
    def __init__(self, model_dir_path):
        if model_dir_path is None:
            model_dir_path = './models'
        self.config = np.load(seq2seq.Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path),
                              allow_pickle=True).item()
        self.summarizer = seq2seq.Seq2SeqSummarizer(self.config)
        self.summarizer.load_weights(weight_file_path=seq2seq.Seq2SeqSummarizer
                                     .get_weight_file_path(model_dir_path=model_dir_path))
        self.reviews_dataframe = self.config['reviews_dataframe']
        self.nlp = spacy.load("en_core_web_sm")
        
        
    def get_keyphrases(self, text):
        document = clean_document(text)
        subject = extract_subject(text)
        doc = self.nlp(text)
        
        sent = set()
        for token in doc:
            if token.lower_ in subject:
                sent.add(token.sent)

        cleaned_review_text = []        
        for t in sent:
            cleaned_text = text_cleaner(t,0)
            if cleaned_text and len(cleaned_text.split(' ')) > 4:
                cleaned_review_text.append([cleaned_text,t])
        reviews = np.array(cleaned_review_text)
        
        #tool = language_check.LanguageTool('en-US')
        summary_sentences = []
        for clean_sent,sent in reviews:
            summary = self.summarizer.summarize(clean_sent)
            summary_capitalized = summary.strip().capitalize()
            #matches = tool.check(summary_capitalized)
            #if len(matches) == 0:
            doc_sum = self.nlp(summary_capitalized)
            for token in doc_sum:
                if 'subj' in token.dep_ or 'obj' in token.dep_:
                    summary_sentences.append(summary_capitalized)
                    break
        print(summary_sentences)
        if summary_sentences:
            keyphrases = orderSentiment(summary_sentences)
            return keyphrases
