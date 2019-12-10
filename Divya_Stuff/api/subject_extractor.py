import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']

def clean_document(document):
    """Remove enronious characters. Extra whitespace and stop words"""
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split() if i not in stop])
    
    return document

def tokenize_sentences(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences

def get_entities(document):
    """Returns Named Entities using NLTK Chunking"""
    entities = []
    sentences = tokenize_sentences(document)

    # Part of Speech Tagging
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                entities.append(' '.join([c[0] for c in chunk]).lower())
    return entities

def word_freq_dist(document):
    """Returns a word count frequency distribution"""
    words = nltk.tokenize.word_tokenize(document)
    words = [word.lower() for word in words if word not in stop]
    fdist = nltk.FreqDist(words)
    
    return fdist

def extract_top_entities(document):
    # Get most frequent Nouns
    fdist = word_freq_dist(document)
    most_freq_nouns = [w for w, c in fdist.most_common(5)
                       if nltk.pos_tag([w])[0][1] in NOUNS]

    # Get Top 5 entities
    entities = get_entities(document)
    top_5_entities = [w for w, c in nltk.FreqDist(entities).most_common(5)]

    # Get the subject noun by looking at the intersection of top 5 entities
    # and most frequent nouns. It takes the first element in the list
    subject_nouns = [entity for entity in top_5_entities
                    if entity.split()[0] in most_freq_nouns]
    
    return subject_nouns

def extract_subject(document):
    # Get most frequent Nouns
    fdist = word_freq_dist(document)
    most_freq_nouns = [w for w, c in fdist.most_common(5)
                       if nltk.pos_tag([w])[0][1] in NOUNS]

    # Get Top 5 entities
    entities = get_entities(document)
    print('entities:', entities)
    print('most_freq_nouns:', most_freq_nouns)
    subject = ''
    for noun in most_freq_nouns:
        if len(noun) > 2:
            for entity in entities:
                if noun in entity:
                    subject = noun
                    break
        if subject:
            break
    print('subject:', subject)
    return [subject]

# document = clean_document(document)
# subject = extract_subject(document)
    
    