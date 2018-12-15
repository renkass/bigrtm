# coding=utf-8
import re
from StopWords import StopWords
from Preprocessing import Preprocessing


class Dictionary:

    def __init__(self, stop_words, excluds_stopwords=False):
        self.stop_words = stop_words.get_stop_words()
        self.vocas = []  # id to word
        self.vocas_id = dict()  # word to id
        self.docfreq = []  # id to document frequency
        self.excluds_stopwords = excluds_stopwords

    def term_to_id(self, term0):

        term = Preprocessing.convert_word_to_normal_form(term0)
        term = Preprocessing.lemmatize(term)
        if not re.match(r'[a-zа-я]+$', term):
            return None
        if self.excluds_stopwords and StopWords.is_stop_word(term):
            return None
        try:
            term_id = self.vocas_id[term]
        except:
            term_id = len(self.vocas)
            self.vocas_id[term] = term_id
            self.vocas.append(term)
            self.docfreq.append(0)
        return term_id

    def doc_to_ids(self, doc):
        l = []
        words = dict()
        for term in doc.split():
            id = self.term_to_id(term)
            if id != None:
                l.append(id)
                if not id in words:
                    words[id] = 1
                    self.docfreq[id] += 1
        return l

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)