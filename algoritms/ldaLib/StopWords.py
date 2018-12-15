import codecs
import nltk


class StopWords:

    def __init__(self):
        self._stopWords = []
        self.__pathToFileWithStopWords = "/home/goncharoff/PythonLab/labs/data/stopwords.dic"
        self.load_stop_words_from_file()
        self.load_stop_word_from_nltk_lib()

    def get_stop_words(self):
        return self._stopWords

    def load_stop_words_from_file(self):
        file = codecs.open(self.__pathToFileWithStopWords, 'r', 'utf-8')
        stopWords = [line.strip() for line in file]
        file.close()
        self._stopWords = stopWords

    def load_stop_word_from_nltk_lib(self):
        stopwords_list = nltk.corpus.stopwords.words('russian')
        self._stopWords.append(list(set(stopwords_list)))

    def is_stop_word(self, stopWord):
        return stopWord in self._stopWords
