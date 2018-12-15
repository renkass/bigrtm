import copy
import re
import pymorphy2


class Preprocessing:
    morphAnalyzer = pymorphy2.MorphAnalyzer()

    @staticmethod
    def textToList(text):
        return re.sub("[^\w]", " ", text).split()

    @staticmethod
    def documentToList(document):
        return re.sub("[^\w]", " ", document.getText()).split()

    @staticmethod
    def toNormalForm(list_of_words):
        normalized_list_of_words = [Preprocessing.morphAnalyzer.normal_forms(word)[0] for word in list_of_words]
        return normalized_list_of_words

    @staticmethod
    def removeStopWords(stop_words, list_of_words):
        copy_set_of_words = copy.copy(list_of_words)
        copy_set_of_stop_words = copy.copy(stop_words)
        answer = copy.copy(copy_set_of_words)
        for word in copy_set_of_words:
            for stop_word in copy_set_of_stop_words:
                if word == stop_word:
                    answer = list(filter(lambda el: el != stop_word, answer))
        return list(answer)
