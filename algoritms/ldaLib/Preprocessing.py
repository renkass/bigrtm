import nltk
import copy
import re
import pymorphy2


class Preprocessing:
    morphAnalyzer = pymorphy2.MorphAnalyzer()

    @staticmethod
    def convert_text_to_list_of_words(text):
        return re.sub("[^\w]", " ", text).split()

    @staticmethod
    def convert_document_to_list_of_words(document):
        return re.sub("[^\w]", " ", document.get_text()).split()

    @staticmethod
    def convert_list_of_words_to_normal_forms(list_of_words):
        normalized_list_of_words = [Preprocessing.morphAnalyzer.normal_forms(word)[0] for word in list_of_words]
        return normalized_list_of_words

    @staticmethod
    def convert_word_to_normal_form(word):
        return Preprocessing.morphAnalyzer.normal_forms(word)[0]

    @staticmethod
    def remove_stop_words_from_list_of_words(stop_words, list_of_words):
        copy_set_of_words = copy.copy(list_of_words)
        copy_set_of_stop_words = copy.copy(stop_words)
        answer = copy.copy(copy_set_of_words)
        for word in copy_set_of_words:
            for stop_word in copy_set_of_stop_words:
                if word == stop_word:
                    answer = list(filter(lambda el: el != stop_word, answer))
        return list(answer)

    @staticmethod
    def lemmatize(word):
        wl = nltk.WordNetLemmatizer()
        w = wl.lemmatize(word.lower())
        return w

    @staticmethod
    def convert_word_list_to_text(word_list ):
        text = str()
        for word in word_list:
            text += word + str(" ")
        return text.strip()
