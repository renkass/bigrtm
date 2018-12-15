from Preprocessing import Preprocessing


class Document:

    def __init__(self, text, tag=''):
        self._text = text.strip()
        self._text_as_list = []
        self._tag = tag.strip()

    def get_text(self):
        return self._text

    def get_tag(self):
        return self._tag

    def set_text(self, text):
        self._text = text.strip()
        self._text_as_list = []

    def set_tag(self, tag=''):
        self._tag = tag.strip()

    def get_text_as_list_of_words(self):
        return Preprocessing.convert_text_to_list_of_words(self._text)
