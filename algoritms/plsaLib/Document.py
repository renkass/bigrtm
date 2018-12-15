class Document:

    def __init__(self, text, topic):
        self._text = text.strip()
        self._text_as_list = []
        self._topic = topic.strip()

    def getText(self):
        return self._text
