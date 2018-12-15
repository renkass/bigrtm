import codecs


class StopWords:

    def __init__(self):
        self._stopWords = []
        self.__pathToFileWithStopWords = "/Users/ruslantagirov/Desktop/Univer/3course/data-repo/LabWorks/data" \
                                         "/stopwords.dic"

    def getStopWords(self):
        return self._stopWords

    def loadStopWords(self):
        file = codecs.open(self.__pathToFileWithStopWords, 'r', 'utf-8')
        stop_words = [line.strip() for line in file]
        file.close()
        self._stopWords = stop_words

