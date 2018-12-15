from Document import Document


class Corpus:

    def __init__(self):
        self.__pathToFile = ''
        self.__documents = []

    def load_corpus_from_list(self, documents, tags=[]):
        for index in range(len(documents)):
            try:
                self.__documents.append(Document(documents[index], tags[index]))
            except IndexError:
                self.__documents.append(Document(documents[index], ''))

    def get_documents(self):
        return self.__documents

    def get_document_by_index(self, index):
        try:
            return self.__documents[index]
        except IndexError:
            raise IndexError
