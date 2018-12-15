class Corpus:

    def __init__(self):
        self.__documents = []

    def generateCorpus(self, filepath):
        from LabWorks.algoritms.plsaLib.Document import Document
        import pandas as pd

        data = pd.read_csv(filepath)
        documents = data["text"].tolist()
        topics = data["topic"].tolist()

        for index in range(len(documents)):
            try:
                self.__documents.append(Document(documents[index], topics[index]))
            except IndexError:
                self.__documents.append(Document(documents[index], ''))

    def getDocuments(self):
        return self.__documents
