import numpy
from Dictionary import Dictionary


class LDA:

    def __init__(self, corpus=None, stop_words=None, K=20, alpha=0.5,
                 beta=0.5, iterations=50):

        self.__vocabulary = Dictionary(stop_words, excluds_stopwords=False)
        docs = [self.__vocabulary.doc_to_ids(doc.get_text()) for doc in corpus.get_documents()]
        self.__V = self.__vocabulary.size()  # number of different words in the vocabulary
        self.__K = K
        self.__alpha = numpy.ones(K) * alpha  # parameter of topics prior
        self.__docs = docs  # a list of documents which include the words
        self.__pers = []  # Array for keeping perplexities over iterations
        self.__beta = numpy.ones(self.__vocabulary.size()) * beta  # parameter of words prior
        self.__z_m_n = {}  # topic assignements for documents
        self.__n_m_z = numpy.zeros((len(self.__docs), K))  # number of words assigned to topic z in document m
        self.__n_z_t = numpy.zeros((K, self.__vocabulary.size())) + beta  # number of times a word v is assigned to a topic z
        self.__theta = numpy.zeros((len(self.__docs), K))  # topic distribution for each document
        self.__phi = numpy.zeros((K, self.__vocabulary.size()))  # topic-words distribution for whole of corpus
        self.__n_z = numpy.zeros(K) + self.__vocabulary.size() * beta  # total number of words assigned to a topic z
        self.__iterations = iterations

        for m, doc in enumerate(docs):  # Initialization
            for n, w in enumerate(doc):
                z = numpy.random.randint(0, K)  # Randomly assign a topic to a word and increase the counting array
                self.__n_m_z[m, z] += 1
                self.__n_z_t[z, w] += 1
                self.__z_m_n[(m, n)] = z
                self.__n_z[z] += 1

    def inference(self, iteration):
        for m, doc in enumerate(self.__docs):
            self.__theta[m] = numpy.random.dirichlet(self.__n_m_z[m] + self.__alpha, 1)
            # sample Theta for each document using uncollapsed gibbs

            for n, w in enumerate(doc):  # update arrays for each word of a document
                z = self.__z_m_n[(m, n)]
                self.__n_m_z[m, z] -= 1
                self.__n_z_t[z, w] -= 1
                self.__n_z[z] -= 1
                self.__phi[:, w] = self.__n_z_t[:, w] / self.__n_z

                p_z = self.__theta[m] * self.__phi[:, w]
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                # sample Z using multinomial distribution of equation 7 of reference 3
                self.__n_m_z[m, new_z] += 1
                self.__n_z_t[new_z, w] += 1
                self.__n_z[new_z] += 1
                self.__z_m_n[(m, n)] = new_z

        per = 0
        b = 0
        c = 0
        self.__phi = self.__n_z_t / self.__n_z[:, numpy.newaxis]

        for m, doc in enumerate(self.__docs):  # find perplexity over whole of the words of test set
            b += len(doc)

            for n, w in enumerate(doc):
                l = 0
                for i in range(self.__K):
                    l += (self.__theta[m, i]) * self.__phi[i, w]
                c += numpy.log(l)

        per = numpy.exp(-c / b)
        print('per:', per)

    def worddist(self):
        return self.__phi

    def run(self):
        for i in range(self.__iterations):
            # print('iteration:', i)
            self.inference(i)

        d = self.worddist()
        for i in range(20):
            ind = numpy.argpartition(d[i], -10)[-10:]
            for j in ind:
                print(self.__vocabulary[j], ' '),
            print()




