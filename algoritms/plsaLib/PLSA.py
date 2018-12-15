import codecs
import re
from numpy import log
from numpy import int8
from numpy import zeros
from pylab import random
from .Preprocessing import Preprocessing


class PLSA:

    def __init__(self, corpus, stopWords):
        self._corpus = corpus
        self._stopWords = stopWords
        self._K = 10  # number of topic
        self._maxIteration = 20
        self._threshold = 10.0
        self._topicWordsNum = 10
        self._doc_topic = '/Users/ruslantagirov/Desktop/Univer/3course/data-repo/LabWorks/labs/LW3/results/' \
                          'doc-topic.txt'
        self._topic_word = '/Users/ruslantagirov/Desktop/Univer/3course/data-repo/LabWorks/labs/LW3/results' \
                           '/topic-word.txt'
        self._dic = '/Users/ruslantagirov/Desktop/Univer/3course/data-repo/LabWorks/labs/LW3/results/dic.dic'
        self._topics = '/Users/ruslantagirov/Desktop/Univer/3course/data-repo/LabWorks/labs/LW3/results/topics.txt'

        self._N = len(corpus.getDocuments())
        self._wordCounts = []
        self._word2id = {}
        self._id2word = {}
        self._currentId = 0

        for document in corpus.getDocuments():
            normalized_list = Preprocessing.toNormalForm(Preprocessing
                                                         .documentToList(document))
            self._segList = Preprocessing.removeStopWords(self._stopWords.getStopWords(),
                                                          normalized_list)
            self._wordCount = {}
            for word in self._segList:
                word = word.lower().strip()
                if len(word) > 1 and not re.search('[0-9]', word) and word not in self._stopWords.getStopWords():
                    if word not in self._word2id.keys():
                        self._word2id[word] = self._currentId
                        self._id2word[self._currentId] = word
                        self._currentId += 1
                    if word in self._wordCount:
                        self._wordCount[word] += 1
                    else:
                        self._wordCount[word] = 1
            self._wordCounts.append(self._wordCount)

        # length of dic
        self._M = len(self._word2id)

        # generate the document-word matrix
        self._X = zeros([self._N, self._M], int8)
        for word in self._word2id.keys():
            j = self._word2id[word]
            for i in range(0, self._N):
                if word in self._wordCounts[i]:
                    self._X[i, j] = self._wordCounts[i][word]

        # lambda: p(zj|di)
        self._lambda = random([self._N, self._K])

        # theta: p(wj|zi)
        self._theta = random([self._K, self._M])

        # p: p(zk|di,wj)
        self._p = zeros([self._N, self._M, self._K])

    def initParameters(self):
        for i in range(0, self._N):
            self._normalization = sum(self._lambda[i, :])
            for j in range(0, self._K):
                self._lambda[i, j] /= self._normalization

        for i in range(0, self._K):
            self._normalization = sum(self._theta[i, :])
            for j in range(0, self._M):
                self._theta[i, j] /= self._normalization

    def EM(self):
        # EM algorithm
        likelihood_before = 1
        for i in range(0, self._maxIteration):
            self.E()
            self.M()
            likelihood_after = self.raspredelenie()
            if likelihood_before != 1 and likelihood_after - likelihood_before < self._threshold:
                break
            likelihood_before = likelihood_after
        self.write_to_files()

    def E(self):
        for i in range(0, self._N):
            for j in range(0, self._M):
                denominator = 0
                for k in range(0, self._K):
                    self._p[i, j, k] = self._theta[k, j] * self._lambda[i, k]
                    denominator += self._p[i, j, k]
                if denominator == 0:
                    for k in range(0, self._K):
                        self._p[i, j, k] = 0
                else:
                    for k in range(0, self._K):
                        self._p[i, j, k] /= denominator

    def M(self):
        # update theta
        for k in range(0, self._K):
            denominator = 0
            for j in range(0, self._M):
                self._theta[k, j] = 0
                for i in range(0, self._N):
                    self._theta[k, j] += self._X[i, j] * self._p[i, j, k]
                denominator += self._theta[k, j]
            if denominator == 0:
                for j in range(0, self._M):
                    self._theta[k, j] = 1.0 / self._M
            else:
                for j in range(0, self._M):
                    self._theta[k, j] /= denominator

        # update lambda
        for i in range(0, self._N):
            for k in range(0, self._K):
                self._lambda[i, k] = 0
                denominator = 0
                for j in range(0, self._M):
                    self._lambda[i, k] += self._X[i, j] * self._p[i, j, k]
                    denominator += self._X[i, j]
                if denominator == 0:
                    self._lambda[i, k] = 1.0 / self._K
                else:
                    self._lambda[i, k] /= denominator

    # calculate the log raspredelenie
    def raspredelenie(self):
        loglikelihood = 0
        for i in range(0, self._N):
            for j in range(0, self._M):
                tmp = 0
                for k in range(0, self._K):
                    tmp += self._theta[k, j] * self._lambda[i, k]
                if tmp > 0:
                    loglikelihood += self._X[i, j] * log(tmp)
        return loglikelihood

    # output the params of model and top words of topics to files
    def write_to_files(self):
        # document-topic distribution
        file = codecs.open(self._doc_topic, 'w', 'utf-8')
        for i in range(0, self._N):
            tmp = ''
            for j in range(0, self._K):
                tmp += str(self._lambda[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()

        # topic-word distribution
        file = codecs.open(self._topic_word, 'w', 'utf-8')
        for i in range(0, self._K):
            tmp = ''
            for j in range(0, self._M):
                tmp += str(self._theta[i, j]) + ' '
            file.write(tmp + '\n')
        file.close()

        # dictionary
        file = codecs.open(self._dic, 'w', 'utf-8')
        for i in range(0, self._M):
            file.write(self._id2word[i] + '\n')
        file.close()

        # top words of each topic
        file = codecs.open(self._topics, 'w', 'utf-8')
        for i in range(0, self._K):
            topic_word = []
            ids = self._theta[i, :].argsort()
            for j in ids:
                topic_word.insert(0, self._id2word[j])
            tmp = ''
            for word in topic_word[0:min(self._topicWordsNum, len(topic_word))]:
                tmp += word + ' '
            file.write(tmp + '\n')
        file.close()
