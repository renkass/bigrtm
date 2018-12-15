# coding=utf-8
# install packages for concrete python version sudo python -m pip install pck

import artm
import re
from nltk.corpus import stopwords
import pymorphy2
from multiprocessing import cpu_count
from labs.algoritms.ldaLib.LDA import LDA
from labs.algoritms.ldaLib.Corpus import Corpus
from labs.algoritms.ldaLib.StopWords import StopWords
import pandas as pd


def preprocessing_for_artm(number_of_docs_as_collection_length=False, number_of_docs=10):
    data = pd.read_csv("/home/goncharoff/PythonLab/labs/data/lenta_ru.csv")
    texts = data["text"]
    all_docs = ""
    morph = pymorphy2.MorphAnalyzer()
    if number_of_docs_as_collection_length:
        number_of_docs = len(texts)
    for i in range(number_of_docs):
        text = texts[i]
        all_docs += " |text "
        text = str(text).decode('utf-8')
        text = re.sub("[0-9!@#$%^&*()\[\],\.<>;:\"{}/~`\-+=«»—\|?\^\n\t']+", '', text)
        list_of_words = re.sub(ur"(u?)\w+", ' ', text, ).split(" ")
        filtered_list_of_word = [morph.parse(w.lower())[0].normal_form
                                 for w in list_of_words if w not in stopwords.words("russian")]
        filtered_text = u" ".join(filtered_list_of_word).encode('utf-8').strip()
        all_docs += filtered_text
        all_docs += "\n"
    f = open("/home/goncharoff/PythonLab/labs/labs/lab5/result/result.txt", "w")
    f.write(all_docs)
    f.close()


def myplsa():
    return 0




def artm_plsa(batch_vectorizer, topics, topic_names, dictionary):
    model_artm = artm.ARTM(num_topics=topics, topic_names=topic_names, num_processors=cpu_count(),
                           class_ids={"text": 1}, reuse_theta=True, cache_theta=True, num_document_passes=1)
    model_artm.initialize(dictionary=dictionary)
    model_artm.scores.add(artm.PerplexityScore("perplexity", class_ids=["text"], dictionary=dictionary))
    model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=50)
    print "\nPeprlexity for BigARTM PLSA: ", model_artm.score_tracker["perplexity"].value[-1]


def artm_lda(batch_vectorizer, topics, dictionary):
    model_lda = artm.LDA(num_topics=topics, num_processors=cpu_count(), cache_theta=True, num_document_passes=1)
    model_lda.initialize(dictionary=dictionary)
    model_lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=50)
    print "\nPerplexity for BigARTM LDA: ", model_lda.perplexity_last_value


def run():
    print 'BigARTM version ', artm.version(), '\n\n\n'
    preprocessing_for_artm(True)
    topics = 10
    batch_vectorizer = artm.BatchVectorizer(
        data_path="/home/goncharoff/PythonLab/labs/labs/lab5/result/result.txt",
        data_format="vowpal_wabbit",
        target_folder="batch_vectorizer_target_folder", batch_size=10)
    topic_names = ["topic#1" + str(i) for i in range(topics - 1)] + ["bcg"]
    dictionary = artm.Dictionary("dictionary")
    dictionary.gather(batch_vectorizer.data_path)
    artm_plsa(batch_vectorizer, topics, topic_names, dictionary)
    artm_lda(batch_vectorizer, topics, dictionary)

    # print "MY LDA: \n"
    #
    # sw = StopWords()
    #
    # data = pd.read_csv("/home/goncharoff/PythonLab/labs/data/lenta_ru.csv")
    # documents = data["text"].tolist()
    #
    # corpus = Corpus()
    # corpus.load_corpus_from_list(documents)
    #
    # lda = LDA(corpus=corpus, stop_words=sw, K=20, alpha=0.5, beta=0.5, iterations=50)
    # lda.run()
    # print("\n", lda.worddist())


run()
