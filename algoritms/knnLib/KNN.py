import numpy as np
from operator import itemgetter
from scipy.spatial import distance
import pandas as pd


class KNN:
    def __init__(self, data, testAnswers, k, className):
        self.data = data
        self.testAnswers = testAnswers
        self.k = k
        self.className = className

    @staticmethod
    def getDistEuclidean(_from, _to):
        return distance.euclidean(_from, _to)

    @staticmethod
    def score(table, beforeName, AfterName):
        counter = 0
        length = len(table)
        for i in range(length):
            if table.iloc[i][beforeName] != table.iloc[i][AfterName]:
                counter = counter + 1
        return 1 - (counter / length)

    def fit(self):
        y_data = self.data[self.className]
        data = self.data.drop([self.className], axis=1)
        kNeighboursList = []
        for test_index, test_row in self.testAnswers.iterrows():
            labels = []
            testDistances = {}
            for data_index, data_row in data.iterrows():
                testDistances[data_index] = self.getDistEuclidean(test_row, data_row)
            testDistancesSorted = (sorted(testDistances.items(), key=itemgetter(1)))
            for i in range(0, self.k):
                labels.append(y_data[testDistancesSorted[i][0]])
            kNeighboursList.append(labels)
        return kNeighboursList

    def predictClassification(self):
        kNeighboursList = self.fit()
        length = len(kNeighboursList)
        labeled_test = []
        for i in range(length):
            a_set = set(kNeighboursList[i])
            mostPopular = None
            mostPopularCount = 0
            for item in a_set:
                counter = kNeighboursList[i].count(item)
                if counter > mostPopularCount:
                    mostPopularCount = counter
                    mostPopular = item
            labeled_test.append(mostPopular)

        return pd.DataFrame(labeled_test)

    def predictRegression(self):
        kNeighboursList = self.fit()
        length = len(kNeighboursList)
        labeled_test = []
        for i in range(length):
            avg_val = np.mean(kNeighboursList[i])
            labeled_test.append(avg_val)

        return pd.DataFrame(labeled_test)
