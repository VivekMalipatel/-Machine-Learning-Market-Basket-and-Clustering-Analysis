import matplotlib.pyplot as plt
import numpy
import pandas
import random

from sklearn.metrics.pairwise import manhattan_distances

def getPositionSRS (nObs, nCluster):

    centroid_pos = []

    kObs = 0
    iSample = 0
    for iObs in range(nObs):
        kObs = kObs + 1
        uThreshold = (nCluster - iSample) / (nObs - kObs + 1)
        if (random.random() < uThreshold):
            centroid_pos.append(iObs)
            iSample = iSample + 1

        if (iSample == nCluster):
            break

    return (centroid_pos)

def assignMember (trainData, centroid):
   
    pair_distance = manhattan_distances(trainData, centroid)
    member = pandas.Series(numpy.argmin(pair_distance, axis = 1), name = 'Cluster')
    wc_distance = pandas.Series(numpy.min(pair_distance, axis = 1), name = 'Distance')

    return (member, wc_distance)

def KMeansCluster (trainData, nCluster, nIteration = 500, nTrial = 10, randomSeed = None):
    n_obs = trainData.shape[0]

    if (randomSeed is not None):
      random.seed(a = randomSeed)

    list_centroid = []
    list_wcss = []
    for iTrial in range(nTrial):
        centroid_pos = getPositionSRS(n_obs, nCluster)
        centroid = trainData.iloc[centroid_pos]
        member_prev = pandas.Series([-1] * n_obs, name = 'Cluster')

        for iter in range(nIteration):
            member, wc_distance = assignMember(trainData, centroid)

            centroid = trainData.join(member).groupby(by = ['Cluster']).mean()
            member_diff = numpy.sum(numpy.abs(member - member_prev))
            if (member_diff > 0):
                member_prev = member
            else:
                break

        list_centroid.append(centroid)
        list_wcss.append(numpy.sum(wc_distance))

    best_solution = numpy.argmin(list_wcss)
    centroid = list_centroid[best_solution]
   
    member, wc_distance = assignMember(trainData, centroid)
   
    return (member, centroid, wc_distance)

trainData = pandas.read_csv('/Users/vivekmalipatel/Downloads/Assignmnts/IntrotoMLAssignment/Question/TwoFeatures.csv')

n_sample = trainData.shape[0]
max_nCluster = 8

nClusters = []
Elbow = []
TotalWCSS = []

for k in range(max_nCluster):
    nCluster = k + 1
    member, centroid, wc_distance = KMeansCluster(trainData, nCluster, nTrial = 20, randomSeed = 20231225)

    print(centroid)

    WCSS = numpy.zeros(nCluster)
    nC = numpy.zeros(nCluster)

    for i in range(n_sample):
        k = member[i]
        nC[k] += 1 
        WCSS[k] += wc_distance[i]**2

    E = 0.0
    T = 0.0
    for k in range(nCluster):
        E += WCSS[k] / nC[k]
        T += WCSS[k]
 
    nClusters.append(nCluster)
    Elbow.append(E)
    TotalWCSS.append(T)

plt.figure(figsize = (8,6), dpi = 200)
plt.plot(nClusters, Elbow, marker = 'o', color = 'royalblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Elbow Value')
plt.xticks(range(1,max_nCluster+1))
plt.grid()
plt.show()

result_df = pandas.DataFrame({'N Cluster': nClusters,
                              'Total WCSS': TotalWCSS,
                              'Elbow': Elbow,})