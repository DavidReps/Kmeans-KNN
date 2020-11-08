#Pset 5
from scipy import stats
import math
import numpy
import scipy
import matplotlib.pylab as plt
from numpy import random
from scipy.sparse.linalg import norm
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from numpy import r_
from sklearn.model_selection import KFold, cross_val_score
from scipy.spatial import distance
import matplotlib.pyplot as plth
from scipy.stats import gaussian_kde
from numpy.random import random

#  Use 20-fold cross-validation to evaluate the classification error rate of k-NN over
# the Iris dataset in sklearn, for each of the values k = 1, 2, 4, 8, 16, 32. Use a
# KNeighborsClassifier with the appropriate parameter values.
# Plot the crossvalidated error rate values as a function of k

iris = datasets.load_iris()
X = iris.data
y = iris.target

k_range = range(1, 33)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X,y)

    scores = cross_val_score(knn, X, y, cv=20)
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.title("nearest neighbor optimization")
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# (a) Write a Python program around the randPointUnitBall(d) function that generates a data set consisting of 1000 independently sampled points in the ddimensional unit ball for each dimension d = 1, 10, 100, and that reports the
# mean Euclidean length (norm) of these examples for each d. Submit your documented source code (copy the text into your writeup, and attach the source file
# separately), as well as the results. Describe the results.
# (b) For additional insight, plot a histogram of the Euclidean lengths for dimensions
# 2 = 1, 10, 100. Use matplotlib.pyplot.hist and specify density=True as one
# of the parameters. All three histograms should be superimposed in the same
# figure, in order to facilitate comparisons. Submit your code (text in writeup, file
# separately). Include the histogram in your writeup. Describe the results.
# (c) Drilling down a bit more, compute the mean Euclidean length of the origin’s three
# nearest neighbors in the data sample for each d = 1, 10, 100.

## QUESTION: 2
pi = math.pi
def cos(x):
    return math.cos(x)

def sin(x):
    return math.sin(x)

def randPointUnitBall(d):
    if d == 1:
        return 2*random(size=(1,))-1
    r = random()**(1/d)
    theta = 2*pi*random()
    p = randPointUnitBall(d-1)

    return r_[r*cos(theta), r*sin(theta)*p/numpy.linalg.norm(p)]


def DimensionDistance(n):
    distance = []
    #create 1000 points measure distance to origin
    #save distance in array
    for i in range(1000):
        point = randPointUnitBall(n)
        origin = numpy.zeros(n)
        dist = scipy.spatial.distance.euclidean(origin, point)
        distance.append(dist)

    return distance

def dimensionArrays():

    #recieve distances and compute the mean
    oneD = DimensionDistance(1)
    mean1D = numpy.mean(oneD)
    sorted1 = numpy.sort(oneD)
    meanMin1 = numpy.mean(sorted1[:3])


    print("mean distance of nearest 3 neighbors for 1 D:", meanMin1)
    print("one dimension mean:", mean1D, '\n')

    tenD = DimensionDistance(10)
    mean10D = numpy.mean(tenD)
    sorted10 = numpy.sort(tenD)
    meanMin10 = numpy.mean(sorted10[:3])


    print("mean distance of nearest 3 neighbors for 10 D:", meanMin10)
    print("10 dimension mean:", mean10D, '\n')

    hunitD = DimensionDistance(100)
    mean100D = numpy.mean(hunitD)
    sorted100 = numpy.sort(hunitD)
    meanMin100 = numpy.mean(sorted100[:3])


    print("mean distance of nearest 3 neighbors: for 100 D", meanMin100)
    print("mean of 100 dimension:", mean100D, '\n')

#plot the distances on a histogram so as to interpret differences in higher dimensions
    plth.hist(oneD, density=True)
    plth.hist(tenD, density=True)
    plth.hist(hunitD, density=True)

    plt.xlabel('Distance from origin')
    plt.ylabel('Relative frequency of distance')
    plt.title('Higher dimension distances')
    plt.grid(True)
    plt.show()

dimensionArrays()


# Write a program in Python that will generate a random sample of 1000 points
# from the uniform distribution on [0, 1), then compute an approximation g(x) to
# the uniform PDF p(x) for each of the five kernel widths w = 10α
# , where α ranges
# between −2 and 0 in 0.5 increments. Each of these results should be evaluated
# using (a discrete version of) the β error metric as described above. Submit plots
# of the PDF estimates, together with their β values. What kernel width yields the
# best result (smallest value of β)? Include a listing of your source code in your
# writeup, and attach the source file separately.
# (b) Evaluate two options for the best kernel width, w
# ∗
# , as a function of the sample
# size, n: w = n^−1/2 and w = n^−1/3
# , empirically, modifying your code from the
# preceding part as needed. Do multiple runs in order to gather information, with
# n ranging from 10 to one million. Which of the two options produces better
# results? Discuss, pointing to specific evidence from your experiments.

def beta(model):
    intervalPoints = numpy.linspace(-1, 2, 50)
    intervalWidth = .06
    step = .001
    final = 0
    #50 interval estimation
    for i in range(49):
        p = 0
        if intervalPoints[i] <= 0 and intervalPoints[i] < 1:
            p = 1
        g_1 = math.sqrt(abs(p-model(intervalPoints[i])))
        derivative = (p-model(intervalPoints[i]+ step)-p+model(intervalPoints[i]))/step
        g_2 = math.sqrt(abs(derivative))
        final += intervalWidth * ((g_1 * (2/3)) + (g_2 * (1/3)))
    return final
#

alpha = [-2,-1.5,-1,-.5,0]
gausData = numpy.random.uniform(0,1,1000)
kernels = list()
err = numpy.linspace(-1,2,50)

for alphas in alpha:
    w = 10**alphas
    # gausData = numpy.random.ranf(1000)
    gkde = scipy.stats.gaussian_kde(gausData, bw_method = w)
    result = beta(gkde)
    print("[1/2] kernal error with width %f :" % w, result)
    kernels.append(gkde)

for kernal in kernels:
    plt.plot(err, kernal(err))
plt.show()
