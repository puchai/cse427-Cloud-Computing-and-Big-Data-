from __future__ import print_function
import sys
from math import sqrt
import numpy as np
from haversine import haversine
from pyspark import SparkContext
import random as rand

# return the index within centroids that is the closest point to p
def closestPoint(p, centroids, mode):
	distance = []
	# Euclidean distance
	if mode == 0:
		for point in centroids:
			distance.append(EuclideanDistance(p, point))
		return distance.index(min(distance))
	# Great Circle Distance		
	else:
		for point in centroids:
			distance.append(GreatCircleDistance(p, point))
		return distance.index(min(distance))



# return a point that is the sum of the two points
def addPoints(p1, p2):
	return p1 + p2


# return the Euclidean distance of two points
def EuclideanDistance(p1, p2):
	total = np.sum((p2 - p1) ** 2)
	return sqrt(total)


# return the Great Circle Distance of two points
def GreatCircleDistance(p1, p2):
	return haversine(p1, p2)


def parsePoint(line):
	pairs = line.split(",") 
	return np.array([float(x) for x in pairs])


if __name__ == "__main__":
	# handles command line input arguments
	if len(sys.argv) != 4:
		print("Usage: kmeans.py [inputURL] [k] [mode]")
		exit(-1)
	sc = SparkContext()
	# sys.argv[1] passes in the directory that stores our preprocessed data
	lines = sc.textFile(sys.argv[1])
	k = int(sys.argv[2])
	mode = int(sys.argv[3])
	data = lines.map(lambda line: parsePoint(line)).cache()
	centroids = data.takeSample(False, k, 1)
	convergeDist = 1.0
	while convergeDist > 0.1:
		closest = data.map(lambda p: (closestPoint(p, centroids, mode), (p, 1)))
		pointStats = closest.reduceByKey(lambda v1, v2: (addPoints(v1[0], v2[0]), v1[1] + v2[1]))
		updated = pointStats.map(lambda s: (s[0], s[1][0] / s[1][1])).collect()
		tempD = 0.0
		for (index, p) in updated:
			tempD += EuclideanDistance(p, centroids[index])
		convergeDist = tempD
		for (index, p) in updated:
			centroids[index] = p

	# write result clusters and centroids to file
	print("Final centroids are: " + str(centroids))
	clusters = data.map(lambda p: (closestPoint(p, centroids, mode), p)).sortByKey()
	# savefile, URL can be changed due to different tasks
	clusters.saveAsTextFile("file:///home/training/training_materials/data/k_m_res")
	sc.stop()


