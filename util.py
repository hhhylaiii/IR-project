import sys

#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
	from numpy import array
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

def euclideanSimilarity(vector1, vector2):
	return 1/(1 + norm(array(vector1) - array(vector2)))  #1/(1+dist) to give a rating between 0 and 1

def euclidean(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		euclidean  = sqrt( sum( ( V1i - V2i )^2 ) ) """
	return norm(array(vector1) - array(vector2)) 