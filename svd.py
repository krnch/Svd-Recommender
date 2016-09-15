 
import json
import gc
from scipy import linalg as LAS
import numpy as np
from numpy import linalg as LA
import csv
import scipy.io
from scipy import linalg, mat, dot
import random
import math

def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

def low_rank_approx(u, s, v, r):
    Ar = np.zeros((len(u), len(v)))
    for i in xrange(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

def avg(matrix):
    matrix = np.array(matrix)
    rows = len(matrix)
    col = len(matrix[0])

    sum_ = 0
    tc=0
    tst=0
    for i in range(rows):
        for j in range(col):
            tc=tc+1
    for i in range(rows):
        for j in range(col):
            
            if matrix[i][j] == 0.0:
                matrix[i][j] = 2.5
                tst=tst+1
               
        if tst>(tc*0.20):
                    break        
                   
                #print matrix[i][j]
 
           
    matrix = np.matrix(matrix)
    #print matrix
    return matrix

def rmse(matrix1, matrix2):
    subtraction = np.subtract(matrix2, matrix1)
    subtraction = np.array(subtraction)
    #print subtraction
    #squared = LA.matrix_power(subtraction, 2)
    for i in range(len(subtraction)):
        for j in range(len(subtraction[0])):
            a = subtraction[i][j] * subtraction[i][j]
            subtraction[i][j] = a
    num = 0.0
    num = num + (len(subtraction)*len(subtraction[0]))
    #print "SUM:\n"
    sum_ = np.sum(subtraction)
    mean = sum_/num
    rmse_ = math.sqrt(mean)
    return rmse_

users = []
businesses = []

for jsonstr in open("C:\\yelp_academic_dataset_review.json").readlines():
    if(jsonstr != ""):
        jsonobj = json.loads(jsonstr)
        users.append(jsonobj["user_id"])

users = unique_list(users)

for jsonstr in open("C:\\yelp_academic_dataset_review.json").readlines():
    if(jsonstr != ""):
        jsonobj = json.loads(jsonstr)
        businesses.append(jsonobj["business_id"])

businesses = unique_list(businesses)

print "Test Before Matrix Creation"

#rating_matrix = np.zeros((len(users),len(businesses)), dtype=float)
mat = [len(users), len(businesses)]
rating_matrix = np.matrix(mat, dtype = float)


#print len(businesses)
#print len(users)
#print rating_matrix

rating_matrix = np.zeros((len(users),len(users)), dtype=float)
s_no = 1
i=1
for jsonstr in open("C:\\yelp_academic_dataset_review.json").readlines():
    if(jsonstr != ""):
        jsonobj = json.loads(jsonstr)
        rating_matrix[users.index(jsonobj["user_id"])][businesses.index(jsonobj["business_id"])] = jsonobj["stars"]
        

print "Test After Matrix Creation"
#print rating_matrix
matrix1 = rating_matrix
#print('\n')
matrix2 = avg(rating_matrix)
#print matrix2
#print rmse(matrix2,matrix1)
#print rating_matrix
#zerofy_average(rating_matrix)
#print (rating_matrix)


matrix = rating_matrix
U, s, V = linalg.svd( matrix )
#print "U:"
#print U
#print "sigma:"
#print s
#print "VT:"
#print V
dimensions = 1
rows, cols = matrix.shape
#for index in xrange(dimensions, rows):
# s[index]=0
#print "reduced sigma:"
#print s
reconstructedMatrix= dot(dot(U,linalg.diagsvd(s,len(matrix),len(V))),V)
#print "reconstructed:"
#print reconstructedMatrix
print "RMSE"
print rmse(U,matrix1)

