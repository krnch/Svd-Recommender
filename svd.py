import unittest
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
                   
     
 
           
    matrix = np.matrix(matrix)

    return matrix

def rmse(matrix1, matrix2):
    subtraction = np.subtract(matrix2, matrix1)
    subtraction = np.array(subtraction)

    for i in range(len(subtraction)):
        for j in range(len(subtraction[0])):
            a = subtraction[i][j] * subtraction[i][j]
            subtraction[i][j] = a
    num = 0.0
    num = num + (len(subtraction)*len(subtraction[0]))
 
    sum_ = np.sum(subtraction)
    mean = sum_/num
    rmse_ = math.sqrt(mean)
    return rmse_

users = []
businesses = []

for jsonstr in open("C:\\sample_data.json").readlines():
    if(jsonstr != ""):
        jsonobj = json.loads(jsonstr)
        users.append(jsonobj["user_id"])

users = unique_list(users)

for jsonstr in open("C:\\sample_data.json").readlines():
    if(jsonstr != ""):
        jsonobj = json.loads(jsonstr)
        businesses.append(jsonobj["business_id"])

businesses = unique_list(businesses)

# Before Matrix Creation


mat = [len(users), len(businesses)]
rating_matrix = np.matrix(mat, dtype = float)


rating_matrix = np.zeros((len(users),len(users)), dtype=float)
s_no = 1
i=1
for jsonstr in open("C:\\sample_data.json").readlines():
    if(jsonstr != ""):
        jsonobj = json.loads(jsonstr)
        rating_matrix[users.index(jsonobj["user_id"])][businesses.index(jsonobj["business_id"])] = jsonobj["stars"]
        

# After Matrix Creation"

matrix1 = rating_matrix

matrix2 = avg(rating_matrix)

matrix = rating_matrix
U, s, V = linalg.svd( matrix )

dimensions = 1
rows, cols = matrix.shape

reconstructedMatrix= dot(dot(U,linalg.diagsvd(s,len(matrix),len(V))),V)

print "RMSE"
print rmse(U,matrix1)


#some random testcases just rechecking the rmse and unique functions
class MyTest(unittest.TestCase):
    def test_1(self):
        a=np.matrix('1 2 ; 3 4')
        b=np.matrix('4 5 ; 6 7')
       
        self.assertEqual(rmse(a,b), 3)
        
    def test_2(self):
        a=np.matrix('1 2 ; 3 4')
        b=np.matrix('2 3 ; 4 5')
       
        self.assertEqual(rmse(a,b), 1)
    
    def test_3(self):
        un1=[1,2,2,3,3,47,81,92]
        un2=[1,2,3,47,81,92]
        
        self.assertEqual(unique_list(un1),un2)
        
    def test_4(self):
        un1=[97,42,33]
        un2=[97,42,33]
        
        self.assertEqual(unique_list(un1),un2)
    
 unittest.main()
