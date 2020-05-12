# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 05:09:27 2020

@author: fahma
"""

# Importing required libraries
import sys
import copy
import numpy as np
import random
from sklearn.metrics import mean_squared_error,r2_score#, ndcg_score
import math
from manifold import manifold_learning
import os.path

def normalise_sim(similarity_matrix):
       """ This function aims to normalize the similarity matrix
       using symmetric normalized Laplacian
       The input must be a square matrix
       """
       
       similarity_matrix = np.matrix(similarity_matrix)
       
       for round in range(200):
            summ = np.sum(similarity_matrix, axis=1)
            a = np.matrix(summ)
            D = np.diag(a.A1) # the diagonal matrix
            D1 = np.linalg.pinv(np.sqrt(D)); 
            similarity_matrix = D1 * similarity_matrix * D1;
    
       return similarity_matrix

def modelEvaluation(real_matrix,predict_matrix,testPosition): 
       """ This function computes the evaluation criteria
       
       real_matrix: is a matrix with cell lines in rows, drugs in columns,
       and real IC50 in its elemnts
       
       predict_matrix: has the same size as the real matrix 
       with the predicted IC50 values
       
       testPosition: is a vecoto, containing the pairs of (i,j) indices of 
       cell line-drug pairs that were considered as the test samples 
       in cross validation
       """
       
       real_pred=[]
       real_labels=[]
       predicted_probability=[]
        
       # gather the test position values in real_matrix and predict_matrix 
       # into vectors
       for i in range(0,len(testPosition)):
           real_pred.append(real_matrix[testPosition[i][0], testPosition[i][1]])
           predicted_probability.append(predict_matrix[testPosition[i][0], testPosition[i][1]])

       real_labels = np.array(real_labels)
       
       # computing evaluation criteria
       mse = mean_squared_error(real_pred, predicted_probability)
       rmse = math.sqrt(mse)
       R2 = r2_score(real_pred, predicted_probability)
       pearson = np.corrcoef(real_pred, predicted_probability)[0, 1]
       results = [mse, rmse, R2, pearson]

       return results


def runapp(response_file, simC_name, simD_name, percent, miu, landa, CV_num, repetition):
    
    """ This function runs the cross validtion
    
    response_file is the address and name of real IC50 file
    SimC_name and SimD_name are the address and name of similarity matrices
    of cell lines and drugs, respectively
    percent is the rank of latent matrix
    miu and landa are two model hyperparametrs.
    miu is the regularization coeeficient for latent matrices
    landa controls the similarity conservation while manifold learning
    CV_num is the number of folds in cross validation
    repetition is the number of repeting the cross validation
    """
    #-----------------------------------------------------------
    
    
    #reading IC50 file
    R = np.loadtxt(response_file, dtype=float, delimiter=",")      
        
    # reading similarity matrices
    simD = np.loadtxt(simD_name, dtype=float, delimiter=",")
    simC = np.loadtxt(simC_name, dtype=float, delimiter=",")
    
    # constructing indices matrix
    seed = 0
    pos_number = 0
    all_position = []
    for i in range(0, len(R)):
            for j in range(0, len(R[0])):
                    pos_number = pos_number + 1
                    all_position.append([i, j])
    
    all_position = np.array(all_position)
    random.seed(seed)
    index = np.arange(0, pos_number)
    random.shuffle(index)  # shuffle the indices
    fold_num = (pos_number)// CV_num

    # initilaizing evaluation criteria
    mse_cv = 0     
    rmse_cv = 0
    R2_cv = 0
    pear_cv = 0 
    
    # repeting the cross valiation
    for rep in range(repetition):
        print('*********repetition:' + str(repetition) + "**********\n")
        mse = 0     
        rmse = 0
        R2 = 0
        pear = 0 
        #running the cross validation
        for CV in range(0, CV_num):
            print('*********round:' + str(CV) + "**********\n")
            # seleting test positions
            test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
            test_index.sort()
            testPosition = all_position[test_index]
            train_IC= copy.deepcopy(R)
            
            # set the IC50 values for the test positions to zero
            for i in range(0, len(testPosition)):
                train_IC[testPosition[i, 0], testPosition[i, 1]] = 0
            testPosition = list(testPosition)
          
            # initialize the latent matrices
            
            N = len(train_IC)
            M = len(train_IC[0])
            dim = min(N, M)
            K  = int(round (percent * dim))
            P = np.random.rand(N ,K)
            Q = np.random.rand(M, K)
            
            # call manifold learning
            predict_matrix1, A1,B1 = manifold_learning(train_IC, P, Q, K, simD, simC, landa, miu)
            predict_matrix2, A2,B2 = manifold_learning(train_IC.T, B1, A1, K, simC, simD, landa, miu)
            predict_matrix = 0.5 * (predict_matrix1 + predict_matrix2.T)
            
            # evaluate the model
            results  = modelEvaluation(R, predict_matrix, testPosition)
            mse = mse + results[0]
            rmse = rmse + results[1]
            R2 = R2 + results[2]
            pear = pear + results[3]

        # averaging criteria over folds
        mse_cv = mse_cv + round(mse / CV_num, 4)
        rmse_cv = rmse_cv + round(rmse / CV_num, 4)
        R2_cv = R2_cv + round(R2 / CV_num, 4)
        pear_cv = pear_cv + round(pear / CV_num, 4)

    # averaging criteria over repetitions   
    mse_rep = round(mse_cv / repetition, 4)
    rmse_rep = round(rmse_cv / repetition, 4)
    R2_rep = round(R2_cv / repetition, 4)
    pear_rep = round(pear_cv / repetition, 4)
    fit_rep = round(pear_rep + R2_rep - rmse_rep, 4)
    
    print( mse_rep, ' ',rmse_rep, ' ', R2_rep, ' ',pear_rep, ' ', fit_rep)

def main():
    # get the options from user
    for arg in sys.argv[1:]:
      (key,val) = arg.rstrip().split('=')
      if key == 'response_dirc':
          response_file=val
      elif key=='simC_dirc':
          simC_name=val
      elif key=='simD_dirc':
          simD_name=val
      elif key=='dim':
          percent=float(val)
      elif key=='miu':
          miu=float(val)
      elif key=='lambda':
          landa=float(val)
      elif key=='CV':
          CV_num=int(val)
      elif key=='repeat':
          repetition=int(val)
          
    # call the method
    runapp(response_file, simC_name, simD_name, percent, miu, landa, CV_num, repetition)
    
main()
