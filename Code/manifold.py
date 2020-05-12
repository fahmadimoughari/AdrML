# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 05:09:27 2020

@author: fahma
"""
import numpy
import numpy as np
from numpy.linalg import inv

def manifold_learning(Train_IC, P, Q, K, SimD, SimC, landa, miu):
    
    """ This function runs the manifold learning and trains the latent matrices
        Train_IC is the training matrix 
        Q and P are the latent matrices
        K is the dimension of latent space
        SimD and SimC are the similarity matrices of drugs and cell lines
        landa controls the similarity conservation while manifold learning
        miu is the regularization coeeficient for latent matrices        
    """
    
    converge = False
    old_pred = P.dot(Q.T)
    while not converge:
        
        # update P
         for i in range(len(Train_IC)):
            P[i] = update_p(i,Train_IC, P, Q, K, SimC, landa, miu)
            
        # update Q
         for j in range(len(Train_IC[0])):
            Q[j] = update_q(j,Train_IC, P, Q, K, SimD, landa, miu)
        
        # computing the difference in predicted matrix of two subsequnt iterations 
         new_pred = P.dot(Q.T)
         difference = 0
         for i in range(len(Train_IC)):
             difference = difference + np.linalg.norm(old_pred[i] - new_pred[i])
         old_pred = new_pred
         
         # if the diffiernce in predicted matrix of two subsequnt iterations is 
         # less than 0.01, the method is converged
         if difference < 0.01:
             converge = True
             
    return new_pred, P, Q

#-----------------------------------------------------
def update_p(i, Train_IC, P, Q, K, SimC, landa, miu):
    """ This function updates the ith row of P matrix
    
        i is the number of row
        Train_IC is the training matrix 
        Q and P are the latent matrices
        K is the dimension of latent space
        SimC is the similarity matrices of cell lines
        landa controls the similarity conservation while manifold learning
        miu is the regularization coeeficient for latent matrices        
    """
    
    pi = Train_IC[i,:].dot(Q)
    pi = pi.reshape(1,K)
    
    xi2 = 0
    for j in range(len(SimC[i])):
        xi2 = numpy.dot(SimC[i,j] + SimC[j,i],(P[j])) + xi2

    pi = pi + numpy.dot(landa, (xi2))
    I = numpy.identity(K)
    xi3 = (numpy.dot((Q.T), Q) + numpy.dot(miu,I))
    
    xi4 = 0
    for j in range(len(SimC[i])):
        xi4 = (SimC[i,j] + SimC[j,i]) + xi4
    
    xi3 = xi3 + numpy.dot(landa * xi4, I)
    xi3 = inv(xi3)
    xi3 = xi3.reshape(K, K)
    final = pi.dot(xi3)
    
    return final
#end update xi---------------------------------------
        
    
    
#B(j,:)=(Y(:,j)'*A+alpha*(W_c(j,:)+(W_c(:,j))')*B)/(A'*A+alpha*sum(W_c(j,:)+(W_c(:,j))')*eye(f)+lamda*eye(f));
#Update yj---------------------------------------
def update_q(j, Train_IC, P, Q, K, SimD, landa, miu):
    """ This function updates the jth row of Q matrix
    
        i is the number of row
        Train_IC is the training matrix 
        Q and P are the latent matrices
        K is the dimension of latent space
        SimD is the similarity matrices of drugs
        landa controls the similarity conservation while manifold learning
        miu is the regularization coeeficient for latent matrices        
    """
    
    
    qj = (Train_IC[:, j].T).dot(P)
    qj = qj.reshape(1, K)
    
    xi2 = 0
    for i in range(len(SimD[j])):
       xi2 = numpy.dot((SimD[i,j] + SimD[j,i]), Q[i]) + xi2

    qj = qj + numpy.dot(landa, (xi2))
    I = numpy.identity(K)
    xi3 = (numpy.dot((P.T), P) + numpy.dot(miu, I))
    
    xi4 = 0 
    for j in range(len(SimD[j])):
        xi4 = (SimD[i,j] + SimD[j,i]) + xi4
    
    xi3 = xi3 + numpy.dot(landa * xi4, I)
    xi3 = inv(xi3)
    xi3.reshape(K, K)
    final = qj.dot(xi3)
    
    return final

    
def loss(IC_real, P, Q, SimC, SimD, landa, miu):
    """ This function computes the loss function in manifold learning
    
        IC-real is the real matrix of IC values
        P and Q are the latent matrices
        SimD and SimC are the similarity matrices of drugs and cell lines
        landa controls the similarity conservation while manifold learning
        miu is the regularization coeeficient for latent matrices 
    """
    
    
    row = IC_real.shape[0]
    col = IC_real.shape[1]

    diff = 0
    IC_pred = P.dot(Q.T)
    
    # the squraed error between real and predicted IC50
    for i in range(row):
        for j in range(col):
            diff = diff + (IC_real[i][j] - IC_pred[i][j])**2
    diff=diff / 2
    
    # computing the norm of P
    norm_P = 0
    for i in range(row):
        norm_P = norm_P + (numpy.linalg.norm(P[i]))**2
    diff = diff + (miu / 2) * norm_P

    # computing the norm of Q
    norm_Q = 0
    for i in range(col):
        norm_Q = norm_Q + (numpy.linalg.norm(Q[i]))**2
    diff = diff + (miu / 2) * norm_Q

    # computing the difference of similarities in the real and latent space
    reg1 = 0
    for i in range(row):
        for j in range(row):
            reg1 = reg1 + ((numpy.linalg.norm(P[i] - P[j]))**2) * SimC[i][j]
    reg2 = 0
    for i in range(col):
        for j in range(col):
            reg2 = reg2 + ((numpy.linalg.norm(Q[i]- Q[j]))**2) * SimD[i][j]
    diff = diff + (landa / 2) * reg1 + (landa / 2) * reg2

    return diff
