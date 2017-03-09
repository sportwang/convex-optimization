#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:58:48 2016

@author: hadoop
"""
import numpy as np 
from datetime import datetime
#%%
def func_x (A,x,b,mu) :
    return 0.5*np.sum(np.square(np.dot(A,x)-b)) + mu*np.sum(np.abs(x))
def f_subgra(A,x,b,mu,n) :
    tol = 1e-5
    gra =  np.zeros((n,1))
    for  i in range(n) :
        if x[i,0] >tol :
            gra[i,0] = 1.0
        elif x[i,0] < -tol :
            gra[i,0] = -1.0
        else :
            gra[i,0] = np.random.uniform(-1,1)
    return  mu * gra + np.dot(np.dot(A.T, A),x) - np.dot(A.T,b)  
#%%
def subgradient (A,b,mu,n) :
    x = np.zeros((n,1))
    step = 6e-4
    path = []
    start_time = datetime.now()
    for mu in [1e3, 1e2, 1, 1e-2, 1e-3]:
        for  i in range(500) :
            path.append(func_x(A,x,b,mu))
            print func_x(A,x,b,mu)
            x = x - step * f_subgra(A,x,b,mu,n) 
            
    while(abs(path[-1] - path[-9]) > 1e-7) :   
        path.append(func_x(A,x,b,mu))
        print func_x(A,x,b,mu)
        x = x -step * f_subgra(A,x,b,mu,n)
    end_time = datetime.now()
    print "used time :",(end_time-start_time).seconds
    print "optimal" ,path[-1] 
        
        