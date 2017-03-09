#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 23:27:40 2016

@author: hadoop
"""
import matplotlib.pylab as plt
import numpy as np 
from sprandn import sprandn
from l1_cvx_mosek import l1_cvx_mosek
#%%
l1_cvx_mosek(n,A,b,mu)
#%%
m = 1024
n = 512
mu = 1e-3

A = np.matrix(np.random.randn(m,n))
u = np.matrix(sprandn(n,1,0.1).todense())
b = np.matrix(np.dot(A,u))
#%%
def f_x(A_h,b,mu,x) :
    return 0.5*np.sqrt(np.sum(np.square(np.dot(A_h,x)-b))) + mu*np.sum(x)
def f_gra(A_h,b,mu,x):
    return np.dot(np.dot(A_h.T,A_h),x)-np.dot(A_h.T,b) + mu* np.ones((1024,1))
def pro_box(x) :
    x[x<0] = 0
    return x
#%%
x = np.matrix(np.random.randn(n*2,1))
A_h = np.hstack((A,-A))
#%%
path = []
step = 1e-4
for mu in [1e3,1e2,1e1,1e-1,1e-2,1e-3] :
    for i in range(50) :
        path.append(f_x(A_h,b,mu,x))
        x = pro_box(x - step*(f_gra(A_h,b,mu,x)))
    
while(abs(path[-1] - path[-9]) >1e-7) :
    path.append(f_x(A_h,b,mu,x))
    x = pro_box(x - step*(f_gra(A_h,b,mu,x)))
    
    
    
