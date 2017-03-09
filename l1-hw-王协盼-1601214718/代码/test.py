#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 04:38:12 2016

@author: hadoop
"""

from sprandn import sprandn
import numpy as np
from l1_cvx_gurobi import l1_cvx_gurobi
from l1_cvx_mosek import l1_cvx_mosek
from l1_gurobi import l1_gurobi
from l1_mosek import l1_mosek
from project_gradient import project_gradient
from subgradient import subgradient 
#%%
m = 512
n = 1024
mu = 1e-3

A = np.random.randn(m,n)
u = sprandn(n,1,0.1).todense()
b = np.dot(A,u)
#%%
l1_cvx_gurobi(n,A,b,mu)
#%%
l1_cvx_mosek(n,A,b,mu)
#%%
Q =np.dot(np.matrix(A).T,np.matrix(A))
c = np.dot(np.matrix(A).T,b)
u = (np.matrix(b).T * b)[0,0]
#%%
l1_gurobi(n,c,Q,u,mu)
#%%
l1_mosek(n,c,Q,u,mu)
#%%