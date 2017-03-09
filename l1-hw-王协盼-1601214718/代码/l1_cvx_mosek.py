#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 04:42:26 2016

@author: hadoop
"""

from datetime import datetime
import numpy as np
from scipy.sparse import random
import cvxpy as cvx

def l1_cvx_mosek(n,A,b,mu) :
    x = cvx.Variable(n,1)
    exp =  0.5* cvx.square(cvx.norm((A*x-b),2)) + mu * cvx.norm(x,1)
    obj = cvx.Minimize(exp)
    pro = cvx.Problem(obj)
    start_time = datetime.now()
    pro.solve(solver='MOSEK')
    end_time = datetime.now()
    print"used time :",(end_time-start_time).seconds
    print"prob status :" ,pro.status
    print"optimal value :", pro.value
    #print "optimal var :" ,x.value