#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 23:31:52 2016

@author: hadoop
"""

import sys
import mosek
import numpy as np

# Since the actual value of Infinity is ignores, we define it solely
# for symbolic purposes:
#inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

# We might write everything directly as a script, but it looks nicer
# to create a function.
def l1_mosek (n,c,Q,u,mu):
    with mosek.Env () as env:
        env.set_Stream (mosek.streamtype.log, streamprinter)
        
        with env.Task(0,0) as task:
            task.set_Stream (mosek.streamtype.log, streamprinter)
            numvar = n*2
            P = np.bmat([[Q,-1*Q],[-1*Q,Q]])
            C1 = mu + np.bmat([[-1*c],[c]])
            c0 = 0.5*u
            
            
            bkx   = [ mosek.boundkey.lo]*numvar
            blx   = [ 0.0]*numvar
            bux   = [ inf]*numvar

            task.appendvars(numvar)
            for j in range(numvar):
                task.putcj(j,C1[j])
                task.putbound(mosek.accmode.var,j,bkx[j],blx[j],bux[j])
            for i in range(numvar):
                for j in range(i+1):
                    task.putqobjij(i,j,P[i,j])
            task.putcfix(c0)
            task.putobjsense(mosek.objsense.minimize)
            print 'optimize start *****************'
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)