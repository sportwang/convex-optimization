#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 04:34:06 2016

@author: hadoop
"""

from gurobipy import *

def l1_gurobi( cols, c, Q, u,mu):

  model = Model()

  # Add variables to model
  vars = []
  for j in range(cols):
      vars.append(model.addVar(lb=-GRB.INFINITY, ub =GRB.INFINITY, vtype=GRB.CONTINUOUS))
  abs_var = []
  for j in range(cols):
      abs_var.append(model.addVar(lb=-GRB.INFINITY, ub =GRB.INFINITY, vtype=GRB.CONTINUOUS))
      
  

  # Populate objective
  obj = QuadExpr()
  #first xQx
  for i in range(cols):
    for j in range(cols):
      if Q[i,j] != 0:
        obj += 0.5*Q[i,j]*vars[i]*vars[j]
   #xAb
  for j in range(cols):
    if c[j] != 0:
        obj += c[j]*vars[j]
    obj += mu*abs_var[j]
  obj += 0.5*u
        
  
  model.setObjective(obj)
  
  #add constrains
  for j in range(cols):
      model.addGenConstrAbs(abs_var[j],vars[j],'absconstr')

  # Solve
  model.optimize()

  # Write model to a file
  #model.write('dense.lp')

  if model.status == GRB.Status.OPTIMAL:
    return True
  else:
    return False
    