# -*- coding: utf-8 -*-
"""
Created on Tue Jan 03 10:55:56 2017

@author: wxp
"""
from __future__ import division
import numpy as np
import time
np.random.seed(1342)
n=100
m=20
A = np.random.randn(m,n)
xs = np.abs(np.random.randn(n,1) * np.random.binomial(1, m/n, (n,1)))
b = A.dot(xs)
y =np.random.randn(m,1)
s =np.multiply(np.random.rand(n,1),(xs==0))
c = A.T.dot(y) + s
#%%
import cvxpy as cy
mosek_x = cy.Variable(n)
objective = cy.Minimize(c.T * mosek_x)
constraints = [0 <= mosek_x, A*mosek_x == b]
prob = cy.Problem(objective, constraints)

#%%
class Alm () :
    def __init__(self,A=A,b=b,s=s,c=c,err=prob.solve()) :
        self.A = A
        self.b = b
        self.s = s
        self.c = c
        self.err = err
    def solve(self,x_0,y_0,p = 0.2):
        x = x_0
        y = y_0
        step = 0.001
        start = time.time()
        for i in range(10000) :
            tol_1 = self.c.T.dot(x)
            proj = x + p * (self.A.T.dot(y)-self.c)
            for j in range(200) :
                py = - self.b + self.A.dot(proj)
                y -= step * py
                proj = x + p * (self.A.T.dot(y)-self.c)
                proj[proj<0] = 0
                
            x = x + p * (self.A.T.dot(y) - self.c)
            x[x<0] = 0
            tol_2 = self.c.T.dot(x)
            if np.abs(tol_1-tol_2) < 1e-8:
                print 'ALM iteration times:',i
                break
        
        print 'Alm：objective value :',self.c.T.dot(x)
        print 'Alm sum(abs(x-xs)):',np.sum(np.abs(x-xs))
        print 'ALm sum(abs(Ax-b)) erro:',np.sum(np.abs(self.A.dot(x) - self.b))
        print 'Alm sum(x[x<0]) erro:',np.sum(x[x<0]),np.sum(x<0)
        print 'Alm used time:',time.time() - start
        print '*****************************************'

#%%
#class semi_smooth_cg(Alm) :
    
#%%

class Admm(Alm):
    def solve(self,x_0,y_0,s_0,p = 0.4) :
        x= x_0
        y,s = y_0,s_0
        cc = 1/p * np.linalg.inv(self.A.dot(self.A.T))
        start = time.time()
        for i in range(10000) :
            tol_1 = self.c.T.dot(x)
            y = np.dot(cc , self.b - self.A.dot(x)-p*self.A.dot(s) + p*self.A.dot(self.c)) 
            s = -x/p + c -self.A.T.dot(y)
            s[s<0.0] = 0
            x = x + p * (self.A.T.dot(y) + s -c)
            tol_2 = self.c.T.dot(x)
            if np.abs(tol_1-tol_2) < 1e-8:
                print 'ADMM iteration times:',i
                break
    
        print 'ADMM：objective value :',self.b.T.dot(y)
        print 'ADMM sum(abs(x-xs)):',np.sum(np.abs(x-xs))
        print 'ADMM sum(abs(Ax-b)) erro:',np.sum(np.abs(self.A.dot(x) - self.b))
        print 'ADMM sum(x[x<0]) erro:',np.sum(x[x<0]),np.sum(x<0)
        print 'ADMM used time:',time.time() - start
        print '*****************************************'

#%%  
class Drs(Alm) :
    def solve(self,x_0,p=0.4) :
        x ,w= x_0,x_0
        cc = self.A.T.dot(np.linalg.inv(self.A.dot(self.A.T)))
        start = time.time()
        for i in range(10000) :
            tol_1 = self.c.T.dot(x)
            u = (-self.c*p+x+w)+np.dot(cc,self.b-self.A.dot(-self.c*p+x+w))
            x =  u - w
            x[x<0.0] = 0
            tol_2 = self.c.T.dot(x)
            w = w + x - u
            if np.abs(tol_1-tol_2) < 1e-8:
                print 'DRS iteration times:',i
                break
       
        print 'DRS：objective value :',self.c.T.dot(x)
        print 'DRS sum(abs(x-xs)):',np.sum(np.abs(x-xs))
        print 'DRS sum(abs(Ax-b)) erro:',np.sum(np.abs(self.A.dot(x) - self.b))
        print 'DRS sum(x[x<0]) erro:',np.sum(x[x<0]),np.sum(x<0)
        print 'DRS used time:',time.time() - start
        print '*****************************************'
#%%
class l1_semi_smooth_newton() :
    def __init__(self,A=A,b=b,c=c,e1=0.1,e2=0.9,lm_ = 0.01,g1=1.5,g2=10.0,v=0.5,t = 0.1,err=prob.solve()) :
        self.A = A
        self.b = b
        self.c = c
        self.e1 = e1
        self.e2 = e2
        self.lm_ = lm_
        self.g1 = g1
        self.g2 = g2
        self.v = v
        self.t = t
        self.err = err
        
        self.aaa = self.A.T.dot(np.linalg.inv(self.A.dot(self.A.T)))
        self.PAT = self.aaa.dot(self.A)
        self.I = np.eye(100) * 1.0
        self.D = self.I -self.PAT
        self.beta = self.aaa.dot(b)
        self.D_I = self.D * 2.0 - self.I

    def F(self,x) :
        cc = x - self.c * self.t
        cc[cc<0.] = 0.
        zz = 2.0 * cc - x
        return cc - self.D.dot(zz) - self.beta
    def J(self,x) :
        cc = x - self.c * self.t
        zz = np.zeros_like(cc)
        zz[cc>= 0.0] = 1.0
        Mz = np.diag(zz[:,0])
        return  Mz - self.D.dot(2.0 * Mz - self.I)
    def update_lm(self,lmk,rho) :
        if rho >= self.e2 :
            return  (self.lm_ + lmk) / 2.0
        elif rho >= self.e1:
            return (lmk + self.g1 * lmk) / 2.0
        else :
            return (self.g1 * lmk + self.g2 * lmk) / 2.0
    def solve(self,x_0) :
        x = x_0
        u = x_0.copy()
        u_ = x_0.copy()
        lmk = 1.5
        start = time.time()
        for i in range(10000) :
            Fk = self.F(x)
            miu_k = lmk * np.linalg.norm(Fk)
            dk = np.dot(np.linalg.inv(self.J(x) + miu_k * self.I ),-1.0 * Fk)
            u = x + dk
            Fu = self.F(u)
            rho = -1.0 * np.dot(Fu.T,dk)[0,0] / (np.linalg.norm(dk) ** 2)
            condition = np.linalg.norm(Fu) <= self.v * np.linalg.norm(self.F(u_))
            if rho >= self.e1  :
                x = u 
                u_ = u 
            elif rho >= self.e1 and not condition :
                x = x - (Fu.T.dot(x - u )[0,0] / (np.linalg.norm(Fu) ** 2) ) * Fu
            else :
                pass
            lmk = self.update_lm(lmk,rho)
            cc = x - self.c * self.t
            cc[cc<0.] = 0.
            if (np.abs(self.err - self.c.T.dot(cc)) < 1e-5) :
                print 'l1_semi_smooth iteration times:',i
                break
        x = x - self.c * self.t
        x[x<0.] = 0.
        print "l1_semi_smooth objective value :",self.c.T.dot(x)
        print 'l1_semi_smooth sum(abs(x-xs)):',np.sum(np.abs(x-xs))
        print 'l1_semi_smooth sum(abs(Ax-b)) erro:',np.sum(np.abs(self.A.dot(x) - self.b))
        print 'l1_semi_smooth sum(x[x<0]) erro:',np.sum(x[x<0]),np.sum(x<0)
        print 'l1_semi_smooth used time :',time.time() - start
        print '******************************************************************'
#%%
class newton_cg():
    def __init__(self, c=c, A=A, b=b, t=0.4, err = prob.solve(),iteration = 10):
        self.c = c
        self.A = A
        self.b = b
        self.m, self.n = self.A.shape
        self.iteration = iteration
        self.cov = np.dot(self.A, self.A.T)
        self.err = err
        
        self.t = t
    def L_t(self, y, x):
        l = - np.dot(self.b.T, y)
        r1_inner = self.t * (np.dot(self.A.T, y) - self.c) + x
        r1_inner[r1_inner<=0.] = 0.
        r = 1.0 / (2 * self.t) * (np.linalg.norm(r1_inner) ** 2 - np.linalg.norm(x) ** 2)
        return l[0,0] + r
    def NCG(self, y, x, delta):
        mu = 0.25
        tau1 = 0.5
        tau2 = 0.5
        for j in range(3):
            inner_proj = self.t * np.dot(self.A.T, y) - self.t * self.c + x
            dia = 1.0 * np.zeros_like(inner_proj)
            dia[inner_proj>0.] = 1.
            V_j = self.t * np.dot(self.A, np.dot(np.diag(dia[:,0]), self.A.T))
            projed = inner_proj[:]
            projed[inner_proj<0.] = 0.
            delta_phi = -self.b + np.dot(self.A, projed)
           
            epsilong_j = tau1 * np.min([tau2, np.linalg.norm(delta_phi)])
            d = - np.dot(np.linalg.inv(V_j + epsilong_j * np.eye(self.m)), delta_phi) 
            i = 0
            cond = self.L_t(y+(delta**i) * d, x) <= self.L_t(y, x)+ mu*(delta**i)*np.dot(delta_phi.T, d)
            while not cond and i <= self.iteration:
                i += 1
                cond = self.L_t(y+(delta**i) * d, x) <= self.L_t(y, x)+mu*(delta**i)*np.dot(delta_phi.T, d)[0,0] 
            y = y + (delta**i) * d
        return y
    def update(self, y, x, delta, delta_0 = 0.8, rho=1.1):
        y = self.NCG(y, x, delta)
        x_inner = x + self.t * (np.dot(self.A.T, y) - self.c)
        x_inner[x_inner<=0.] = 0.
        return y, x_inner, delta
    def solve(self,x_0,y_0):
        import time
        start_time = time.time()
        self.y = y_0
        self.x = x_0
        self.delta = 0.5
        self.condition = np.sum((np.dot(self.A, self.x) - self.b) ** 2)
        self.solve = np.dot(self.c.T, self.x)
       
        for i in range(1000):
            self.y, self.x, self.delta = self.update(self.y, self.x, self.delta)
            self.condition = np.sum((np.dot(self.A, self.x) - self.b) ** 2)
            self.solve = np.dot(self.c.T, self.x)
            if (np.abs(self.solve-self.err)<1e-8) :
                print 'newton_cg iterations :', i
                break
        self.run_time = time.time() - start_time
        
    
        


#%%
if __name__=='__main__' :
    print 'MOSEK objective:',prob.solve()
    print 'MOSEK sum(abs(x-xs)):',np.sum(np.abs(mosek_x.value-xs))
    print 'MOSEK sum(abs(Ax-b)) erro:',np.sum(np.abs(A.dot(mosek_x.value) - b))
    print 'MOSEK sum(x[x<0]) erro:',np.sum(mosek_x.value[mosek_x.value<0]),np.sum(mosek_x.value<0)
    print '*****************************************'
    y_0 = np.random.randn(m,1)
    x_0 = np.random.randn(n,1)
    s_0 = x_0.copy()
    
    model = Alm()
    model.solve(x_0,y_0)
            
    
    Admm().solve(x_0,y_0,s_0)
    
    Drs().solve(x_0)
    
    l1_semi_smooth_newton().solve(x_0)
    new = newton_cg()
    new.solve(x_0,y_0)
    print "newton_cg objective value :",new.solve
    print 'newton_cg sum(abs(x-xs)):',np.sum(np.abs(new.x-xs))
    print 'newton_cg sum(abs(Ax-b)) erro:',np.sum(np.abs(A.dot(new.x) - b))
    print 'newton_cg sum(x[x<0]) erro:',np.sum(new.x[new.x<0]),np.sum(new.x<0)     
    print 'newton_cg used time :',new.run_time
    print '******************************************************************'