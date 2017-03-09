#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 04:03:12 2016

@author: hadoop
"""

from sprandn import sprandn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 



class l1_square () :
    def __init__(self,A,b,m,n,mu,acce_path,acce_num,max_iteration,tol) :
        self.A = A
        self.b = b
        self.m = m
        self.n = n
        self.mu = mu 
        self.acce_path = acce_path
        self.num_acce = acce_num
        self.max_iteration = max_iteration
        self.tol = tol#gradient epsilo;stop citerial
        self.ATA = np.dot(self.A.T,self.A)
        self.ATb = np.dot(self.A.T,self.b)
        #step size in pro_gra
        self.step = 1.0 / np.linalg.svd(self.ATA,compute_uv=False)[0]
        #m,L in fast_pro_gra
        self.svd_m =  np.linalg.svd(self.ATA,compute_uv=False)[n-1]
        self.svd_L =  np.linalg.svd(self.ATA,compute_uv=False)[0]
        self.momentume = (1.0-np.sqrt(self.svd_m / self.svd_L)) / (1.0+np.sqrt(self.svd_m / self.svd_L))
    def show_path(self,axis) :
        plt.figure(figsize=(8,8))
        m = axis[0]
        n = axis[1]
        plt.plot(self.path_1[m:n], color='blue',label = "proxiaml gradient")
        plt.plot(self.path_2[m:n],color='red',label ="fast proximal gradient")
        plt.plot(self.path_3[m:n],color='black',label = "smoothed gradient")
        plt.plot(self.path_4[m:n],color = 'yellow',label = "fast smoothed gradient")
        plt.axis(axis)
        plt.legend(loc='upper right')
    def f_x (self,x) :
        return 0.5 * np.sum(np.square(np.dot(self.A,x) - self.b)) + self.mu * np.sum(np.abs(x))
    def pro_operation(self,u,mu_var) :
        self.shrinkage = self.step * mu_var
        for i in range(u.shape[0]) :
            if u[i,0] >= self.shrinkage :
                u[i,0] -= self.shrinkage
            elif u[i,0] <= -self.shrinkage :
                u[i,0] += self.shrinkage
            else :
                u[i,0] = 0
        return u

    def pro_gra(self,x_0,path) :
        self.path_1 = []
        from datetime import datetime
        start_time = datetime.now()
        x = x_0
        y = x_0-self.step * (np.dot(self.ATA,x_0) -self.ATb)
        y = self.pro_operation(y,self.mu)
        # fast at the near optimal x*    
        if path > 0 :
            for lamda in self.acce_path :
                print "lamda :*****************", lamda
                for i in range(self.num_acce) :
                    self.path_1.append(self.f_x(x))
                    print self.f_x(x)
                    x = y
                    y = x-self.step * (np.dot(self.ATA,x) -self.ATb)
                    y = self.pro_operation(y,lamda)
                    
        op = 1
        while(np.sum(np.abs(y-x)) > tol and op < self.max_iteration) :
            self.path_1.append(self.f_x(x))
            print self.f_x(x)
            x = y
            y = x-self.step * (np.dot(self.ATA,x) -self.ATb)
            y = self.pro_operation(y,self.mu)
            op += 1
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
    def fast_pro_gra(self,x_0,path) :
        self.path_2 = []
        from datetime import datetime
        start_time = datetime.now()
        x = x_0
        x_1 = x_0
        y = x_0-self.step * (np.dot(self.ATA,x_0) -self.ATb)
        # fast at the near optimal x* 
        if path > 0 :
            for lamda in self.acce_path :
                print "lamda :*****************", lamda
                for i in range(self.num_acce) :
                    self.path_2.append(self.f_x(x))
                    print self.f_x(x)
                    y = x + self.momentume * (x - x_1)
                    g = y - self.step * (np.dot(self.ATA,y) -self.ATb)
                    x_1 = x
                    x = self.pro_operation(g,self.mu)
        
        op =1
        while(np.sum(np.abs(y-x)) > tol and op < self.max_iteration) :
            self.path_2.append(self.f_x(x))
            print self.f_x(y)
            y = x + self.momentume * (x - x_1)
            g = y - self.step * (np.dot(self.ATA,y) -self.ATb)
            x_1 = x
            x = self.pro_operation(g,self.mu)
            op += 1
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
    def f_smo_gra(self,x,beta,mu_var) :
        #huber penalty
        g = np.random.randn(self.n,1)
        for i in range(x.shape[0]) :
            if np.abs(x[i,0]) <= beta :
                g[i,0] = x[i,0] / beta
            elif x[i,0] > beta :
                g[i,0] = 1
            else :
                g[i,0] = -1
        return np.dot(self.ATA,x) - self.ATb + mu_var * g
        
  
    def smooth_gra(self,x_0,acuracy,path) :
        self.path_3 = []
        from datetime import datetime 
        start_time = datetime.now()
        x = x_0
        y = x - self.step * self.f_smo_gra(x,acuracy,self.mu)
        # fast at the near optimal x* 
        if path > 0 :
            for lamda in self.acce_path :
                print "lamda :*****************", lamda
                for i in range(self.num_acce) :
                    self.path_3.append(self.f_x(x))
                    print self.f_x(x)
                    x = y
                    y = x - self.step * self.f_smo_gra(x,acuracy,self.mu)
                    
        op = 1
        while(np.sum(np.square(self.f_smo_gra(x,acuracy,self.mu))) > self.tol and op < self.max_iteration) :
            self.path_3.append(self.f_x(x))
            print self.f_x(x)
            x = y
            y = x - self.step * self.f_smo_gra(x,acuracy,self.mu)
            op +=1
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
    
    def fast_smoo_gra(self,x_0,acuracy,path) :
        self.path_4 = []
        start_time = datetime.now()
        x = x_0
        y = x
        
        # fast at the near optimal x* 
        k = 2
        if path > 0 :
            for lamda in self.acce_path :
                print "lamda :*****************", lamda
                for i in range(self.num_acce) :
                    self.path_4.append(self.f_x(x))
                    print self.f_x(x)
                    x_1 = x
                    x = y - self.step * self.f_smo_gra(y,acuracy,self.mu)
                   
                    #y = x +  self.momentume *(x - x_1)
                    y = x + (k-1) / (k+2) *(x - x_1)
                    k += 1.0
        op = 1
        while(np.sum(np.square(self.f_smo_gra(x,acuracy,self.mu))) > self.tol and op < self.max_iteration) :
            self.path_4.append(self.f_x(x))
            print self.f_x(x)
            x_1 = x
            x = y - self.step * self.f_smo_gra(y,acuracy,self.mu)
            #y = x +  self.momentume*(x - x_1)
            y = x + (k-1) / (k+2) *(x - x_1)
            k += 1.0
            op += 1
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
        
    def shrinkage(self,u,mu) :
        y = np.random.randn(self.n,1)
        for i in range(self.n) :
            if u[i,0] > mu :
                y[i,0] = u[i,0] - mu 
            elif u[i,0] < -mu :
                y[i,0] = u[i,0] + mu
            else :
                y[i,0] = 0.0
        return y
        
    def admm_primal (self,x_0) :
        x = x_0
        y = np.ones((self.n,1))
        c = 1.0
        start_time = datetime.now()
        path = []
        ww = np.linalg.inv(self.ATA + c*np.identity(self.n))
        for i in range(self.max_iteration) :
            print(self.f_x(x))
            path.append(self.f_x(x))
            z = self.shrinkage((x+(y/c)),self.mu/c)
            x = np.dot(ww,self.ATb - y + c * z)
            y = y + c * (x - z)
            if i>2 and abs(path[i] - path[i-1]) < self.tol :
                break
        #plt.figure(figsize=(8,8))
        plt.plot(path,color='red',label ="ADMM")
        plt.legend(loc='upper right')
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
        
        
    def admm_primal_linear (self,x_0) :
        x = x_0
        y = np.ones((self.n,1))
        c = 1.0
        start_time = datetime.now()
        path = []
        t=2.0
        ww = np.linalg.inv(t * self.ATA + np.identity(self.n))
        for i in range(self.max_iteration) :
            print(self.f_x(x))
            path.append(self.f_x(x))
            z = self.shrinkage((x+(y/c)),self.mu/c)
            g = y + c*(x-z)
            x = np.dot(ww,t*self.ATb +x-t*g)
            y = y + c * (x - z)
            if i>2 and abs(path[i] - path[i-1]) < self.tol :
                break
        #plt.figure(figsize=(8,8))
        plt.plot(path,color='red',label ="ADMM")
        plt.legend(loc='upper right')
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
        
    def admm_var_gra (self,x_0) :
        x = x_0
        y = np.ones((self.n,1))
        c = 10
        ###
        step = 1e-4
        start_time = datetime.now()
        path = []
        for i in range(self.max_iteration) :
            print(self.f_x(x))
            path.append(self.f_x(x))
            z = self.shrinkage((x+(y/c)),self.mu/c)
            ###different
            x = x - step * (np.dot(self.A.T,(np.dot(self.A,x)-self.b)) + y + c * (x - z))
            y = y + c * (x - z)
            '''
            if i>2 and abs(path[i] - path[i-1]) < self.tol :
                break 
            '''
        '''
        plt.figure(figsize=(8,8))
        plt.plot(path,color='blue',label ="ADMM_linear")
        plt.legend(loc='upper right')
        '''
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
        
        
    def projection(self,z,mu) :
        y = 1.0 * z[:]
        '''
        for i in range(z.shape[0]):
            if z[i,0] >  mu:
                y[i,0] = mu
            elif z[i,0] < -mu :
                y[i,0] = -mu
            else :
                y[i,0] = z[i,0]
        '''
        y[z>mu] = mu * 1.0
        y[z<-mu] = -1.0 * mu
        
        return y
            
    def admm_dual (self,x_0) :
        x = x_0
        y = np.zeros((self.m,1))
        c = 1.0 
        path = []
        start_time = datetime.now()
        mm =np.linalg.inv(np.identity(self.m) + c *  np.dot(self.A,self.A.T))
        for i in range(self.max_iteration) :
            print(self.f_x(x))
            path.append(self.f_x(x))
            
            z = self.projection(-1.0*(x/c)+self.A.T.dot(y),self.mu)
            y = np.dot(mm,-b+np.dot(self.A,c*z+x))
            x = x+c*(-self.A.T.dot(y)+z)
            
            '''
            z = self.projection(-1.0*((x/c)+self.A.T.dot(y)),self.mu)
            y = np.dot(mm,b+np.dot(self.A,-c*z-x))
            x = x+c*(self.A.T.dot(y)+z)
            '''
            '''
            z = self.projection((x/c)+self.A.T.dot(y),self.mu)
            y = np.dot(mm,b+np.dot(self.A,c*z-x))
            x = x+c*(self.A.T.dot(y)-z)
            '''
            #问题代码
            '''
            z = self.projection(-1.0*(x/c)+self.A.T.dot(y),self.mu)
            y = np.dot(mm,-b+np.dot(self.A,c*z-x))
            x = x+c*(self.A.T.dot(y)-z)
            '''
            
            
            if i>2 and abs(path[i] - path[i-1]) < self.tol :
                break
        '''    
        plt.figure(figsize=(8,8))
        plt.plot(path,color='red',label ="ADMM_dual")
        plt.legend(loc='upper right')
        '''
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
    def admm_alm(self,x) :
        x = x_0
        y = np.zeros((self.m,1))
        c = 1.0 
        path = []
        start_time = datetime.now()
        mm =np.linalg.inv(np.identity(self.m) + c *  np.dot(self.A,self.A.T))
        for i in range(self.max_iteration) :
            print(self.f_x(x))
            path.append(self.f_x(x))
            j =1
            while (j<20) :
                z = self.projection((x/c)+self.A.T.dot(y),self.mu)
                y = np.dot(mm,b+np.dot(self.A,c*z-x))
                j += 1
            x = x+c*(self.A.T.dot(y)-z)
           
            '''
            z =self.projection(x-self.A.T.dot(y),self.mu)
            y = np.dot(np.linalg.inv(np.identity(self.m)+c*np.dot(self.A,self.A.T)),c*np.dot(self.A,(x-z))-b)
            x = z - 0.0001*(np.dot(self.A.T,y)+z)
            '''
            
            if i>2 and abs(path[i] - path[i-1]) < self.tol :
                break
        '''   
        plt.figure(figsize=(8,8))
        plt.plot(path,color='red',label ="ADMM_dual")
        plt.legend(loc='upper right')
        '''
        end_time = datetime.now()
        print"used time :",(end_time-start_time).seconds
        
if __name__ == "__main__" :
    m = 512
    n = 1024
    mu = 1e-3

    A = np.random.randn(m,n)
    u = sprandn(n,1,0.1).todense()
    b = np.dot(A,u)
    tol = 1e-7
    x_0 = np.zeros((n,1))
    accepath = [1e3,1e2,1e1,1e-1,1e-2,1e-3]
    accenum = 50
    maxiteration =8000
    model = l1_square(A,b,m,n,mu,accepath,accenum,maxiteration,tol)
    t= 6
    #model.admm_primal(x_0)
    #model.admm_primal_linear(x_0)
    model.admm_dual(x_0)
    #model.admm_alm(x_0)
    



        
        
        
        
       
        
        
