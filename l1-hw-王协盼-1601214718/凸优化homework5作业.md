##                                                                     凸优化homework5

### 1 直接调用gurobi和mosek solver

在python下用cvxpy求解问题，solver调用gurobi

```python
from datetime import datetime
import numpy as np
from scipy.sparse import random
import cvxpy as cvx

def l1_cvx_gurobi(n,A,b,mu):
    x = cvx.Variable(n,1)
    exp =  0.5* cvx.square(cvx.norm((A*x-b),2)) + mu * cvx.norm(x,1)
    obj = cvx.Minimize(exp)
    pro =cvx.Problem(obj)
    start_time = datetime.now()
    pro.solve(solver ="GUROBI")
    end_time = datetime.now()
    print"used time :",(end_time-start_time).seconds
    print"prob status :" ,pro.status
    print"optimal value :", pro.value
    #print "optimal var :" ,x.value
```

```python
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
```

输出结果：cvx_gurobi : 20s

​                   cvx_mosek: 1s

### 2, 直接调用mosek和gurobi

#### 2.1 直接调用gurobi

原问题引入新的变量$y$ ,转化为等价的问题如下形式：
$$
min  \  \  \  1/2*||Ax-b||^2_2 +mu \ * ||y||_1 \\
subject \ to :  \    y_i =  |x_i|  \  \ i = 1,2,...,n

$$

```python
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
```

#### 2.2 直接调用mosek

因为python版的mosek有两种接口，一个是fusion，另一个是optimizer。前一个比较简洁，而optimizer偏底层。这里是把原问题转化为等价的QP问题，然后用optimizer求解：


$$
min \ 1/2 * ||A_1x_1 - b||_2^2 + mu \ ||x_1||_1 \\
subject \ to :  x_1 >= 0  \\
这里 : x_1 = (x^+,x^-) ^T是(2*n ,1) 维的  \\
|x|_i =x^+_i +x_i^ - \ \  i = 1,2,...,n\\
x_i = x^+_i - x^-_i   \  \ i =1,2,...,n
\\A_1 =(A,-A)
\\ mosek 需要显示输入Q 和c，c_0
$$

```python
Q =np.dot(np.matrix(A).T,np.matrix(A))
c = np.dot(np.matrix(A).T,b)
c_0 = (np.matrix(b).T * b)[0,0]
```

```python
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
inf = 0.0

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
```

输出结果的时间：

### 3,projection_gradient and subgradient 

##### 3.1projection gradient 

利用前面2.2的形式转化为等价的凸问题：
$$
min \ 1/2 * ||A_1x_1 - b||_2^2 + mu \ *||x_1||
\\ subject\ to :x_1 >= 0
$$

```python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 23:27:40 2016

@author: hadoop
"""
import numpy as np 
#%%
def f_x(A_h,b,mu,x) :
    return 0.5*np.sum(np.square(np.dot(A_h,x)-b)) + mu*np.sum(x)
def f_gra(A_h,b,mu,x):
    return np.dot(np.dot(A_h.T,A_h),x)-np.dot(A_h.T,b) + mu* np.ones((2048,1))
def pro_box(x) :
    x[x<0] = 0.0
    return x
#%%
def project_gradient (A,b,mu,n) :
    from datetime import datetime 
    x = np.matrix(np.random.randn(n*2,1))
    A_h = np.hstack((A,-A))
    path = []
    step = 1e-4
    start_time = datetime.now()
    for mu in [1e3,1e2,1e1,1e-1,1e-2,1e-3] :
        for i in range(200) :
            path.append(f_x(A_h,b,mu,x))
            x = pro_box(x - step*(f_gra(A_h,b,mu,x)))      
    while(abs(path[-1] - path[-7]) > 1e-7) :
        path.append(f_x(A_h,b,mu,x))
        x = pro_box(x - step*(f_gra(A_h,b,mu,x)))
    end_time = datetime.now()
    print "used time :",(end_time-start_time).seconds
    print "optimal" ,path[-1] 
```

##### 3.2  subgradient method

计算原问题目标函数的次梯度，其中1范数的次梯度如下：

![次梯度](C:\Users\wxp\Desktop\次梯度.PNG)

```python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 16:58:48 2016

@author: hadoop
"""
import numpy as np 
from datetime import datetime
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
```

### 4.总结：

次梯度方法和投影梯度法，选的都是固定步长，投影梯度法能较快的收敛到最优解；

次梯度算法能不断下降，能够收敛到接近最优解的地方。

进一步需要改进的地方：选用BB步长。

