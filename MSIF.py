# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:52:51 2020

Multi-source integration fusion simulation

"""

import numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def f_theo(x):           #Theoretical curve
    return (np.sin(x))
def f_scan(x):           #Add system error
    return f_theo(x) + 0.2*np.sin(x/3)

def get_d(x, data):      #Get the probability density 
    mu=np.mean(data)
    std=np.std(data)       
    y=[stats.norm(mu,std).pdf(a) for a in x]
    return y

Noise_s = 0.1
Noise_t = 0.01
N = 20                        #Times of experiments
N_s = 3                       #Number of scanning data
Ns,Nt,N_test = 100,9,75       #Number of scanning/target points
xmin,xmax = -5,5

#Define test data
x_test = np.linspace(xmin,xmax,N_test)
x_test = np.atleast_2d(x_test).T
y_test=f_theo(x_test).ravel()

#Define target data
x_target = np.linspace(xmin+0.6,xmax-0.6,Nt)
x_target = np.atleast_2d(x_target).T
y_target = f_theo(x_target).ravel()
noise = np.random.normal(0,Noise_t,y_target.shape)
y_target += noise

#Devide target data into two parts(for stacking)
x_target1 = x_target[::2]
x_target2 = x_target[1::2]
y_target1 = y_target[::2]
y_target2 = y_target[1::2]

MSE_S = []
MSE_W = []
MSE_RA = []
MSE_IF = []

for j in range(N):           #N times of experiments
    Yt=[0 for x in range(0, len(x_test))]
    W=[]
    X_scan=[0 for x in range(Ns)]
    Y_scan=[0 for x in range(Ns)]
    Ys_target=[0 for x in range(Nt)]
    Ys_test=[0 for x in range(N_test)]
    
    for i in range(3):       #RA separately
        #Define scanning data
        x_scan = [xmin+(xmax-xmin)/Ns*(i+np.random.rand()) for i in range(Ns)]
        x_scan = np.array(x_scan)
        x_scan = np.atleast_2d(x_scan).T
        y_scan=f_scan(x_scan).ravel()
        noise = np.random.normal(0,Noise_s,y_scan.shape)
        y_scan += noise 
        #Fit and prediction using low-precision(scanning) data set 
        kernel1 = C(1, (1e-3, 10000)) * RBF(5, (3, 10))
        model1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=10)
        model1.fit(x_scan,y_scan)
        ys_target_i, _ = model1.predict(x_target, return_std=True)#Prediction value at x_target
        ys_test_i, _ = model1.predict(x_test, return_std=True)    #Prediction value at x_test
        
        '''For reverse'''
        #Record the scanned surface of each prediction
        Ys_target = np.c_[np.array(Ys_target),np.array(ys_target_i)]
        Ys_test = np.c_[np.array(Ys_test),np.array(ys_test_i)]
        
        '''For weighted intergration'''
        res_i = y_target - ys_target_i                          #Residual
        kernel2 = C(1.0, (1e-3, 10000)) * RBF(10, (2, 1e1))
        model2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=10)
        model2.fit(x_target,res_i)
        res_t_i, _ = model2.predict(x_test, return_std=True)     
        yt_i = ys_test_i+ res_t_i                                   
        #Error and loss 
        E = abs(yt_i - y_test)                               
        mes_ra = np.array(E).sum()/len(yt_i)
        L = np.array(E**2).sum()
        MSE_RA.append(mes_ra)
        W.append(1/L)                                           #Weights
        Yt=np.c_[np.array(Yt),np.array(yt_i)]
        
        '''For stacking'''
        X_scan = np.c_[np.array(X_scan),np.array(x_scan)]
        Y_scan = np.c_[np.array(Y_scan),np.array(y_scan)]
        
      
    '''Weighted intergration'''
    yt=(W[0]*Yt[:,1]+W[1]*Yt[:,2]+W[2]*Yt[:,3])/(W[0]+W[1]+W[2])
    err1 = abs(yt-y_test)                                  
    MSE_W.append(np.array(err1).sum()/len(err1))           
    
    '''Reverse'''    
    # Integrate low-precision surfaces
    ys_target = (Ys_target[:,1]+Ys_target[:,2]+Ys_target[:,3])/3
    ys_test = (Ys_test[:,1]+Ys_test[:,2]+Ys_test[:,3])/3    
    res = y_target- ys_target                                
    model2.fit(x_target,res)
    res_t, _ = model2.predict(x_test, return_std=True)        
    yt= ys_test + res_t                                       
    err = abs(yt-y_test)                                   
    MSE_IF.append(np.array(err).sum()/len(ys_test))        
 
    '''stacking'''
    #Define training data
    train_X1 = np.r_[X_scan[:,2],X_scan[:,3]].reshape((2*np.size(x_scan),1))
    train_X2 = np.r_[X_scan[:,3],X_scan[:,1]].reshape((2*np.size(x_scan),1))
    train_X3 = np.r_[X_scan[:,1],X_scan[:,2]].reshape((2*np.size(x_scan),1))   
    train_Y1 = np.r_[Y_scan[:,2],Y_scan[:,3]].reshape((2*np.size(x_scan),1))
    train_Y2 = np.r_[Y_scan[:,3],Y_scan[:,1]].reshape((2*np.size(x_scan),1))
    train_Y3 = np.r_[Y_scan[:,1],Y_scan[:,2]].reshape((2*np.size(x_scan),1))
    
    #Cross-validation(The first layer of the stacking model)
    W1 = []
    Yt_test = [0 for x in range(len(y_test))]
    Yt_scan = [0 for x in range(X_scan.shape[0])]
    for i in range(3):
        model1.fit(locals()['train_X'+str(i+1)],locals()['train_Y'+str(i+1)])
        ys_target1, _ = model1.predict(x_target1, return_std=True)
        ys_test, _ = model1.predict(x_test, return_std=True)
        res_1 = y_target1.reshape(-1,1)-ys_target1                    
        model2.fit(x_target1,res_1)
        res_t, _ = model2.predict(x_test, return_std=True)
        yt_test = ys_test+res_t                        
        yr_scan, _ = model2.predict(X_scan[:,i+1].reshape(-1,1), return_std=True)
        yt_scan = Y_scan[:,i+1].reshape(-1,1)+yr_scan                     
        E = abs(yt-y_test.reshape(-1,1))
        L = np.array(E**2).sum()/len(E)
        W1.append(1/L)
        Yt_test = np.c_[np.array(Yt_test),np.array(yt_test)]       #Record cross-validation results 
        Yt_scan = np.c_[np.array(Yt_scan),np.array(yt_scan)]       #For the second layer of stacking  
        
    #Weighted average cross-validation results
    yt_testA = (W1[0]*Yt_test[:,1]+W1[1]*Yt_test[:,2]+W1[2]*Yt_test[:,3])/(W1[0]+W1[1]+W1[2])
    EA = abs(yt_testA-y_test.reshape(-1,1))
    L_A = np.array(EA**2).sum()
    
    #The second layer of the stacking model
    X_scan = np.hstack((X_scan[:,1],X_scan[:,2],X_scan[:,3])).reshape(-1,1)
    Yt_scan = np.hstack((Yt_scan[:,1],Yt_scan[:,2],Yt_scan[:,3]))
    model1.fit(X_scan,Yt_scan)
    ys_target2,sigma1 = model1.predict(x_target2, return_std=True)
    ys_testB,sigma2 = model1.predict(x_test, return_std=True)
    res_2 = y_target2-ys_target2                       
    model2.fit(x_target2,res_2)
    res_t, sigma = model2.predict(x_test, return_std=True)
    yt_testB = ys_testB+res_t                       
    EB = abs(yt_testB-y_test.reshape(-1,1))
    L_B = np.array(EB**2).sum()
    #Final Results
    Yt_test = (1/L_A*yt_testA+1/L_B*yt_testB)/(1/L_A+1/L_B)
    #Error of stacking
    err2 = abs(Yt_test-y_test)
    MSE_S.append(np.array(err2).sum()/len(err2)) 
    
#Display 
x = np.linspace(0,0.04,100)
fig, ax = plt.subplots(figsize=(6,4), dpi=100)
plt.plot(x,get_d(x, MSE_RA),'g.-',label=u'RA')
plt.plot(x,get_d(x,MSE_S),'r.-',label=u'MSIF-S')
plt.plot(x,get_d(x,MSE_W),'b.-',label=u'MSIF-W')
plt.plot(x,get_d(x,MSE_IF),'k.-',label=u'Reverse')
size = 12
fontfamily = 'NSimSun'
font = {'family':fontfamily,
        'size':12,
        'weight':23}
ax.set_xlabel('RMSE(mm)',fontproperties = fontfamily, size = size)
ax.set_ylabel('Probability Density',fontproperties = fontfamily, size = size)
plt.yticks([]) 
plt.xticks(fontproperties = fontfamily, size = size) 
ax.set_title('${\sigma}_{S} = 0.1 mm$,  ${\sigma}_{T} = 0.01 mm$', fontproperties = fontfamily, size = size)
plt.legend(prop=font)
plt.tight_layout()
plt.legend(prop=font)
plt.show() 
