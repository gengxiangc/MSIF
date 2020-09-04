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
import warnings
warnings.filterwarnings("ignore")

def f_theo(x):           #Theoretical curve
    return (np.sin(x))
def f_scan(x):           #Add system error
    return f_theo(x) + 0.2*np.sin(x/3)

def get_d(x, data):      #Get distribution of probability density 
    mu=np.mean(data)
    std=np.std(data)       
    y=[stats.norm(mu,std).pdf(a) for a in x]
    return y

Noise_s = 0.1
Noise_t = 0.01
N_w = 10
N = 20                        #Times of experiments
N_s = 3                       #Number of scanning data
Ns,Nt,N_test = 100,15,75       #Number of scanning/target points
xmin,xmax = -5,5

#Define test data
x_test = np.linspace(xmin+0.1,xmax-0.1,N_test)
x_test = np.atleast_2d(x_test).T
y_test=f_theo(x_test).ravel()
#data set for weighting
x_w = np.linspace(xmin+0.3,xmax-0.3,N_w)
x_w = np.atleast_2d(x_w).T
y_w = f_theo(x_w).ravel()
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
#New target data
x_target_ = np.vstack((x_target,x_w))
y_target_ = np.vstack((y_target.reshape(-1,1),y_w.reshape(-1,1))).ravel()

MSE_S = []
MSE_W = []
MSE_RA = []
MSE_IF = []
MSE_RA1 = []

for j in range(N):           #N times of experiments
    Yt=[0 for x in range(0, len(x_test))]
    W=[]
    X_scan=[0 for x in range(Ns)]
    Y_scan=[0 for x in range(Ns)]
    Ys_target=[0 for x in range(Nt)]
    Ys_test=[0 for x in range(N_test)]
    
    print('Experiments : ', str(j), 'in ', str(N))
    
    for i in range(3):       #RA separately
        #Define scanning data
        x_scan = [xmin+(xmax-xmin)/Ns*(i+np.random.rand()) for i in range(Ns)]
        x_scan = np.array(x_scan)
        x_scan = np.atleast_2d(x_scan).T
        y_scan = f_scan(x_scan).ravel()
        noise = np.random.normal(0,Noise_s,y_scan.shape)
        y_scan += noise 
        #Define model and kernel function
        kernel1 = C(1.0, (1e-3, 10000)) * RBF(5, (3, 10))
        kernel2 = C(1.0, (1e-3, 10000)) * RBF(3, (0.6, 10))
        kernel3 = C(1.0, (1e-3, 10000)) * RBF(3, (0.01, 10))
        kernel4 = C(1.0, (1e-3, 10000)) * RBF(5, (0.8, 10))
        model1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=10)
        model2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=10)
        model3 = GaussianProcessRegressor(kernel=kernel3, n_restarts_optimizer=10)
        model4 = GaussianProcessRegressor(kernel=kernel4, n_restarts_optimizer=10)
        
        #Fit and prediction using low-precision(scanning) data set 
        model1.fit(x_scan,y_scan)
        ys_target_i, _ = model1.predict(x_target, return_std=True)#Prediction value at x_target
        ys_test_i, _ = model1.predict(x_test, return_std=True)    #Prediction value at x_test
        ys_w, _ = model1.predict(x_w, return_std=True)
        
        '''For reverse'''
        #Record the scanned surface of each prediction
        Ys_target = np.c_[np.array(Ys_target),np.array(ys_target_i)]
        Ys_test = np.c_[np.array(Ys_test),np.array(ys_test_i)]
        
        '''For weighted intergration'''
        res_i = y_target - ys_target_i                          #Residual
        model2.fit(x_target,res_i)
        res_w,_ = model2.predict(x_w, return_std=True) 
        res_t_i, _ = model2.predict(x_test, return_std=True)     
        yt_i = ys_test_i+ res_t_i    
        yw_i = ys_w+ res_w 
        
        #Error and loss 
        
        E1 = abs(yt_i - y_test)
        e = abs(yw_i-y_w)                               
        mes_ra = np.array(E1).sum()/len(yt_i)
        L = np.array(e**2).sum()
        MSE_RA.append(mes_ra)
        W.append(1/L)                                           #Weights
        Yt=np.c_[np.array(Yt),np.array(yt_i)]
        
        '''For stacking'''
        X_scan = np.c_[np.array(X_scan),np.array(x_scan)]
        Y_scan = np.c_[np.array(Y_scan),np.array(y_scan)]
        
        '''A new set of comparative experiments proves that it is better to use some probe data to take weights'''
        model1.fit(x_scan,y_scan)
        ys_target_i, _ = model1.predict(x_target_, return_std=True)#Prediction value at x_target
        ys_test_i, _ = model1.predict(x_test, return_std=True)    #Prediction value at x_test
        res_i = y_target_ - ys_target_i                           #Residual
        model3.fit(x_target_,res_i)
        res_t_i, _ = model3.predict(x_test, return_std=True)     
        yt_i = ys_test_i+ res_t_i 
        E = abs(yt_i - y_test)
        mes_ra = np.array(E).sum()/len(yt_i)
        MSE_RA1.append(mes_ra)
        
    '''Weighted intergration'''
    yt=(W[0]*Yt[:,1]+W[1]*Yt[:,2]+W[2]*Yt[:,3])/(W[0]+W[1]+W[2])
    err1 = abs(yt-y_test)                                  
    MSE_W.append(np.array(err1).sum()/len(err1))           
    
    '''Reverse'''  
    # Integrate low-precision surfaces
    ys_target = (Ys_target[:,1]+Ys_target[:,2]+Ys_target[:,3])/3
    ys_test = (Ys_test[:,1]+Ys_test[:,2]+Ys_test[:,3])/3    
    res = y_target - ys_target                              
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
    Yt_test = [0 for x in range(len(y_test))]
    Yt_scan = [0 for x in range(X_scan.shape[0])]
    Y_w = [0 for x in range(N_w)]
    for i in range(3):
        model1.fit(locals()['train_X'+str(i+1)],locals()['train_Y'+str(i+1)])
        ys_target1, _ = model1.predict(x_target1, return_std=True)
        ys_test, _ = model1.predict(x_test, return_std=True)
        res_1 = y_target1.reshape(-1,1)-ys_target1
        ys_w, _ = model1.predict(x_w, return_std=True)        
        model4.fit(x_target1,res_1)
        res_t, _ = model4.predict(x_test, return_std=True)
        res_w,_ = model4.predict(x_w, return_std=True) 
        yt_test = ys_test+res_t
        yw_i=  ys_w+ res_w               
        yr_scan, _ = model4.predict(X_scan[:,i+1].reshape(-1,1), return_std=True)
        yt_scan = Y_scan[:,i+1].reshape(-1,1)+yr_scan                     
        Yt_test = np.c_[np.array(Yt_test),np.array(yt_test)]       #Record cross-validation results 
        Yt_scan = np.c_[np.array(Yt_scan),np.array(yt_scan)]       #For the second layer of stacking  
        Y_w = np.c_[np.array(Y_w),np.array(yw_i)]
     
    #Weighted average cross-validation results
    yt_testA = (Yt_test[:,1]+Yt_test[:,2]+Yt_test[:,3])/3
    y_wA = (Y_w[:,1]+Y_w[:,2]+Y_w[:,3])/3
    EA = abs(y_wA-y_w)
    L_A = np.array(EA**2).sum()
    
    #The second layer of the stacking model
    X_scan = np.hstack((X_scan[:,1],X_scan[:,2],X_scan[:,3])).reshape(-1,1)
    Yt_scan = np.hstack((Yt_scan[:,1],Yt_scan[:,2],Yt_scan[:,3]))
    model1.fit(X_scan,Yt_scan)
    ys_target2,sigma1 = model1.predict(x_target2, return_std=True)
    ys_testB,sigma2 = model1.predict(x_test, return_std=True)
    res_2 = y_target2-ys_target2
    ys_w, _ = model1.predict(x_w, return_std=True)                   
    model4.fit(x_target2,res_2)
    res_t, _ = model4.predict(x_test, return_std=True)
    res_w,_ = model4.predict(x_w, return_std=True)
    yt_testB = ys_testB+res_t       
    yw_B =  ys_w+ res_w            
    EB = abs(yw_B-y_w)
    L_B = np.array(EB**2).sum()
   
    #Final Results
    Yt_test = (1/L_A*yt_testA+1/L_B*yt_testB)/(1/L_A+1/L_B)
    #Error of stacking
    err2 = abs(Yt_test-y_test)
    MSE_S.append(np.array(err2).sum()/len(err2)) 
    
#Display 
x = np.linspace(0,0.03,100)
fig, ax = plt.subplots(figsize=(6,4), dpi=200)
plt.plot(x,get_d(x, MSE_RA),'g.-',label=u'RA(target)')
plt.plot(x,get_d(x, MSE_RA1),'y.-',label=u'RA(target+w)')
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