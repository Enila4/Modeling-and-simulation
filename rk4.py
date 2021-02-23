#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
def rk4 (fct,y0,t):
    n = len(t)
    h = t[1]-t[0]
    y = np.zeros(n)
    y[0] = y0 
    for i in range (n -1):
        k1 = h*fct(t[i])
        k2 = h*fct(t[i]+h/2)
        k3 = h*fct(t[i]+h/2)
        k4 = h*fct(t[i]+h)
        y[i+1]= y[i]+ (1/6)*(k1+2*k2+2*k3+k4)
    return y

def EDO(r):
    return C[p]/((1+1E-6*N0*np.exp(-(r-r0)/Hn))*r**2*np.cos(np.arcsin(C[p]/((1+1E-6*N0*np.exp(-(r-r0)/Hn))*r))))

T0=0
N0=315
Hn=7
phi=np.array([0,30,60,89.9])*np.pi/180
r0=6371
rmax=r0+650
h=0.1
r=np.arange(r0,rmax,h)
x0 = 0
C=(1+1E-6*N0*np.exp(-(r0-r0)/Hn))*r0*np.sin(phi)

labels=['0°','30°','60°','89,9°']
ax = plt.subplot(projection='polar')
for p in range(4):
    y = rk4(EDO,x0,r)
    d=y[-1]*r0
    print(d)
    ax.plot(y,r,label=labels[p])
    ax.set_rmin(r0)
    ax.set_rmax(rmax)
    ax.set_thetamin(-5)
    ax.set_thetamax(45)
    ax.set_xlabel(r' $\Theta~~[°]$',fontsize=12)
    ax.set_ylabel(r'r [km]',fontsize=12)
    ax.set_theta_direction(-1)

plt.title('rk4, h=1km')
plt.legend(title=r'$\varphi~~[°]$',loc='lower left')    
#plt.savefig('rk4_100m',transparent=False)


# In[30]:


phi=np.array([0,30,60,89.99])*np.pi/180
theta=phi[3]-np.arcsin((r0*np.sin(np.pi-phi[3]))/(r0+650))
theta=theta*180/np.pi


# In[31]:


d=(theta*2*np.pi*r0)/360
d


# In[ ]:





# In[ ]:




