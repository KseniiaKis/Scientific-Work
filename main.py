import random as rd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

def GlobalOrdPar(Phi):
    return ( (np.sum(np.exp(1j * Phi))) / len(Phi) )

def rhs(t, X):
    Phi = np.array(X[ : (len(X) // 2) : ])
    Psi = np.array(X[ (len(X) // 2) : len(X) :])
    
    Z = GlobalOrdPar(Phi)
    
    dPhi = Psi
    dPsi = (1/tau)*(Psi * epsilon + K * ((Z * np.exp(-1j*(Phi + Omega * tau))).imag) - Psi  -  tau * K * ((Z * np.exp(-1j * (Phi + Omega * tau))).real) * Psi)
    
    return(np.append(dPhi, dPsi))

NN = 128
tau = 1.1
K = 1.0
epsilon = 1.0
t = 0

Omega = np.zeros(NN) 
for i in range(NN):
    Omega[i] = rd.uniform(-0.01, 0.01)
 
Phi0 = np.zeros(NN) 
for i in range(NN):
    Phi0[i] = rd.uniform(-0.1, 0.1)

Psi0 = np.zeros(NN)

Sol = solve_ivp(rhs,(0, 2000), np.append(Phi0, Psi0), method = 'RK45', max_step = 0.05)

print(Sol)

NameDir = 'NN='+str(NN)+'_tau='+str(tau)+'_K='+str(K)+'_epsilon='+str(epsilon)

if not os.path.isdir(NameDir):
    os.mkdir(NameDir)        

f1 = open(NameDir+"/Phi_dPhi.txt", "w")
np.savetxt(NameDir+"/Phi_dPhi.txt", Sol.y)
np.save(NameDir+"/Phi_dPhi", Sol.y)
f1.close()

f2 = open(NameDir+"/t.txt", "w")
np.savetxt(NameDir+"/t.txt", Sol.y)
np.save(NameDir+"/t", Sol.y)
f2.close()

ZVec = np.zeros(len(Sol.t))
for i in range(len(Sol.t)):
    ZVec[i] = np.abs(GlobalOrdPar(np.transpose(Sol.y)[i][:NN]))

plt.plot(Sol.t, ZVec)
plt.xlabel('$t$')
plt.ylabel('$|Z|$')
plt.savefig(NameDir+'/ZVec.png')
plt.show()

n = 1

plt.plot(Sol.t, Sol.y[NN + n - 1])
plt.xlabel('$t$')
plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$')
plt.savefig(NameDir+'/Psi['+str(n)+'].png')
plt.show()

n = 8

plt.plot(Sol.t, Sol.y[NN + n - 1])
plt.xlabel('$t$')
plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$')
plt.savefig(NameDir+'/Psi['+str(n)+'].png')
plt.show()


n = 32

plt.plot(Sol.t, Sol.y[NN + n - 1])
plt.xlabel('$t$')
plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$')
plt.savefig(NameDir+'/Psi['+str(n)+'].png')
plt.show()

n = 64

plt.plot(Sol.t, Sol.y[NN + n - 1])
plt.xlabel('$t$')
plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$')
plt.savefig(NameDir+'/Psi['+str(n)+'].png')
plt.show()

n = 128

plt.plot(Sol.t, Sol.y[NN + n - 1])
plt.xlabel('$t$')
plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$')
plt.savefig(NameDir+'/Psi['+str(n)+'].png')
plt.show()


PhiEnd = np.mod(np.transpose(Sol.y)[-1][:NN], 2*np.pi) - np.pi
plt.ylim([-np.pi, np.pi])
plt.plot(PhiEnd, 'b.')
plt.xlabel('n')
plt.ylabel('${\\varphi}_{end}$')
plt.savefig(NameDir+'/PhiEnd.png')
plt.show()
