import random as rd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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
tau = 0.0
K = 1.0
epsilon = 1.0
t = 0

for j in range(21):
    K = j * 0.2
    for i in range(40, 101):
        tau = i * 0.1
        
        Omega = np.zeros(NN) 
        for i in range(NN):
            Omega[i] = rd.uniform(-0.01, 0.01)
         
        Phi0 = np.zeros(NN) 
        for i in range(NN):
            Phi0[i] = rd.uniform(-0.1, 0.1)

        Psi0 = np.zeros(NN)

        Sol = solve_ivp(rhs,(0, 2000), np.append(Phi0, Psi0), method = 'RK45', max_step = 0.05)

        NameDir1 = 'K='+str(K)

        if not os.path.isdir(NameDir1):
            os.mkdir(NameDir1) 

        NameDir2 = NameDir1+'/NN='+str(NN)+'_tau='+str(tau)+'_K='+str(K)+'_epsilon='+str(epsilon) 

        if not os.path.isdir(NameDir2):
            os.mkdir(NameDir2)        

        f1 = open(NameDir2+"/Phi_dPhi.txt", "w")
        np.savetxt(NameDir2+"/Phi_dPhi.txt", Sol.y)
        np.save(NameDir2+"/Phi_dPhi", Sol.y)
        f1.close()

        f2 = open(NameDir2+"/t.txt", "w")
        np.savetxt(NameDir2+"/t.txt", Sol.y)
        np.save(NameDir2+"/t", Sol.y)
        f2.close()

        ZVec = np.zeros(len(Sol.t))
        for i in range(len(Sol.t)):
            ZVec[i] = np.abs(GlobalOrdPar(np.transpose(Sol.y)[i][:NN]))
        
        f3 = open(NameDir2+"/ZVec.txt", "w")
        np.savetxt(NameDir2+"/ZVec.txt", ZVec)
        np.save(NameDir2+"/ZVec", ZVec)
        f3.close()


        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot()
        x = np.arange(-np.pi, np.pi, 0.1)
        ax.plot(Sol.t, ZVec)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.ylim([0, 1.1])
        plt.xlabel('$t$', fontsize=14)
        plt.ylabel('$|Z|$', fontsize=14)
        plt.grid(True)
        plt.savefig(NameDir2+'/ZVec.png', dpi = 150)

        n = 1

        plt.plot(Sol.t, Sol.y[NN + n - 1])
        plt.xlabel('$t$', fontsize=14)
        plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$', fontsize=14)
        plt.grid(True)
        plt.ylim([-np.pi, np.pi])
        plt.savefig(NameDir2+'/Psi['+str(n)+'].png', dpi = 150)

        n = 8

        plt.plot(Sol.t, Sol.y[NN + n - 1])
        plt.xlabel('$t$', fontsize=14)
        plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$', fontsize=14)
        plt.ylim([-np.pi, np.pi])
        plt.grid(True)
        plt.savefig(NameDir2+'/Psi['+str(n)+'].png', dpi = 150)
        plt.show()


        n = 32

        plt.plot(Sol.t, Sol.y[NN + n - 1])
        plt.xlabel('$t$', fontsize=14)
        plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$', fontsize=14)
        plt.ylim([-np.pi, np.pi])
        plt.grid(True)
        plt.savefig(NameDir2+'/Psi['+str(n)+'].png', dpi = 150)
        plt.show()

        n = 64

        plt.plot(Sol.t, Sol.y[NN + n - 1])
        plt.xlabel('$t$', fontsize=14)
        plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$', fontsize=14)
        plt.ylim([-np.pi, np.pi])
        plt.grid(True)
        plt.savefig(NameDir2+'/Psi['+str(n)+'].png', dpi = 150)

        n = 128

        plt.plot(Sol.t, Sol.y[NN + n - 1])
        plt.xlabel('$t$', fontsize=14)
        plt.ylabel('$\dot{\\varphi}_{'+str(n)+'}$', fontsize=14)
        plt.ylim([-np.pi, np.pi])
        plt.grid(True)
        plt.savefig(NameDir2+'/Psi['+str(n)+'].png', dpi = 150)

        PhiEnd = np.mod(np.transpose(Sol.y)[-1][:NN], 2*np.pi) - np.pi
        plt.ylim([-np.pi, np.pi])
        plt.plot(PhiEnd, 'b.')
        plt.xlabel('n', fontsize=14)
        plt.ylabel('${\\varphi}_{end}$', fontsize=14)
        plt.grid(True)
        plt.yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ['$-{\\pi}$', '$-{\\pi} / 2$','0', '${\\pi} / 2$', ' ${\\pi}$'])
        plt.savefig(NameDir2+'/PhiEnd.png', dpi = 150)

