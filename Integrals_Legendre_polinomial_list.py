# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:56:23 2021

@author: peter
"""

from sympy import integrate, Symbol, legendre_poly, diff
import numpy as np
import scipy.io as spio

NumberOfModes = 9

def IntIa (n,i,j):
    x = Symbol("x")
    nshift = n + 2
    ishift = i + 2
    jshift = j + 2
    pnshift = legendre_poly(nshift,x)
    pishift = legendre_poly(ishift,x)
    pjshift = legendre_poly(jshift,x)
    p = pnshift * pishift * pjshift
    
    Ia = integrate(p,(x,-1,1))
    return Ia

 
def IntIc (n,i,j):
    x = Symbol("x")
    nshift = n + 2
    ishift = i + 2
    jshift = j + 2
    pnshift = legendre_poly(nshift,x)
    pishift = legendre_poly(ishift,x)
    pjshift = legendre_poly(jshift,x)
    p = -(1 - x ** 2) * pnshift * diff(pishift,x) * diff(pjshift,x)
    
    Ic = integrate(p,(x,-1,1))
    return Ic

def IntIg (n,i,j):
    x = Symbol("x")
    nshift = n + 2
    ishift = i + 2
    jshift = j + 2
    pnshift = legendre_poly(nshift,x)
    pishift = legendre_poly(ishift,x)
    pjshift = legendre_poly(jshift,x)
    p = (1 - x ** 2) * ((nshift + 1) * (ishift + 1) * diff(pnshift * pishift, x) + diff((1 - x ** 2) * diff(pnshift,x) * diff(pishift,x), x)) * diff(pjshift,x)
    
    Ig = integrate(p,(x,-1,1))
    return Ig

def LambdaA(n,i,j):
    nsft = n + 2.0
    isft = i + 2.0
    lmdA = (nsft + 1.0) * (nsft + isft + 1.0) * Ia[n,i,j] + Ic[n,i,j]
    return lmdA

def LambdaB(n,i,j):
    nsft = n + 2.0
    lmdB = 0.5 * (nsft + 1.0) * (nsft**2.0 + 4.0 * nsft + 4.0) * Ia[n,i,j] + (nsft + 2.0) * Ic[i,n,j]
    return lmdB

def LambdaC(n,i,j):
    nsft = n + 2.0
    isft = i + 2.0
    lmbC = (nsft + isft + 1.0) * (nsft + isft + 2.0) * Ia[n,i,j] + Ic[n,i,j] + Ic[i,n,j]
    return lmbC

def LambdaD(n,i,j):
    nsft = n + 2.0
    isft = i + 2.0
    lmbd = (nsft + isft + 3.0) * (nsft + isft + 4.0) * ((nsft + 1.0) * (isft + 1.0) * Ia[n,i,j] - Ic[j,n,i])
    return lmbd

##Gs

def Ga(n,i,j):
    nsft = n + 2.0
    isft = i + 2.0
    ga = LambdaB(n,i,j) - nsft * (nsft - 1.0) * Ia[n,i,j] - 2.0 * LambdaC(n,i,j) / (isft + 1.0)
    return ga

def Gb(n,i,j):
    nsft = n + 2.0
    jsft = j + 2.0
    gb = -((nsft**2.0 + jsft**2.0 + nsft + 5.0 * jsft + 4.0) * Ia[n,i,j] + Ic[n,j,i] + Ic[j,n,i]) / (jsft + 1.0)
    return gb

def Gc(n,i,j):
    nsft = n + 2.0
    isft = i + 2.0
    gc = ((2.0 * nsft**2.0 - nsft + 1.0) / (nsft + 1.0)) * Ia[n,i,j] - 2.0 / (nsft + 1.0) * (LambdaB(n,i,j) - 2.0 * LambdaA(n,i,j) / (isft + 1.0))
    return gc

def Gd(n,i,j):
    nsft = n + 2.0
    jsft = j + 2.0
    gd = 2.0 * LambdaC(n,j,i) / (nsft + 1.0) / (jsft + 1.0) - (LambdaB(j, i, n) - jsft * (jsft - 1.0) * Ia[n,i,j]) / (jsft + 1.0) - 4.0 * (nsft - 1.0) / (nsft + 1.0) * Ia[n,i,j]
    return gd

def Ge(n,i,j):
    nsft = n + 2.0
    jsft = j + 2.0
    ge = ((nsft * nsft - jsft * nsft + 3.0 * jsft + 3.0) * Ia[n,i,j] + Ic[n,j,i]) / (nsft + 1.0) / (jsft + 1.0)
    return ge

##Ms

def Ma(n,i,j):
    ma = Gd(i,j,n) - Gc(n,i,j) - Gc(i,n,j) - Gc(i,j,n)
    return ma

def Mb(n,i,j):
    mb = Gd(j,i,n) + Gd(i,j,n) + 2.0 * (Ge(n,i,j) + Ge(j,i,n)) - Gd(n,i,j) - Gd(i,n,j)
    return mb

def Mc(n,i,j):
    mc = Ge(n,i,j) + Ge(j,i,n)
    return mc

def Md(n,i,j):
    md = Ge(n,i,j) + Ge(j,i,n) - Ge(i,n,j)
    return md

##Qs

def Qb(n,i,j):
    nsft = n + 2.0
    jsft = j + 2.0
    qb = 2.0 * (nsft + 2.0) * (2.0 * nsft + 1.0) / (nsft + 1.0) * Gb(n,i,j) + (LambdaD(n,j,i) - Ig[n,j,i]) / (nsft + 1.0) / (jsft + 1.0)
    return qb

def Qc(n,i,j):
    nsft = n + 2.0
    isft = i + 2.0
    jsft = j + 2.0
    qc = (8.0 * (nsft - 1.0) - (jsft + 3.0) * (jsft + 4.0) * (jsft + 4.0)) * Ia[n,i,j] - 2.0 * (jsft + 4.0) * Ic[i,j,n] + 2.0 * (nsft + 2.0) * (nsft - 1.0) / (nsft + 1.0) * Gb(n,i,j) \
          + 4.0 / (jsft + 1.0) * ((jsft + 2.0) * (jsft + 0.5) * Ga(j,i,n) - LambdaC(i,j,n) + LambdaB(j,i,n) + LambdaD(i,j,n) / (isft + 1.0) - Ig[n,j,i] / (nsft + 1.0))
    return qc

Ia = np.zeros((NumberOfModes,NumberOfModes,NumberOfModes))

for n in range(NumberOfModes):
    for i in range(NumberOfModes):
        for j in range(NumberOfModes):
            Ia[n,i,j] = IntIa(n,i,j)

Ic = np.zeros((NumberOfModes,NumberOfModes,NumberOfModes))

for n in range(NumberOfModes):
    for i in range(NumberOfModes):
        for j in range(NumberOfModes):
            Ic[n,i,j] = IntIc(n,i,j)

Ig = np.zeros((NumberOfModes,NumberOfModes,NumberOfModes))

for n in range(NumberOfModes):
    for i in range(NumberOfModes):
        for j in range(NumberOfModes):
            Ig[n,i,j] = IntIg(n,i,j)

ia = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)            
gd = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
ma = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
mb = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
mc = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
md = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
ga = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)

gb = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
gc = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
gd = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
ge = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
qb = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
qc = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
lmba = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
lmbb = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
lmbc = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)
lmbd = np.ones(NumberOfModes * NumberOfModes * NumberOfModes)

for n in range(NumberOfModes):
    for i in range(NumberOfModes):
        for j in range(NumberOfModes):
            
            ga[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Ga(n,i,j)
            gb[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Gb(n,i,j)
            gc[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Gc(n,i,j)
            gd[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Gd(n,i,j)
            ge[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Ge(n,i,j)
            
            lmba[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = LambdaA(n,i,j)
            lmbb[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = LambdaB(n,i,j)
            lmbc[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = LambdaC(n,i,j)
            lmbd[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = LambdaD(n,i,j)
            
            ma[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Ma(n,i,j)
            mb[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Mb(n,i,j)
            mc[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Mc(n,i,j)
            md[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Md(n,i,j)
            
            qb[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Qb(n,i,j)
            qc[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = Qc(n,i,j)
            
            ia[j + NumberOfModes * i + (NumberOfModes * NumberOfModes) * n] = IntIa(n,i,j)


np.savetxt("ia.txt", ia)
np.savetxt("gd.txt", gd)
np.savetxt("ma.txt", ma)
np.savetxt("mb.txt", mb)
np.savetxt("mc.txt", mc)
np.savetxt("md.txt", md)
np.savetxt("qb.txt", qb)
np.savetxt("qc.txt", qc)

'''            
np.save('Ia.npy',Ia)
np.save('gd.npy',gd)
np.save('ma.npy',ma)
np.save('mb.npy',mb)
np.save('mc.npy',mc)
np.save('md.npy',md)
np.save('qb.npy',qb)
np.save('qc.npy',qc)

matIa = spio.loadmat('Ia.mat')
matIa = matIa['Ia']
matIb = spio.loadmat('Ib.mat')
matIb = matIb['Ib']
matIc = spio.loadmat('Ic.mat')
matIc = matIc['Ic']
matIg = spio.loadmat('Ig.mat')
matIg = matIg['Ig']

matAa = spio.loadmat('Aa.mat')
matAa = matAa['Aa']
matAb = spio.loadmat('Ab.mat')
matAb = matAb['Ab']
matAc = spio.loadmat('Ac.mat')
matAc = matAc['Ac']
matAd = spio.loadmat('Ad.mat')
matAd = matAd['Ad']

matGa = spio.loadmat('Ga.mat')
matGa = matGa['Ga']
matGb = spio.loadmat('Gb.mat')
matGb = matGb['Gb']
matGc = spio.loadmat('Gc.mat')
matGc = matGc['Gc']
matGd = spio.loadmat('Gd.mat')
matGd = matGd['Gd']
matGe = spio.loadmat('Ge.mat')
matGe = matGe['Ge']

matMa = spio.loadmat('Ma.mat')
matMa = matMa['Ma']
matMb = spio.loadmat('Mb.mat')
matMb = matMb['Mb']
matMc = spio.loadmat('Mc.mat')
matMc = matMc['Mc']
matMd = spio.loadmat('Md.mat')
matMd = matMd['Md']

matQa = spio.loadmat('Qa.mat')
matQa = matQa['Qa']
matQb = spio.loadmat('Qb.mat')
matQb = matQb['Qb']
matQc = spio.loadmat('Qc.mat')
matQc = matQc['Qc']
'''