import numpy as np

def Hermitian(M): # returns the hermitian conjugate of a matrix
    return np.transpose(np.conjugate(M))

def frob_norm(M): # returns frobenius norm
    return np.sqrt(np.trace(np.matmul(Hermitian(M), M)))

def normalize(M): # normalizes a transfer matrix to transmission of 1
    N = len(M)
    return np.sqrt(N)*M/frob_norm(M)

def transmission(M): # transmission of a lossy transform
    return frob_norm(M)**2/len(M)

def mixing_entropy(M): # mixing entropy of a transform 
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10432384&tag=1
    N = len(M)
    LogNM2 = np.log(abs(M)**2) / np.log(N)
    return np.sum((abs(M)**2)*LogNM2)*(-1/N)

def fidelity(U1, U2):
    Z = np.matmul(Hermitian(U2), U1)
    N1 = frob_norm(U1)
    N2 = frob_norm(U2)
    return (np.abs(np.trace(Z))/(N1*N2))**2

def DFT(N=4): # generates a discrete fourier transform matrix
    dftmtx = np.zeros((N, N), dtype='complex')
    for i in range(N):
        for j in range(N):
            theta = 2*i*j*np.pi/N
            dftmtx[i][j] = np.exp(-1j*theta)
    return (1/np.sqrt(N))*dftmtx