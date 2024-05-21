import metrics
import transfer_matrix
import numpy as np
import pickle
import math

if __name__ == '__main__':
    data_folder = "SSC_Data_r35_n8" #CHANGE THIS TO WHATEVER YOU NEED
    t_matrix = transfer_matrix.create_tranfer_matrix(data_folder)

    print()
    print()
    print("Transfer Matrix: ")
    print(t_matrix)
    print()
    print("DATA ANALYSIS RESULTS")
    print()
    print("Hermitian:")
    print(metrics.Hermitian(t_matrix))
    print()
    print("Frobenius Norm:")
    print(metrics.frob_norm(t_matrix))
    print()
    print("Normaized:")
    print(metrics.normalize(t_matrix))
    print()
    print("Transmission:")
    print(metrics.transmission(t_matrix))
    print()
    print("Mixing Entropy:")
    print(metrics.mixing_entropy(t_matrix))
    print()
    print("DFT:")
    print(metrics.DFT(int(math.sqrt(t_matrix.size))))