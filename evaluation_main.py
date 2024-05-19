import metrics
import transfer_matrix
import numpy as np
import pickle
import math

if __name__ == '__main__':
    t_matrix = transfer_matrix.create_tranfer_matrix("sample_data")

    print()
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