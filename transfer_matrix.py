import glob
from os import walk
import numpy as np
import pickle

def create_tranfer_matrix(data_folder):

# THE PLAN

#walk through files in data folder
    #glob to get itr files in each folder
    #extract itr and input num
    #store vars
    #perform devision to add to matrix

    # this uses walk to get a list of folders and subdirectories
    f = []
    layer = 1
    w = walk(data_folder)
    for (dirpath, dirnames, filenames) in w:
        if layer == 1:
            f.extend(dirnames)
        layer += 1


    print(f)
    print("hi")

    #generate blank transfer matrix
    num_wg = len(f)
    transfer_matrix = np.zeros((num_wg,num_wg))
    print(transfer_matrix)

    # empty arrays to store flux values
    flux_in = np.zeros(num_wg)
    flux_out = np.zeros(num_wg)

    # go folder by folder (each folder is a a different waveguide being tested)
    for folder in f:
        print("x")
        files = glob.glob(data_folder+"/"+folder+"/itr*")


        iteration = int(folder[folder.rfind("_")+1:])
        print(iteration)

        # iteration = folder

        #go through each mointor file in the folder and load the data
        for file in files:
            if file.find("input") != -1:
                monitor = open(file, 'rb')
                data = pickle.load(monitor)
                monitor.close()
                print(data)

                print("flux: ")
                print(abs(data.alpha[0,0,0])**2)

                # this gets the flux we want from the pickle
                wg_num = int(file[file.rfind("_")+1:])
                flux_in[wg_num] = abs(data.alpha[0,0,0])**2


            elif file.find("output") != -1:
                monitor = open(file, 'rb')
                data = pickle.load(monitor)
                monitor.close()
                print(data)

                print("flux: ")
                print(abs(data.alpha[0,0,0])**2)

                # this gets the flux we want from the pickle
                wg_num = int(file[file.rfind("_")+1:])
                flux_out[wg_num] = abs(data.alpha[0,0,0])**2
            else:
                print("File error: " + files)
                exit()
                

            print(file)
            print(flux_in)
            print(flux_out)

            # add fluc data to transfer matric
            for wg in range(len(flux_out)):
                transfer_matrix[iteration,wg] = flux_out[wg]/flux_in[iteration]


        # print(files)
    print("Transfer Matrix: ")
    print(transfer_matrix)
    return transfer_matrix
