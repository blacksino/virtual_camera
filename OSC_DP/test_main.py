import numpy as np

from TSP_greedy import *
from choose_refer_and_recirc import *
# from SC_cost_fast_fun import SC_cost_fast_fun
from dist2 import dist2

if __name__ == '__main__':
    X = np.loadtxt('Xpoint_set.txt')
    Y = np.loadtxt('Ypoint_set.txt')
    # X = np.round(X,5)
    # Y = np.round(Y,5)
    # MST = creatMSTMap(X)
    # plotMST(X, MST)
    # Hamiltonian_path, total_dist = TSP_greedy(X)
    # plotHamiltonianPath(X, Hamiltonian_path)
    # pathed_X = X[Hamiltonian_path,:]
    # pathed_X = choose_refer_and_recirc(pathed_X, 'close')
    Hamiltonian_path = TSP_greedy_heuristic(X)
    X = choose_refer_and_recirc(Hamiltonian_path, 'close')
    Nx = X.shape[0]
    edges = np.array([(np.arange(Nx)),np.zeros(Nx)]).T
    edges = edges.astype(int)
    nbins_theta = 12
    nbins_r = 5
    r_inner = 1 / 8
    r_outer = 2
    #################
    Ny = Y.shape[0]
    Ne = edges.shape[0]

    diff_Y = np.zeros((Ny, Ny,2))
    for i in range(0, Ny):
        for j in range(0, Ny):
            diff_Y[i, j] = Y[i, :] - Y[j, :]

    angle_mat_Y = np.zeros((Ny, Ny))
    for i in np.arange(0, Ny).reshape(-1):
        for j in np.arange(1, Ny).reshape(-1):
            angle_mat_Y[i, j] = np.arctan2(diff_Y[i, j, 1], diff_Y[i, j, 0])

    frame_edges = X[edges[:, 0]] - X[edges[:, 1]]
    frame_edges_angle = np.arctan2(frame_edges[:, 1], frame_edges[:, 0])

    r_array = np.real(np.sqrt(dist2(X.T, X.T)))
