import numpy as np

def choose_refer_and_recirc(X = None,openness = None):
    Nx = X.shape[0]
    # X is a triangulated point set from TSP_greedy
    if openness == 'open':
        dif = X - X[np.hstack((np.arange(1,Nx),np.array(0))),:]
        edgelen = np.linalg.norm(dif,axis=1,keepdims=True)
        max_index = np.argmax(edgelen)
        X = X[np.array([np.arange(max_index,Nx),np.arange(0,max_index)]),:]
        D = np.zeros((Nx, Nx))
        for ii in np.arange(1, Nx + 1).reshape(-1):
            for jj in np.arange(1, Nx + 1).reshape(-1):
                D[ii, jj] = np.linalg.norm(X[ii, :] - X[jj, :])

        d_max = np.max(D.sum(axis=1))
        farthest_idx = np.argmax(D.sum(axis=1))[-1]
        farthest = X[farthest_idx, :]
        if farthest[0] == 2:
            X = np.flipud(X)
    elif openness == 'close':
        # Generate distance matrix
        D = np.zeros((Nx,Nx))
        for ii in np.arange(0,Nx).reshape(-1):
            for jj in np.arange(0,Nx).reshape(-1):
                D[ii,jj] = np.linalg.norm(X[ii,:] - X[jj,:])
        farthest_idx = np.argmax(D.sum(axis=1))
        farthest = X[farthest_idx,:]
        X = X[np.hstack([np.arange(farthest_idx,Nx),np.arange(0,farthest_idx)]),:]
    return farthest_idx,X