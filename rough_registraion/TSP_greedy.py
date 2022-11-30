import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


# define a timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('time used: ', end - start)
        return result

    return wrapper


def TSP_greedy(X):
    # generate Hamiltonian path as trianglation
    N = X.shape[0]
    # generate distance matrix
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(X[i, :] - X[j, :])

    total_dist = 0
    path = []
    remaining_path = list(range(N))
    random_start = 11
    path.append(random_start)
    remaining_path.remove(random_start)

    while len(path) < N:
        min_dist = np.inf
        for i in remaining_path:
            if D[path[-1], i] < min_dist:
                min_dist = D[path[-1], i]
                min_index = i
        path.append(min_index)
        total_dist += min_dist
        remaining_path.remove(min_index)
    return path, total_dist


def TSP_greedy_heuristic(X):
    # generate Hamiltonian path as trianglation
    N = X.shape[0]
    # generate distance matrix
    D = np.zeros((N, N))
    edges = np.array([[-1, -1]])
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(X[i, :] - X[j, :])

    for i in range(N):
        for j in range(i, N):
            D[i, j] = np.inf

    while 1:
        x_index, y_index = np.unravel_index(D.argmin(), D.shape)
        D[x_index, y_index] = np.inf

        x_index_in_edges = np.where(edges == x_index)
        if x_index_in_edges[0].size < 2:
            y_index_in_edges = np.where(edges == y_index)
            if y_index_in_edges[0].size < 2:
                if edges.shape[0] == N:
                    edges = np.concatenate((edges, np.array([[x_index, y_index]])), axis=0)
                    break
                else:
                    edges_temp = edges
                    x_index_tmp = x_index
                    while 1:
                        x_index_in_edges_temp = np.where(edges_temp == x_index_tmp)
                        if x_index_in_edges_temp[0].size != 0:
                            x_index_tmp = edges_temp[x_index_in_edges_temp[0], 1 - x_index_in_edges_temp[1]]
                            edges_temp = np.delete(edges_temp, x_index_in_edges_temp[0], axis=0)
                        else:
                            break
                    if x_index_tmp != y_index:
                        edges = np.concatenate((edges, np.array([[x_index, y_index]])), axis=0)
    edges = np.delete(edges, 0, axis=0)
    Z = np.zeros((N, 2))
    point_index = [edges[0,0], edges[0,1]]
    edges = np.delete(edges, 0, axis=0)
    for i in range(1,N):
        index = np.where(edges == point_index[-1])
        if index[0].size == 0:
            print('error')
        point_index.append(edges[index[0],1-index[1]][0])
        edges = np.delete(edges, index[0], axis=0)

    for i in range(N):
        Z[i,:] = X[point_index[i],:]


    return Z




@timer
def creatMSTMap(points):
    # generate Minimum Spanning Tree as trianglation
    # using Prim algorithm
    in_tree_array = [0]
    N = len(points)
    out_tree_array = [i for i in range(1, len(points))]
    map = np.zeros((N, N))
    while len(out_tree_array) > 0:
        min_dist = 100000
        for i in in_tree_array:
            for j in out_tree_array:
                dist = np.linalg.norm(points[i] - points[j])
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j
        map[min_i][min_j] = 1
        map[min_j][min_i] = 1
        in_tree_array.append(min_j)
        out_tree_array.remove(min_j)
    return map


@timer
def creatMSTmap_using_kdtree(points):
    # generate Minimum Spanning Tree as trianglation
    # using kdtree
    in_tree_array = [0]
    N = len(points)
    out_tree_array = [i for i in range(1, len(points))]
    map = np.zeros((N, N))
    points = np.float32(points)
    kdtree = cv2.flann.Index()
    params = dict(algorithm=1, trees=1)
    kdtree.build(points, params)
    indices, dists = kdtree.knnSearch(points, points.shape[0], params=-1)
    iter_times = 0
    while len(out_tree_array) > 0:
        min_distance = 100000000
        min_index1 = -1
        min_index2 = -1
        search_indices = indices[out_tree_array, :]
        search_dists = dists[out_tree_array, :]
        for i in range(len(search_indices)):
            for k in range(1, N):
                iter_times += 1
                if search_indices[i][k] in in_tree_array and search_dists[i][k] < min_distance:
                    min_distance = search_dists[i][k]
                    min_index1 = i
                    min_index2 = search_indices[i][k]
                    continue
        map[min_index2][out_tree_array[min_index1]] = 1
        in_tree_array.append(out_tree_array[min_index1])
        out_tree_array.remove(out_tree_array[min_index1])
    print('iter_times: ', iter_times)
    # print(np.sum(map, axis=0))
    return map


# plot MST according to the map
def plotMST(points, map):
    for i in range(len(map)):
        for j in range(len(map)):
            if map[i][j] == 1:
                plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'r')
    # annotate points
    for i in range(len(points)):
        plt.annotate(i, (points[i][0], points[i][1]))


    plt.show()


def plotHamiltonianPath(points, path):
    for i in range(len(path) - 1):
        plt.plot([points[path[i]][0], points[path[i + 1]][0]], [points[path[i]][1], points[path[i + 1]][1]], 'r')
    # annotate point index
    for i in range(len(path)):
        plt.annotate(str(i), (points[path[i]][0], points[path[i]][1]))
    plt.show()
    plt.cla()
    plt.clf()


if __name__ == '__main__':
    X = np.loadtxt('Xpoint_set.txt')
    MST = creatMSTMap(X)
    plotMST(X, MST)
    edges_list = []
    # get edges list from MST map
    for i in range(MST.shape[0]):
        for j in range(0, i):
            if MST[i][j] == 1:
                edges_list.append([i, j])
    edges_list = np.array(edges_list)



    # Hamiltonian_path, total_dist = TSP_greedy(X)
    # Hamiltonian_path = TSP_greedy_heuristic(X)
    # for i in range(len(Hamiltonian_path)-1):
    #     plt.plot([Hamiltonian_path[i][0],Hamiltonian_path[i+1][0]], [Hamiltonian_path[i][1],Hamiltonian_path[i+1][1]], 'r')
    # plotHamiltonianPath(X, Hamiltonian_path)
