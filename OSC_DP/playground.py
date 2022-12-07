import numpy
import numpy as np
from matplotlib import pyplot as plt
import os
from ShapeContextMatching import *
from MatchTestForPnP import plot_points_in_log_polar
from TSP_greedy import TSP_greedy

if __name__ == '__main__':
    # read the TXT
    source_points = np.loadtxt('fish_model_set.txt')
    target_points = np.loadtxt('fish_target_set.txt')
    # plot the X_points and Y_points
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(source_points[:, 0], source_points[:, 1], 'b+', label='source_points')
    ax[1].plot(target_points[:, 0], target_points[:, 1], 'r+', label='target_points')
    ax[0].legend()
    ax[1].legend()
    plt.show()

    # greedy_result = TSP_greedy(source_points)
    # #plot the greedy result
    # plt.plot(greedy_result[:, 0], greedy_result[:, 1], 'g-', label='greedy_result')
    # plt.legend()
    # plt.show()

    shape_source = Shape(shape=source_points.tolist())
    shape_target = Shape(shape=target_points.tolist())

    matching_index = shape_source.matching(shape_target)
    matching_index = np.array(matching_index)

    plt.scatter(source_points[:, 0], source_points[:, 1], c='b', label='source_points')
    plt.scatter(target_points[:, 0], target_points[:, 1], c='r', label='target_points')
    for i in range(matching_index.shape[1]):
        plt.plot([matching_index[0,i,0],matching_index[1,i,0]],[matching_index[0,i,1],matching_index[1,i,1]],'g-')
    plt.axis('equal')
    plt.legend()
    plt.show()


    plot_points_in_log_polar(shape_source,shape_target,result=matching_index,use_shape=1)






