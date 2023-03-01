import numpy as np


class FarthestPointSampler:
    def __init__(self, input_points, n_sample):
        self.points = input_points
        self.n_sample = n_sample
        assert self.n_sample <= self.points.shape[0], "Sample number is larger than point number"

    def sample(self):
        # find the farthest point as seed
        farthest_idx = np.argmax(np.sum((self.points - self.points[0, :]) ** 2, axis=1))
        sampled_points = []
        sampled_points.append(self.points[0])
        # init dist with max double value
        dist = np.ones((self.points.shape[0], 1)) * np.iinfo(np.int32).max
        dist = dist[0]
        while len(sampled_points) < self.n_sample:
            new_dist = np.linalg.norm(self.points - sampled_points[-1], axis=1)
            # update dist
            dist = np.minimum(dist, new_dist)
            # find the farthest point from sampled points
            print(farthest_idx)
            farthest_idx = np.argmax(dist)
            sampled_points.append(self.points[farthest_idx])
        return np.array(sampled_points)


if __name__ == '__main__':
    # some test
    points = np.loadtxt('/data/FPsource.txt')
    sampler = FarthestPointSampler(points, 108)
    sampled_points = sampler.sample()
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 2)
    # set figure size
    fig.set_size_inches(10, 5)
    fig.dpi = 300
    fig.tight_layout()
    ax[0].scatter(points[:, 0], points[:, 1], s=0.5)
    ax[1].scatter(sampled_points[:, 0], sampled_points[:, 1],s=1)
    ax[0].set_xlim(0,1920)
    ax[0].set_ylim(0,1080)
    ax[1].set_xlim(0,1920)
    ax[1].set_ylim(0,1080)
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    plt.show()
