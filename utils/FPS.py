import numpy as np

class FarthestPointSampler:
    def __init__(self,input_points,n_sample):
        self.points = input_points
        self.n_sample = n_sample
        assert self.n_sample <= self.points.shape[0], "Sample number is larger than point number"

    def sample(self):
        # find the farthest point as seed
        farthest_idx = np.argmax(np.sum((self.points - self.points[0,:])**2,axis=1))
        sampled_points = []
        sampled_points.append(self.points[farthest_idx])
        # init dist with max int value
        dist = np.ones((self.points.shape[0],1)) * np.iinfo(np.int32).max
        while len(sampled_points) < self.n_sample:
            new_dist = np.linalg.norm(self.points - sampled_points[-1],axis=1)
            # update dist
            dist = np.minimum(dist,new_dist)
            # find the farthest point from sampled points
            farthest_idx = np.argmax(dist)
            sampled_points.append(self.points[farthest_idx])
        return np.array(sampled_points)

