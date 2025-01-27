import torch
from torch.utils.data import Dataset
import inspect 
try: 
    from deepreach.utils.comparisons import GroundTruthHJSolution
    from deepreach.dynamics import dynamics_hjr
except: 
    from utils.comparisons import GroundTruthHJSolution
    from dynamics import dynamics_hjr
import numpy as np 

"""
1. Create the ground truth solution
2. Just have all the points in the ground truth solution as datapoints that you pull from with __getitem__
"""
# uses model input and real boundary fn
# Dataset for ground truth reachability dataset 
class GTReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples):
        self.dynamics = dynamics
        # self.numpoints = numpoints
        # self.pretrain = pretrain
        # self.pretrain_counter = 0
        # self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        # self.num_src_samples = num_src_samples
        # self.num_target_samples = num_target_samples

        dynamics_class_name = self.dynamics.__class__.__name__
        dynamics_class = getattr(dynamics_hjr, dynamics_class_name)
        dynamics_params = inspect.signature(dynamics_class).parameters
        dynamics_args = {argname: getattr(dynamics, argname) for argname in dynamics_params 
                        if hasattr(dynamics, argname)}
        dynamics_args['tMin'] = self.tMin
        dynamics_args['tMax'] = self.tMax

        hj_dyn = dynamics_class(dynamics, **dynamics_args)
        print("Creating GT loss ground truth solution")
        self.gt_hj_solution = GroundTruthHJSolution(hj_dynamics=hj_dyn)
        print("Finished GT loss ground truth solution")

        self.grid = self.gt_hj_solution.grid
        self.times = torch.tensor(np.array(self.gt_hj_solution.times), dtype=torch.float32) 
        self.all_states = torch.tensor(np.array(self.grid.states), dtype=torch.float32)
        self.values = torch.tensor(np.array(self.gt_hj_solution.value_functions), dtype=torch.float32)

        self.total_shape = [len(self.times)]
        self.total_shape.extend(list(self.grid.states.shape[:-1]))
        self.length = np.prod(self.total_shape)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        idx = idx % self.length 

        grid_idx = np.unravel_index(idx, self.total_shape)
        
        time = self.times[grid_idx[0]]
        state = self.all_states[grid_idx[1], grid_idx[2], grid_idx[3], grid_idx[4]]
        try: 
            model_coords = torch.cat((torch.tensor([time]), state), dim=0)
            value = self.values[grid_idx]
        except: 
            print("In get item of dataloader")
            import pdb; pdb.set_trace()

        return model_coords, value

