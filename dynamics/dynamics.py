from abc import ABC, abstractmethod
try:
    from utils import diff_operators
except: 
    from deepreach.utils import diff_operators

import math
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
from PIL import Image

# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parameterized models
class Dynamics(ABC):
    def __init__(self, 
    loss_type:str, set_mode:str, 
    state_dim:int, input_dim:int, 
    control_dim:int, disturbance_dim:int, 
    periodic_dims:list,
    state_mean:list, state_var:list, 
    value_mean:float, value_var:float, value_normto:float, 
    deepreach_model:str):
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim 
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.periodic_dims = periodic_dims
        if self.periodic_dims is None:
            self.periodic_dims = []
        elif isinstance(self.periodic_dims, int):
            self.periodic_dims = [self.periodic_dims]
        assert isinstance(self.periodic_dims, list), 'periodic_dims must be a list'

        self.state_mean = torch.tensor(state_mean) 
        self.state_var = torch.tensor(state_var)
        self.state_bounds = torch.stack([self.state_mean - self.state_var, self.state_mean + self.state_var], dim=-1)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepreach_model = deepreach_model
        assert self.loss_type in ['brt_hjivi', 'brat_ci_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        elif self.loss_type == 'brat_ci_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in ['reach', 'avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + str(len(state_descriptor)) + ' != ' + str(self.state_dim)
    
    # ALL METHODS ARE BATCH COMPATIBLE

    # MODEL-UNIT CONVERSIONS (TODO: refactor into separate model-unit conversion class?)

    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepreach_model=="diff":
            # V(s,t) = boundary_fn(s) + NN(s,t) -> NN(s,t) = NN(s,t) * value_var / value_normto
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepreach_model=="exact":
            # V(s,t) = boundary_fn(s) + t * NN(s,t) -> NN(s,t) = NN(s,t) * value_var / value_normto
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            # V(s,t) = NN(s,t) -> NN(s,t) = NN(s,t) * value_var / value_normto
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepreach_model=="diff":
            # \partial V/\partial t = autograd(NN(s,t)) for t
            # \partial V/\partial s = autograd(NN(s,t)) for s + \partial boundary_fn(s) / \partial s
            # (Include /self.state_var division for normalization for dvdx_term1)
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepreach_model=="exact":
            # \partial V/\partial t = NN(s,t) + t * autograd(NN(s,t)) for t (output = NN(s,t))
            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            # \partial V/\partial s = t * autograd(NN(s,t)) for s + \partial boundary_fn(s) / \partial s
            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            # \partial V/\partial t = NN(s,t) for t
            # \partial V/\partial s = NN(s,t) for s
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # ALL FOLLOWING METHODS USE REAL UNITS

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        for periodic_dim in self.periodic_dims:
            wrapped_state[..., periodic_dim] = (wrapped_state[..., periodic_dim] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state 

    @abstractmethod
    def dsdt(self, state, control, disturbance, time):
        raise NotImplementedError
    
    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, time, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError
    

class ControlandDisturbanceAffineDynamics(Dynamics):
    def dsdt(self, state, control, disturbance, time):
        dsdt = self.open_loop_dynamics(state, time)
        dsdt += torch.bmm(self.control_jacobian(state, time), control.unsqueeze(-1)).squeeze(-1)
        dsdt += torch.bmm(self.disturbance_jacobian(state, time), disturbance.unsqueeze(-1)).squeeze(-1)
        return dsdt

    def __call__(self, state, control, disturbance, time):
        return self.dsdt(state, control, disturbance, time)

    @abstractmethod
    def open_loop_dynamics(self, state, time):
        raise NotImplementedError
    
    @abstractmethod
    def control_jacobian(self, state, time):
        raise NotImplementedError
    
    @abstractmethod
    def disturbance_jacobian(self, state, time):
        raise NotImplementedError


class ParameterizedVertDrone2D(Dynamics):
    def __init__(self, gravity:float, input_multiplier_max:float, input_magnitude_max:float):
        self.gravity = gravity                             # g
        self.input_multiplier_max = input_multiplier_max   # k_max
        self.input_magnitude_max = input_magnitude_max     # u_max
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 1.5, self.input_multiplier_max/2], # v, z, k
            state_var=[4, 2, self.input_multiplier_max/2],    # v, z, k
            periodic_dims=[],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-4, 4],                        # v
            [-0.5, 3.5],                    # z
            [0, self.input_multiplier_max], # k
        ]

    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    # \dot k = 0
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2]*control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        dsdt[..., 2] = 0
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        raise NotImplementedError

    def hamiltonian(self, state, time, dvds):
        return state[..., 2]*torch.abs(dvds[..., 0]*self.input_magnitude_max) \
                - dvds[..., 0]*self.gravity \
                + dvds[..., 1]*state[..., 0]
    
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    def plot_config(self):
        return {
            'state_slices': [0, 1.5, self.input_multiplier_max/2],
            'state_labels': ['v', 'z', 'k'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class SimpleAir3D(Dynamics):
    def __init__(self, collisionR:float, velocity:float, omega_max:float, angle_alpha_factor:float):
        self.collisionR = collisionR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=1,
            state_mean=[0, 0, 0], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi],
            periodic_dims=[2],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
        ]

    # Air3D dynamics
    # \dot x    = -v + v \cos \psi + u y
    # \dot y    = v \sin \psi - u x
    # \dot \psi = d - u
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = -self.velocity + self.velocity*torch.cos(state[..., 2]) + control[..., 0]*state[..., 1]
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2]) - control[..., 0]*state[..., 0]
        dsdt[..., 2] = disturbance[..., 0] - control[..., 0]
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, time, dvds):
        ham = self.omega_max * torch.abs(dvds[..., 0] * state[..., 1] - dvds[..., 1] * state[..., 0] - dvds[..., 2])  # Control component
        ham = ham - self.omega_max * torch.abs(dvds[..., 2])  # Disturbance component
        ham = ham + (self.velocity * (torch.cos(state[..., 2]) - 1.0) * dvds[..., 0]) + (self.velocity * torch.sin(state[..., 2]) * dvds[..., 1])  # Constant component
        return ham

    def optimal_control(self, state, dvds):
        det = dvds[..., 0]*state[..., 1] - dvds[..., 1]*state[..., 0]-dvds[..., 2]
        return (self.omega_max * torch.sign(det))[..., None]
    
    def optimal_disturbance(self, state, dvds):
        return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', 'theta'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class Air3D(ControlandDisturbanceAffineDynamics):
    def __init__(self, collisionR:float, evader_speed:float, pursuer_speed:float, evader_omega_max:float, 
                 pursuer_omega_max:float, angle_alpha_factor:float):
        self.collisionR = collisionR
        self.evader_speed = evader_speed
        self.pursuer_speed = pursuer_speed
        self.evader_omega_max = evader_omega_max
        self.pursuer_omega_max = pursuer_omega_max
        self.angle_alpha_factor = angle_alpha_factor
        from utils.boundary_functions import InputSet
        self.control_space = InputSet(lo=-torch.Tensor([self.evader_omega_max]), 
                                      hi=torch.Tensor([self.evader_omega_max]))
        self.disturbance_space = InputSet(lo=-torch.Tensor([self.pursuer_omega_max]), 
                                          hi=torch.Tensor([self.pursuer_omega_max]))
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=1,
            state_mean=[7, 0, 0], 
            state_var=[13, 10, self.angle_alpha_factor*math.pi],
            periodic_dims=[2],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-6, 20],
            [-10, 10],
            [-math.pi, math.pi],
        ]
    
    def open_loop_dynamics(self, state, time):
        open_loop_dynamics = torch.zeros_like(state)
        open_loop_dynamics[..., 0] = -self.evader_speed + self.pursuer_speed*torch.cos(state[..., 2])
        open_loop_dynamics[..., 1] = self.pursuer_speed*torch.sin(state[..., 2]) 
        open_loop_dynamics[..., 2] = 0
        return open_loop_dynamics
    
    def control_jacobian(self, state, time):
        control_jacobian = torch.zeros((*state.shape[:-1], self.state_dim, self.control_dim), device=state.device)
        control_jacobian[..., 0, 0] = state[..., 1]
        control_jacobian[..., 1, 0] = -state[..., 0]
        control_jacobian[..., 2, 0] = -1.0
        return control_jacobian

    def disturbance_jacobian(self, state, time):
        disturbance_jacobian = torch.zeros((*state.shape[:-1], self.state_dim, self.disturbance_dim), device=state.device)
        disturbance_jacobian[..., 0, 0] = 0
        disturbance_jacobian[..., 1, 0] = 0
        disturbance_jacobian[..., 2, 0] = 1.0
        return disturbance_jacobian

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, time, dvds):
        # opt_control = self.optimal_control(state, dvds).squeeze(0)
        # opt_disturbance = self.optimal_disturbance(state, dvds).squeeze(0)
        # flow = self.dsdt(state.squeeze(0), opt_control, opt_disturbance, time.squeeze(0))
        # return torch.sum(flow*dvds, dim=-1)
        ham = self.evader_omega_max * torch.abs(dvds[..., 0] * state[..., 1] - 
                                                dvds[..., 1] * state[..., 0] - 
                                                dvds[..., 2])  # Control component
        ham = ham - self.pursuer_omega_max * torch.abs(dvds[..., 2])  # Disturbance component
        ham = ham + ((self.pursuer_speed * torch.cos(state[..., 2]) - self.evader_speed) * dvds[..., 0] + 
                     (self.pursuer_speed * torch.sin(state[..., 2]) * dvds[..., 1]))  # Constant component
        return ham

    
    def optimal_control(self, state, dvds):
        det = dvds[..., 0]*state[..., 1] - dvds[..., 1]*state[..., 0]-dvds[..., 2]
        # return torch.where(det >= 0, self.evader_omega_max, -self.evader_omega_max)[..., None]
        return (self.evader_omega_max * torch.sign(det))[..., None]
    
    def optimal_disturbance(self, state, dvds):
        # return torch.where(dvds[..., 2] >= 0, -self.pursuer_omega_max, self.pursuer_omega_max)[..., None]
        return (-self.pursuer_omega_max * torch.sign(dvds[..., 2]))[..., None]

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', 'theta'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }


class Dubins3D(Dynamics):
    def __init__(self, goalR:float, velocity:float, omega_max:float, angle_alpha_factor:float, set_mode:str, freeze_model: bool):
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=[0, 0, 0], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi],
            periodic_dims=[2],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
        ]
        
    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, time, dvds):
        if self.freeze_model:
            raise NotImplementedError
        if self.set_mode == 'reach':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) 
        elif self.set_mode == 'avoid':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return (-self.omega_max*torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode == 'avoid':
            # return torch.where(dvds[..., 2] >= 0, self.omega_max, -self.omega_max)
            return (self.omega_max*torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }
    

class Dubins3DParameterizedDisturbance(Dynamics):
    def __init__(self, goalR:float, velocity:float, omega_max:float, angle_alpha_factor:float, set_mode:str, freeze_model: bool,
                 max_disturbance:float):
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        self.max_disturbance = max_disturbance
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=4, input_dim=5, control_dim=1, disturbance_dim=1,
            state_mean=[0, 0, 0, self.max_disturbance/2], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi, self.max_disturbance/2],
            periodic_dims=[2],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
            [0, self.max_disturbance],  # disturbance term
        ]
        
    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    # \dot \beta = 0
    def dsdt(self, state, control, disturbance, time):
        if self.freeze_model:
            raise NotImplementedError
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 2]) + state[..., 3] * disturbance[..., 0]
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        dsdt[..., 3] = 0 
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, time, dvds):
        if self.freeze_model:
            raise NotImplementedError
        if self.set_mode == 'reach':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) + state[..., 3] * torch.abs(dvds[..., 0])
        elif self.set_mode == 'avoid':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2]) - state[..., 3] * torch.abs(dvds[..., 0])

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return (-self.omega_max*torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode == 'avoid':
            return (self.omega_max*torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == 'reach':
            return (1 * state[..., 3] * torch.sign(dvds[..., 0]))[..., None]
        elif self.set_mode == 'avoid':
            return (-1 * state[..., 3] * torch.sign(dvds[..., 0]))[..., None]
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$', r'$\beta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': [2, 3],
            'angle_dims': [2],
        }

class Dubins4D(Dynamics):
    def __init__(self, bound_mode:str):
        self.vMin = 0.2
        self.vMax = 14.8
        self.collisionR = 1.5
        self.bound_mode = bound_mode
        assert self.bound_mode in ['v1', 'v2']

        xMean = 0
        yMean = 0
        thetaMean = 0
        vMean = 7.5
        aMean = 0
        oMean = 0

        xVar = 10
        yVar = 10
        thetaVar = 1.2*math.pi
        vVar = 7.5
        aVar = 10
        oVar = 3*math.pi if self.bound_mode == 'v1' else 2.0
        
        super().__init__(
            loss_type='brt_hjivi',
            state_dim=14, input_dim=15,  control_dim=2, disturbance_dim=0,
            state_mean=[xMean, yMean, thetaMean, vMean, xMean, yMean, aMean, aMean, oMean, oMean, aMean, aMean, oMean, oMean],
            state_var=[xVar, yVar, thetaVar, vVar, xVar, yVar, aVar, aVar, oVar, oVar, aVar, aVar, oVar, oVar],
            periodic_dims=[2],
            value_mean=13,
            value_var=14,
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
            [self.vMin, self.vMax],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
        ]

    def boundary_fn(self, state):
        return torch.norm(state[..., 0:2] - state[..., 4:6], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        raise NotImplementedError

    def dsdt(self, state, control, disturbance, time):
        raise NotImplementedError

    def hamiltonian(self, state, time, dvds):
        raise NotImplementedError

    def optimal_control(self, state, dvds):
        raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    def plot_config(self):
        raise NotImplementedError


class Quad2DAttitude(Dynamics):
    def __init__(self, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, max_pos_dist: float = 0.0, 
                 max_vel_dist: float = 0.0, set_mode: str='avoid'):
        self.gravity = gravity
        self.max_angle = max_angle
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        self.max_pos_dist = max_pos_dist
        self.max_vel_dist = max_vel_dist
        from utils import boundary_functions
        space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                        torch.Tensor([4.0, 2.5, 1.9, 1.9]))
        circle = boundary_functions.Circle([0, 1], 0.5, torch.Tensor([2.0, 1.5]))
        rectangle = boundary_functions.Rectangle([0, 1], torch.Tensor([-2.0, 0.5]), torch.Tensor([0.0, 1.5]))
        self.sdf = boundary_functions.build_sdf(space_boundary, [circle, rectangle])
        from utils.boundary_functions import InputSet
        # self.control_space = InputSet(lo=-self.evader_omega_max, hi=self.evader_omega_max)
        # self.disturbance_space = InputSet(lo=-self.pursuer_omega_max, hi=self.pursuer_omega_max)
        self.control_space = InputSet(lo=[-max_angle, min_thrust], hi=[max_angle, max_thrust])
        self.disturbance_space = InputSet(lo=[-max_pos_dist, -max_pos_dist, -max_vel_dist, -max_vel_dist], 
                                     hi=[max_pos_dist, max_pos_dist, max_vel_dist, max_vel_dist])
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=4, input_dim=5, control_dim=2, disturbance_dim=4,
            state_mean=[0., 1.3, 0, 0],
            state_var=[5., 1.5, 2, 2],
            periodic_dims=[],
            value_mean=0.2,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-5, 5], 
            [-0.2, 2.8],
            [-1.4, 1.4],
            [-1.4, 1.4]
        ]
    
    # Quadcopter Dynamics
    # \dot y = v_y + d_1
    # \dot z = v_z + d_2
    # \dot v_y = g * u_1 + d_3
    # \dot v_z = u_2 - g + d_4
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] + disturbance[..., 0]
        dsdt[..., 1] = state[..., 3] + disturbance[..., 1]
        dsdt[..., 2] = self.gravity * control[..., 0] + disturbance[..., 2]
        dsdt[..., 3] = control[..., 1] - self.gravity + disturbance[..., 3]
        return dsdt
    
    def open_loop_dynamics(self, state, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] 
        dsdt[..., 1] = state[..., 3] 
        dsdt[..., 2] = 0
        dsdt[..., 3] = - self.gravity 
        return dsdt

    def control_jacobian(self, state, time):
        control_jacobian = torch.zeros((*state.shape[:-1], self.state_dim, self.control_dim), device=state.device)
        # torch.tensor([
        #     [0., 0.],
        #     [0., 0.],
        #     [self.gravity, 0.],
        #     [0., 1.],
        # ])
        
        control_jacobian[..., 2, 0] = self.gravity
        control_jacobian[..., 3, 1] = 1.0
        return control_jacobian
        
    def disturbance_jacobian(self, state, time):
        disturbance_jacobian = torch.zeros((*state.shape[:-1], self.state_dim, self.disturbance_dim), device=state.device)
        # torch.tensor([
        #     [1., 0., 0., 0.],
        #     [0., 1., 0., 0.],
        #     [0., 0., 1., 0.],
        #     [0., 0., 0., 1.],
        # ])

        disturbance_jacobian[..., 0, 0] = 1.0
        disturbance_jacobian[..., 1, 1] = 1.0
        disturbance_jacobian[..., 2, 2] = 1.0
        disturbance_jacobian[..., 3, 3] = 1.0
        return disturbance_jacobian

    def boundary_fn(self, state):
        return self.sdf(state)

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, time, dvds):
        optimal_control = self.optimal_control(state, dvds)
        optimal_disturbance = self.optimal_disturbance(state, dvds)
        flow = self.dsdt(state, optimal_control, optimal_disturbance, time)
        return torch.sum(flow*dvds, dim=-1)
    
    def optimal_control(self, state, dvds):
        if self.set_mode == "avoid":
            # a1 = torch.sign(dvds[..., 2]) * self.max_angle
            # a2 = self.min_thrust + torch.sign(dvds[..., 3]) * (self.max_thrust - self.min_thrust)
            a1 = torch.where(dvds[..., 2] < 0, -self.max_angle, self.max_angle)
            a2 = torch.where(dvds[..., 3] < 0, self.min_thrust, self.max_thrust)
        elif self.set_mode == "reach":
            # a1 = -torch.sign(dvds[..., 2]) * self.max_angle
            # a2 = self.max_thrust - torch.sign(dvds[..., 3]) * (self.max_thrust - self.min_thrust)
            a1 = torch.where(dvds[..., 2] > 0, -self.max_angle, self.max_angle)
            a2 = torch.where(dvds[..., 3] > 0, self.min_thrust, self.max_thrust)
        else:
            raise NotImplementedError("{self.set_mode} is not a valid set mode")
        return torch.cat((a1[..., None], a2[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == "avoid":
            # d1 = -torch.sign(dvds[..., 0]) * self.max_pos_dist
            # d2 = -torch.sign(dvds[..., 1]) * self.max_pos_dist
            # d3 = -torch.sign(dvds[..., 2]) * self.max_vel_dist
            # d4 = -torch.sign(dvds[..., 3]) * self.max_vel_dist
            d1 = torch.where(dvds[..., 0] > 0, -self.max_pos_dist, self.max_pos_dist)
            d2 = torch.where(dvds[..., 1] > 0, -self.max_pos_dist, self.max_pos_dist)
            d3 = torch.where(dvds[..., 2] > 0, -self.max_vel_dist, self.max_vel_dist)
            d4 = torch.where(dvds[..., 3] > 0, -self.max_vel_dist, self.max_vel_dist)
        elif self.set_mode == "reach":
            # d1 = torch.sign(dvds[..., 0]) * self.max_pos_dist
            # d2 = torch.sign(dvds[..., 1]) * self.max_pos_dist
            # d3 = torch.sign(dvds[..., 2]) * self.max_vel_dist
            # d4 = torch.sign(dvds[..., 3]) * self.max_vel_dist
            d1 = torch.where(dvds[..., 0] < 0, -self.max_pos_dist, self.max_pos_dist)
            d2 = torch.where(dvds[..., 1] < 0, -self.max_pos_dist, self.max_pos_dist)
            d3 = torch.where(dvds[..., 2] < 0, -self.max_vel_dist, self.max_vel_dist)
            d4 = torch.where(dvds[..., 3] < 0, -self.max_vel_dist, self.max_vel_dist)
        else:
            raise NotImplementedError("{self.set_mode} is not a valid set mode")
        return torch.cat((d1[..., None], d2[..., None], d3[..., None], d4[..., None]), dim=-1)
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0],
            'state_labels': ['y', 'z', r'$v_y$', r'$v_z$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': [2, 3],
        }


class Quad2DAttitudeReachAvoid(Dynamics):
    def __init__(self, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, max_pos_dist: float = 0.0, 
                 max_vel_dist: float = 0.0, set_mode: str='avoid'):
        self.gravity = gravity
        self.max_angle = max_angle
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        self.max_pos_dist = max_pos_dist
        self.max_vel_dist = max_vel_dist
        from utils import boundary_functions
        space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                        torch.Tensor([4.0, 2.5, 1.9, 1.9]))
        circle = boundary_functions.Circle([0, 1], 0.5, torch.Tensor([2.0, 1.5]))
        rectangle = boundary_functions.Rectangle([0, 1], torch.Tensor([-2.0, 0.5]), torch.Tensor([0.0, 1.5]))
        self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [circle, rectangle])

        self.target_region = boundary_functions.Ellipse([0, 1, 2, 3], 1.0, [0.75, 1.0, 0.0, 0.0], [2.0, 1.0, 3.0, 3.0])
        self.sdf_reach = self.target_region.boundary_sdf
        
        from utils.boundary_functions import InputSet
        # self.control_space = InputSet(lo=-self.evader_omega_max, hi=self.evader_omega_max)
        # self.disturbance_space = InputSet(lo=-self.pursuer_omega_max, hi=self.pursuer_omega_max)
        self.control_space = InputSet(lo=[-max_angle, min_thrust], hi=[max_angle, max_thrust])
        self.disturbance_space = InputSet(lo=[-max_pos_dist, -max_pos_dist, -max_vel_dist, -max_vel_dist], 
                                     hi=[max_pos_dist, max_pos_dist, max_vel_dist, max_vel_dist])
        super().__init__(
            loss_type='brat_ci_hjivi', set_mode=set_mode,
            state_dim=4, input_dim=5, control_dim=2, disturbance_dim=4,
            state_mean=[0., 1.3, 0, 0],
            state_var=[5., 1.5, 2, 2],
            periodic_dims=[],
            value_mean=0.2,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="vanilla"
        )

    def state_test_range(self):
        return [
            [-5, 5], 
            [-0.2, 2.8],
            [-1.4, 1.4],
            [-1.4, 1.4]
        ]
    
    # Quadcopter Dynamics
    # \dot y = v_y + d_1
    # \dot z = v_z + d_2
    # \dot v_y = g * u_1 + d_3
    # \dot v_z = u_2 - g + d_4
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] + disturbance[..., 0]
        dsdt[..., 1] = state[..., 3] + disturbance[..., 1]
        dsdt[..., 2] = self.gravity * control[..., 0] + disturbance[..., 2]
        dsdt[..., 3] = control[..., 1] - self.gravity + disturbance[..., 3]
        return dsdt

    def reach_fn(self, state):
        # Not negated here, to be positive in the target region
        return self.sdf_reach(state)
    
    def avoid_fn(self, state):
        return self.sdf_avoid(state)
    
    def open_loop_dynamics(self, state, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] 
        dsdt[..., 1] = state[..., 3] 
        dsdt[..., 2] = 0
        dsdt[..., 3] = - self.gravity 
        return dsdt

    def control_jacobian(self, state, time):
        control_jacobian = torch.zeros((*state.shape[:-1], self.state_dim, self.control_dim), device=state.device)
        # torch.tensor([
        #     [0., 0.],
        #     [0., 0.],
        #     [self.gravity, 0.],
        #     [0., 1.],
        # ])
        
        control_jacobian[..., 2, 0] = self.gravity
        control_jacobian[..., 3, 1] = 1.0
        return control_jacobian
        
    def disturbance_jacobian(self, state, time):
        disturbance_jacobian = torch.zeros((*state.shape[:-1], self.state_dim, self.disturbance_dim), device=state.device)
        # torch.tensor([
        #     [1., 0., 0., 0.],
        #     [0., 1., 0., 0.],
        #     [0., 0., 1., 0.],
        #     [0., 0., 0., 1.],
        # ])

        disturbance_jacobian[..., 0, 0] = 1.0
        disturbance_jacobian[..., 1, 1] = 1.0
        disturbance_jacobian[..., 2, 2] = 1.0
        disturbance_jacobian[..., 3, 3] = 1.0
        return disturbance_jacobian

    def boundary_fn(self, state):
        return torch.minimum(self.avoid_fn(state), self.reach_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        reach_values = self.reach_fn(state_traj)
        avoid_values = self.avoid_fn(state_traj)
        # Situations:
        # 1. If we start in target outside of obstacle, at time 0 inner is > 0 -> cost > 0
        # 2. If we reach target before hitting obstacle, at that time cummin(avoid) > 0 & reach_values > 0 -> cost > 0
        # 3. If we hit obstacle before reaching target, at that time cummin(avoid) < 0 & reach_values > 0 -> cost < 0

        return torch.max(torch.minimum(reach_values, torch.cummin(avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, time, dvds):
        optimal_control = self.optimal_control(state, dvds)
        optimal_disturbance = self.optimal_disturbance(state, dvds)
        flow = self.dsdt(state, optimal_control, optimal_disturbance, time)
        return torch.sum(flow*dvds, dim=-1)
    
    def optimal_control(self, state, dvds):
        if self.set_mode == "avoid":
            # a1 = torch.sign(dvds[..., 2]) * self.max_angle
            # a2 = self.min_thrust + torch.sign(dvds[..., 3]) * (self.max_thrust - self.min_thrust)
            a1 = torch.where(dvds[..., 2] < 0, -self.max_angle, self.max_angle)
            a2 = torch.where(dvds[..., 3] < 0, self.min_thrust, self.max_thrust)
        elif self.set_mode == "reach":
            # a1 = -torch.sign(dvds[..., 2]) * self.max_angle
            # a2 = self.max_thrust - torch.sign(dvds[..., 3]) * (self.max_thrust - self.min_thrust)
            a1 = torch.where(dvds[..., 2] > 0, -self.max_angle, self.max_angle)
            a2 = torch.where(dvds[..., 3] > 0, self.min_thrust, self.max_thrust)
        else:
            raise NotImplementedError("{self.set_mode} is not a valid set mode")
        return torch.cat((a1[..., None], a2[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == "avoid":
            # d1 = -torch.sign(dvds[..., 0]) * self.max_pos_dist
            # d2 = -torch.sign(dvds[..., 1]) * self.max_pos_dist
            # d3 = -torch.sign(dvds[..., 2]) * self.max_vel_dist
            # d4 = -torch.sign(dvds[..., 3]) * self.max_vel_dist
            d1 = torch.where(dvds[..., 0] > 0, -self.max_pos_dist, self.max_pos_dist)
            d2 = torch.where(dvds[..., 1] > 0, -self.max_pos_dist, self.max_pos_dist)
            d3 = torch.where(dvds[..., 2] > 0, -self.max_vel_dist, self.max_vel_dist)
            d4 = torch.where(dvds[..., 3] > 0, -self.max_vel_dist, self.max_vel_dist)
        elif self.set_mode == "reach":
            # d1 = torch.sign(dvds[..., 0]) * self.max_pos_dist
            # d2 = torch.sign(dvds[..., 1]) * self.max_pos_dist
            # d3 = torch.sign(dvds[..., 2]) * self.max_vel_dist
            # d4 = torch.sign(dvds[..., 3]) * self.max_vel_dist
            d1 = torch.where(dvds[..., 0] < 0, -self.max_pos_dist, self.max_pos_dist)
            d2 = torch.where(dvds[..., 1] < 0, -self.max_pos_dist, self.max_pos_dist)
            d3 = torch.where(dvds[..., 2] < 0, -self.max_vel_dist, self.max_vel_dist)
            d4 = torch.where(dvds[..., 3] < 0, -self.max_vel_dist, self.max_vel_dist)
        else:
            raise NotImplementedError("{self.set_mode} is not a valid set mode")
        return torch.cat((d1[..., None], d2[..., None], d3[..., None], d4[..., None]), dim=-1)
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0],
            'state_labels': ['y', 'z', r'$v_y$', r'$v_z$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': [2, 3],
        }


class Quad2DAttitudeReachAvoidOriginal(Dynamics):
    def __init__(self, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, max_pos_dist: float = 0.0, 
                 max_vel_dist: float = 0.0, set_mode: str='reach'):
        self.gravity = gravity
        self.max_angle = max_angle
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        self.max_pos_dist = max_pos_dist
        self.max_vel_dist = max_vel_dist
        from utils import boundary_functions
        space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                        torch.Tensor([4.0, 2.5, 1.9, 1.9]))
        circle = boundary_functions.Circle([0, 1], 0.5, torch.Tensor([2.0, 1.5]))
        rectangle = boundary_functions.Rectangle([0, 1], torch.Tensor([-2.0, 0.5]), torch.Tensor([0.0, 1.5]))
        self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [circle, rectangle])  # Negative when in obstacle

        self.target_region = boundary_functions.Ellipse([0, 1, 2, 3], 1.0, [0.75, 1.0, 0.0, 0.0], [2.0, 1.0, 3.0, 3.0])
        self.sdf_reach = self.target_region.boundary_sdf  # Currently positive when in target region
        
        from utils.boundary_functions import InputSet
        # self.control_space = InputSet(lo=-self.evader_omega_max, hi=self.evader_omega_max)
        # self.disturbance_space = InputSet(lo=-self.pursuer_omega_max, hi=self.pursuer_omega_max)
        self.control_space = InputSet(lo=[-max_angle, min_thrust], hi=[max_angle, max_thrust])
        self.disturbance_space = InputSet(lo=[-max_pos_dist, -max_pos_dist, -max_vel_dist, -max_vel_dist], 
                                     hi=[max_pos_dist, max_pos_dist, max_vel_dist, max_vel_dist])
        super().__init__(
            loss_type='brat_hjivi', set_mode=set_mode,
            state_dim=4, input_dim=5, control_dim=2, disturbance_dim=4,
            state_mean=[0., 1.3, 0, 0],
            state_var=[5., 1.5, 2, 2],
            periodic_dims=[],
            value_mean=0.2,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="vanilla"
        )

    def state_test_range(self):
        return [
            [-5, 5], 
            [-0.2, 2.8],
            [-1.4, 1.4],
            [-1.4, 1.4]
        ]
    
    # Quadcopter Dynamics
    # \dot y = v_y + d_1
    # \dot z = v_z + d_2
    # \dot v_y = g * u_1 + d_3
    # \dot v_z = u_2 - g + d_4
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] + disturbance[..., 0]
        dsdt[..., 1] = state[..., 3] + disturbance[..., 1]
        dsdt[..., 2] = self.gravity * control[..., 0] + disturbance[..., 2]
        dsdt[..., 3] = control[..., 1] - self.gravity + disturbance[..., 3]
        return dsdt
    
    def open_loop_dynamics(self, state, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] 
        dsdt[..., 1] = state[..., 3] 
        dsdt[..., 2] = 0
        dsdt[..., 3] = - self.gravity 
        return dsdt

    def control_jacobian(self, state, time):
        control_jacobian = torch.zeros((*state.shape[:-1], self.state_dim, self.control_dim), device=state.device)
        # torch.tensor([
        #     [0., 0.],
        #     [0., 0.],
        #     [self.gravity, 0.],
        #     [0., 1.],
        # ])
        
        control_jacobian[..., 2, 0] = self.gravity
        control_jacobian[..., 3, 1] = 1.0
        return control_jacobian
        
    def disturbance_jacobian(self, state, time):
        disturbance_jacobian = torch.zeros((*state.shape[:-1], self.state_dim, self.disturbance_dim), device=state.device)
        # torch.tensor([
        #     [1., 0., 0., 0.],
        #     [0., 1., 0., 0.],
        #     [0., 0., 1., 0.],
        #     [0., 0., 0., 1.],
        # ])

        disturbance_jacobian[..., 0, 0] = 1.0
        disturbance_jacobian[..., 1, 1] = 1.0
        disturbance_jacobian[..., 2, 2] = 1.0
        disturbance_jacobian[..., 3, 3] = 1.0
        return disturbance_jacobian

    def reach_fn(self, state):
        # Negated to be negative in target region
        return -self.sdf_reach(state)
    
    def avoid_fn(self, state):
        return self.sdf_avoid(state)

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError  # FIXME: Do we need this?
    
    def cost_fn(self, state_traj):
        reach_values = self.reach_fn(state_traj)
        avoid_values = -self.avoid_fn(state_traj)
        # Situations:
        # 1. If we start in target outside of obstacle \exists reach_values < 0 and cummax(-avoid_values) < 0 -> cost < 0
        # 2. If we reach target before hitting obstacle \exists reach_values < 0 and cummax(-avoid_values) < 0 -> cost < 0
        # 3. If we hit obstacle before reaching target \exists reach_values > 0 and cummax(-avoid_values) > 0 -> cost > 0
        # 4. If we hit obstacle after reaching target \exists reach_values < 0 and cummax(-avoid_values) < 0 -> cost < 0
        return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, time, dvds):
        optimal_control = self.optimal_control(state, dvds)
        optimal_disturbance = self.optimal_disturbance(state, dvds)
        flow = self.dsdt(state, optimal_control, optimal_disturbance, time)
        return torch.sum(flow*dvds, dim=-1)
    
    def optimal_control(self, state, dvds):
        if self.set_mode == "avoid":
            # a1 = torch.sign(dvds[..., 2]) * self.max_angle
            # a2 = self.min_thrust + torch.sign(dvds[..., 3]) * (self.max_thrust - self.min_thrust)
            a1 = torch.where(dvds[..., 2] < 0, -self.max_angle, self.max_angle)
            a2 = torch.where(dvds[..., 3] < 0, self.min_thrust, self.max_thrust)
        elif self.set_mode == "reach":
            # a1 = -torch.sign(dvds[..., 2]) * self.max_angle
            # a2 = self.max_thrust - torch.sign(dvds[..., 3]) * (self.max_thrust - self.min_thrust)
            a1 = torch.where(dvds[..., 2] > 0, -self.max_angle, self.max_angle)
            a2 = torch.where(dvds[..., 3] > 0, self.min_thrust, self.max_thrust)
        else:
            raise NotImplementedError("{self.set_mode} is not a valid set mode")
        return torch.cat((a1[..., None], a2[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        if self.set_mode == "avoid":
            # d1 = -torch.sign(dvds[..., 0]) * self.max_pos_dist
            # d2 = -torch.sign(dvds[..., 1]) * self.max_pos_dist
            # d3 = -torch.sign(dvds[..., 2]) * self.max_vel_dist
            # d4 = -torch.sign(dvds[..., 3]) * self.max_vel_dist
            d1 = torch.where(dvds[..., 0] > 0, -self.max_pos_dist, self.max_pos_dist)
            d2 = torch.where(dvds[..., 1] > 0, -self.max_pos_dist, self.max_pos_dist)
            d3 = torch.where(dvds[..., 2] > 0, -self.max_vel_dist, self.max_vel_dist)
            d4 = torch.where(dvds[..., 3] > 0, -self.max_vel_dist, self.max_vel_dist)
        elif self.set_mode == "reach":
            # d1 = torch.sign(dvds[..., 0]) * self.max_pos_dist
            # d2 = torch.sign(dvds[..., 1]) * self.max_pos_dist
            # d3 = torch.sign(dvds[..., 2]) * self.max_vel_dist
            # d4 = torch.sign(dvds[..., 3]) * self.max_vel_dist
            d1 = torch.where(dvds[..., 0] < 0, -self.max_pos_dist, self.max_pos_dist)
            d2 = torch.where(dvds[..., 1] < 0, -self.max_pos_dist, self.max_pos_dist)
            d3 = torch.where(dvds[..., 2] < 0, -self.max_vel_dist, self.max_vel_dist)
            d4 = torch.where(dvds[..., 3] < 0, -self.max_vel_dist, self.max_vel_dist)
        else:
            raise NotImplementedError("{self.set_mode} is not a valid set mode")
        return torch.cat((d1[..., None], d2[..., None], d3[..., None], d4[..., None]), dim=-1)
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0],
            'state_labels': ['y', 'z', r'$v_y$', r'$v_z$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': [2, 3],
        }



class NarrowPassage(Dynamics):
    def __init__(self, avoid_fn_weight:float, avoid_only:bool):
        self.L = 2.0

        # # Target positions
        self.goalX = [6.0, -6.0]
        self.goalY = [-1.4, 1.4]

        # State bounds
        self.vMin = 0.001
        self.vMax = 6.50
        self.phiMin = -0.3*math.pi + 0.001
        self.phiMax = 0.3*math.pi - 0.001

        # Control bounds
        self.aMin = -4.0
        self.aMax = 2.0
        self.psiMin = -3.0*math.pi
        self.psiMax = 3.0*math.pi

        # Lower and upper curb positions (in the y direction)
        self.curb_positions = [-2.8, 2.8]

        # Stranded car position
        self.stranded_car_pos = [0.0, -1.8]

        self.avoid_fn_weight = avoid_fn_weight

        self.avoid_only = avoid_only

        super().__init__(
            loss_type='brt_hjivi' if self.avoid_only else 'brat_hjivi', set_mode='avoid' if self.avoid_only else 'reach',
            state_dim=10, input_dim=11, control_dim=4, disturbance_dim=0,
            # state = [x1, y1, th1, v1, phi1, x2, y2, th2, v2, phi2]
            state_mean=[
                0, 0, 0, 3, 0, 
                0, 0, 0, 3, 0
            ],
            state_var=[
                8.0, 3.8, 1.2*math.pi, 4.0, 1.2*0.3*math.pi, 
                8.0, 3.8, 1.2*math.pi, 4.0, 1.2*0.3*math.pi,
            ],
            periodic_dims=[2, 4, 7, 9],
            value_mean=0.25*8.0,
            value_var=0.5*8.0,
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-8, 8],
            [-3.8, 3.8],
            [-math.pi, math.pi],
            [-1, 7],
            [-0.3*math.pi, 0.3*math.pi],
            [-8, 8],
            [-3.8, 3.8],
            [-math.pi, math.pi],
            [-1, 7],
            [-0.3*math.pi, 0.3*math.pi],
        ]

    # NarrowPassage dynamics
    # \dot x   = v * cos(th)
    # \dot y   = v * sin(th)
    # \dot th  = v * tan(phi) / L
    # \dot v   = u1
    # \dot phi = u2
    # \dot x   = ...
    # \dot y   = ...
    # \dot th  = ...
    # \dot v   = ...
    # \dot phi = ...
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]*torch.cos(state[..., 2])
        dsdt[..., 1] = state[..., 3]*torch.sin(state[..., 2])
        dsdt[..., 2] = state[..., 3]*torch.tan(state[..., 4]) / self.L
        dsdt[..., 3] = control[..., 0]
        dsdt[..., 4] = control[..., 1]
        dsdt[..., 5] = state[..., 8]*torch.cos(state[..., 7])
        dsdt[..., 6] = state[..., 8]*torch.sin(state[..., 7])
        dsdt[..., 7] = state[..., 8]*torch.tan(state[..., 9]) / self.L
        dsdt[..., 8] = control[..., 2]
        dsdt[..., 9] = control[..., 3]
        return dsdt

    def reach_fn(self, state):
        if self.avoid_only:
            raise RuntimeError
        # vehicle 1
        goal_tensor_R1 = torch.tensor([self.goalX[0], self.goalY[0]], device=state.device)
        dist_R1 = torch.norm(state[..., 0:2] - goal_tensor_R1, dim=-1) - self.L
        # vehicle 2
        goal_tensor_R2 = torch.tensor([self.goalX[1], self.goalY[1]], device=state.device)
        dist_R2 = torch.norm(state[..., 5:7] - goal_tensor_R2, dim=-1) - self.L
        return torch.maximum(dist_R1, dist_R2)
    
    def avoid_fn(self, state):
        # distance from lower curb
        dist_lc_R1 = state[..., 1] - self.curb_positions[0] - 0.5*self.L
        dist_lc_R2 = state[..., 6] - self.curb_positions[0] - 0.5*self.L
        dist_lc = torch.minimum(dist_lc_R1, dist_lc_R2)
        
        # distance from upper curb
        dist_uc_R1 = self.curb_positions[1] - state[..., 1] - 0.5*self.L
        dist_uc_R2 = self.curb_positions[1] - state[..., 6] - 0.5*self.L
        dist_uc = torch.minimum(dist_uc_R1, dist_uc_R2)
        
        # distance from the stranded car
        stranded_car_pos = torch.tensor(self.stranded_car_pos, device=state.device)
        dist_stranded_R1 = torch.norm(state[..., 0:2] - stranded_car_pos, dim=-1) - self.L
        dist_stranded_R2 = torch.norm(state[..., 5:7] - stranded_car_pos, dim=-1) - self.L
        dist_stranded = torch.minimum(dist_stranded_R1, dist_stranded_R2)

        # distance between the vehicles themselves
        dist_R1R2 = torch.norm(state[..., 0:2] - state[..., 5:7], dim=-1) - self.L

        return self.avoid_fn_weight * torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2)

    def boundary_fn(self, state):
        if self.avoid_only:
            return self.avoid_fn(state)
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):    
        if self.avoid_only:
            return torch.min(self.avoid_fn(state_traj), dim=-1).values
        else:   
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn 
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, time, dvds):
        optimal_control = self.optimal_control(state, dvds)
        return state[..., 3] * torch.cos(state[..., 2]) * dvds[..., 0] + \
               state[..., 3] * torch.sin(state[..., 2]) * dvds[..., 1] + \
               state[..., 3] * torch.tan(state[..., 4]) * dvds[..., 2] / self.L + \
               optimal_control[..., 0] * dvds[..., 3] + \
               optimal_control[..., 1] * dvds[..., 4] + \
               state[..., 8] * torch.cos(state[..., 7]) * dvds[..., 5] + \
               state[..., 8] * torch.sin(state[..., 7]) * dvds[..., 6] + \
               state[..., 8] * torch.tan(state[..., 9]) * dvds[..., 7] / self.L + \
               optimal_control[..., 2] * dvds[..., 8] + \
               optimal_control[..., 3] * dvds[..., 9]

    def optimal_control(self, state, dvds):
        a1_min = self.aMin * (state[..., 3] > self.vMin)
        a1_max = self.aMax * (state[..., 3] < self.vMax)
        psi1_min = self.psiMin * (state[..., 4] > self.phiMin)
        psi1_max = self.psiMax * (state[..., 4] < self.phiMax)
        a2_min = self.aMin * (state[..., 8] > self.vMin)
        a2_max = self.aMax * (state[..., 8] < self.vMax)
        psi2_min = self.psiMin * (state[..., 9] > self.phiMin)
        psi2_max = self.psiMax * (state[..., 9] < self.phiMax)

        if self.avoid_only:
            a1 = torch.where(dvds[..., 3] < 0, a1_min, a1_max)
            psi1 = torch.where(dvds[..., 4] < 0, psi1_min, psi1_max)
            a2 = torch.where(dvds[..., 8] < 0, a2_min, a2_max)
            psi2 = torch.where(dvds[..., 9] < 0, psi2_min, psi2_max)

        else:
            a1 = torch.where(dvds[..., 3] > 0, a1_min, a1_max)
            psi1 = torch.where(dvds[..., 4] > 0, psi1_min, psi1_max)
            a2 = torch.where(dvds[..., 8] > 0, a2_min, a2_max)
            psi2 = torch.where(dvds[..., 9] > 0, psi2_min, psi2_max)

        return torch.cat((a1[..., None], psi1[..., None], a2[..., None], psi2[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [
                -6.0, -1.4, 0.0, 6.5, 0.0, 
                -6.0, 1.4, -math.pi, 0.0, 0.0
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$', r'$\theta_1$', r'$v_1$', r'$\phi_1$',
                r'$x_2$', r'$y_2$', r'$\theta_2$', r'$v_2$', r'$\phi_2$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class ReachAvoidRocketLanding(Dynamics):
    def __init__(self):
        super().__init__(
            loss_type='brat_hjivi', set_mode='reach',
            state_dim=6, input_dim=7, control_dim=2, disturbance_dim=0,
            state_mean=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            state_var=[150.0, 70.0, 1.2*math.pi, 200.0, 200.0, 10.0],
            periodic_dims=[2],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            deepreach_model="exact",
        )

    def state_test_range(self):
        return [
            [-150, 150],
            [10, 150],
            [-math.pi, math.pi],
            [-200, 200],
            [-200, 200],
            [-10, 10],
        ]

    # \dot x = v_x
    # \dot y = v_y
    # \dot th = w
    # \dot v_x = u1 * cos(th) - u2 sin(th)
    # \dot v_y = u1 * sin(th) + u2 cos(th) - 9.81
    # \dot w = 0.3 * u1
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]
        dsdt[..., 1] = state[..., 4]
        dsdt[..., 2] = state[..., 5]
        dsdt[..., 3] = control[..., 0]*torch.cos(state[..., 2]) - control[..., 1]*torch.sin(state[..., 2])
        dsdt[..., 4] = control[..., 0]*torch.sin(state[..., 2]) + control[..., 1]*torch.cos(state[..., 2]) - 9.81
        dsdt[..., 5] = 0.3*control[..., 0]
        return dsdt

    def reach_fn(self, state):
        # Only target set in the xy direction
        # Target set position in x direction
        dist_x = torch.abs(state[..., 0]) - 20.0 #[-20, 150] boundary_fn range

        # Target set position in y direction
        dist_y = state[..., 1] - 20.0  #[-10, 130] boundary_fn range

        # First compute the target function as you normally would but then normalize it later.
        max_dist = torch.max(dist_x, dist_y)
        return torch.where((max_dist >= 0), max_dist/150.0, max_dist/10.0)

    def avoid_fn(self, state):
        # distance to floor
        dist_y = state[..., 1]

        # distance to wall
        wall_left = -30
        wall_right = -20
        wall_bottom = 0
        wall_top = 100
        dist_left = wall_left - state[..., 0]
        dist_right = state[..., 0] - wall_right
        dist_bottom = wall_bottom - state[..., 1]
        dist_top = state[..., 1] - wall_top
        dist_wall_x = torch.max(dist_left, dist_right)
        dist_wall_y = torch.max(dist_bottom, dist_top)
        dist_wall = torch.norm(torch.cat((torch.max(torch.tensor(0), dist_wall_x).unsqueeze(-1), torch.max(torch.tensor(0), dist_wall_y).unsqueeze(-1)), dim=-1), dim=-1) + torch.min(torch.tensor(0), torch.max(dist_wall_x, dist_wall_y))

        return torch.min(dist_y, dist_wall)

    def boundary_fn(self, state):
        return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-20, 20] # y in [-20, 20]
        target_state_range[1] = [10, 20]  # z in [10, 20]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn 
        reach_values = self.reach_fn(state_traj)
        avoid_values = self.avoid_fn(state_traj)
        return torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values

    def hamiltonian(self, state, time, dvds):
        # Control Hamiltonian
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        ham_ctrl = -250.0 * torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)
        # Constant Hamiltonian
        ham_constant = dvds[..., 0] * state[..., 3] + dvds[..., 1] * state[..., 4] + \
                      dvds[..., 2] * state[..., 5]  - dvds[..., 4] * 9.81
        # Compute the Hamiltonian
        ham_vehicle = ham_ctrl + ham_constant
        return ham_vehicle

    def optimal_control(self, state, dvds):
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
        return torch.cat((250.0 * torch.cos(opt_angle)[..., None], 250.0 * torch.sin(opt_angle)[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [-100, 120, 0, 150, -5, 0.0],
            'state_labels': ['x', 'y', r'$\theta$', r'$v_x$', r'$v_y$', r'$\omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }

class RocketLanding(Dynamics):
    def __init__(self):
        super().__init__(
            loss_type='brt_hjivi', set_mode='reach',
            state_dim=6, input_dim=8, control_dim=2, disturbance_dim=0,
            state_mean=[0.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            state_var=[150.0, 70.0, 1.2*math.pi, 200.0, 200.0, 10.0],
            periodic_dims=[2],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            deepreach_model="exact",
        )

    # convert model input to real coord
    def input_to_coord(self, input):
        input = input[..., :-1]
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        input = torch.cat((input, torch.zeros((*input.shape[:-1], 1), device=input.device)), dim=-1)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepreach_model=="diff":
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)[..., :-1]

        if self.deepreach_model=="diff":
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)


    def state_test_range(self):
        return [
            [-150, 150],
            [10, 150],
            [-math.pi, math.pi],
            [-200, 200],
            [-200, 200],
            [-10, 10],
        ]

    # \dot x = v_x
    # \dot y = v_y
    # \dot th = w
    # \dot v_x = u1 * cos(th) - u2 sin(th)
    # \dot v_y = u1 * sin(th) + u2 cos(th) - 9.81
    # \dot w = 0.3 * u1
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3]
        dsdt[..., 1] = state[..., 4]
        dsdt[..., 2] = state[..., 5]
        dsdt[..., 3] = control[..., 0]*torch.cos(state[..., 2]) - control[..., 1]*torch.sin(state[..., 2])
        dsdt[..., 4] = control[..., 0]*torch.sin(state[..., 2]) + control[..., 1]*torch.cos(state[..., 2]) - 9.81
        dsdt[..., 5] = 0.3*control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        # Only target set in the yz direction
        # Target set position in y direction
        dist_y = torch.abs(state[..., 0]) - 20.0 #[-20, 150] boundary_fn range

        # Target set position in z direction
        dist_z = state[..., 1] - 20.0  #[-10, 130] boundary_fn range

        # First compute the l(x) as you normally would but then normalize it later.
        lx = torch.max(dist_y, dist_z)
        return torch.where((lx >= 0), lx/150.0, lx/10.0)

    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-20, 20] # y in [-20, 20]
        target_state_range[1] = [10, 20]  # z in [10, 20]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, time, dvds):
        # Control Hamiltonian
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        ham_ctrl = -250.0 * torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)
        # Constant Hamiltonian
        ham_constant = dvds[..., 0] * state[..., 3] + dvds[..., 1] * state[..., 4] + \
                      dvds[..., 2] * state[..., 5]  - dvds[..., 4] * 9.81
        # Compute the Hamiltonian
        ham_vehicle = ham_ctrl + ham_constant
        return ham_vehicle
    
    def optimal_control(self, state, dvds):
        u1_coeff = dvds[..., 3] * torch.cos(state[..., 2]) + dvds[..., 4] * torch.sin(state[..., 2]) + 0.3 * dvds[..., 5]
        u2_coeff = -dvds[..., 3] * torch.sin(state[..., 2]) + dvds[..., 4] * torch.cos(state[..., 2])
        opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
        return torch.cat((250.0 * torch.cos(opt_angle)[..., None], 250.0 * torch.sin(opt_angle)[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [-100, 120, 0, 150, -5, 0.0],
            'state_labels': ['x', 'y', r'$\theta$', r'$v_x$', r'$v_y$', r'$\omega'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }

class Quadrotor(Dynamics):
    def __init__(self, collisionR:float, thrust_max:float, set_mode:str):
        self.thrust_max = thrust_max
        self.m=1 #mass
        self.arm_l=0.17
        self.CT=1
        self.CM=0.016
        self.Gz=-9.8

        self.thrust_max = thrust_max
        self.collisionR = collisionR


        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=0,
            state_mean=[0 for i in range(13)], 
            state_var=[1.5, 1.5, 1.5, 1, 1, 1, 1, 10, 10 ,10 ,10 ,10 ,10],
            periodic_dims=[],
            value_mean=(math.sqrt(1.5**2+1.5**2+1.5**2)-2*self.collisionR)/2, 
            value_var=math.sqrt(1.5**2+1.5**2+1.5**2), 
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-10, 10],
            [-10, 10],
            [-10, 10],
            [-10, 10],
            [-10, 10],
            [-10, 10],
        ]

    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance, time):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        u1 = control[...,0] * 1.0
        u2 = control[...,1] * 1.0
        u3 = control[...,2] * 1.0
        u4 = control[...,3] * 1.0


        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx*qx+wy*qy+wz*qz)/2.0 
        dsdt[..., 4] =  (wx*qw+wz*qy-wy*qz)/2.0
        dsdt[..., 5] = (wy*qw-wz*qx+wx*qz)/2.0
        dsdt[..., 6] = (wz*qw+wy*qx-wx*qy)/2.0
        dsdt[..., 7] = 2*(qw*qy+qx*qz)*self.CT/self.m*(u1+u2+u3+u4)
        dsdt[..., 8] =2*(-qw*qx+qy*qz)*self.CT/self.m*(u1+u2+u3+u4)
        dsdt[..., 9] =self.Gz+(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m*(u1+u2+u3+u4)
        dsdt[..., 10] = 4*math.sqrt(2)*self.CT*(u1-u2-u3+u4)/(3*self.arm_l*self.m)-5*wy*wz/9.0
        dsdt[..., 11] = 4*math.sqrt(2)*self.CT*(-u1-u2+u3+u4)/(3*self.arm_l*self.m)+5*wx*wz/9.0
        dsdt[..., 12] =12*self.CT*self.CM/(7*self.arm_l**2*self.m)*(u1-u2+u3-u4)
        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :3], dim=-1) - self.collisionR

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, time, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0


            C1=2*(qw*qy+qx*qz)*self.CT/self.m
            C2=2*(-qw*qx+qy*qz)*self.CT/self.m
            C3=(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m
            C4=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
            C5=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
            C6=12*self.CT*self.CM/(7*self.arm_l**2*self.m)

            # Compute the hamiltonian for the quadrotor
            ham= dvds[..., 0]*vx + dvds[..., 1]*vy+ dvds[..., 2]*vz
            ham+= -dvds[..., 3]* (wx*qx+wy*qy+wz*qz)/2.0 
            ham+= dvds[..., 4]*(wx*qw+wz*qy-wy*qz)/2.0
            ham+= dvds[..., 5]*(wy*qw-wz*qx+wx*qz)/2.0
            ham+= dvds[..., 6]*(wz*qw+wy*qx-wx*qy)/2.0
            ham+= dvds[..., 9]*-9.8
            ham+= -dvds[..., 10]*5*wy*wz/9.0+ dvds[..., 11]*5*wx*wz/9.0

            ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                +dvds[..., 10]*C4-dvds[..., 11]*C5+dvds[..., 12]*C6)*self.thrust_max

            ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                -dvds[..., 10]*C4-dvds[..., 11]*C5-dvds[..., 12]*C6)*self.thrust_max

            ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                -dvds[..., 10]*C4+dvds[..., 11]*C5+dvds[..., 12]*C6)*self.thrust_max

            ham+=torch.abs(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                +dvds[..., 10]*C4+dvds[..., 11]*C5-dvds[..., 12]*C6)*self.thrust_max

            return ham

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0


            C1=2*(qw*qy+qx*qz)*self.CT/self.m
            C2=2*(-qw*qx+qy*qz)*self.CT/self.m
            C3=(1-2*torch.pow(qx,2)-2*torch.pow(qy,2))*self.CT/self.m
            C4=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
            C5=4*math.sqrt(2)*self.CT/(3*self.arm_l*self.m)
            C6=12*self.CT*self.CM/(7*self.arm_l**2*self.m)


            u1=self.thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                +dvds[..., 10]*C4-dvds[..., 11]*C5+dvds[..., 12]*C6)
            u2=self.thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                -dvds[..., 10]*C4-dvds[..., 11]*C5-dvds[..., 12]*C6)
            u3=self.thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                -dvds[..., 10]*C4+dvds[..., 11]*C5+dvds[..., 12]*C6)
            u4=self.thrust_max*torch.sign(dvds[..., 7]*C1+dvds[..., 8]*C2+dvds[..., 9]*C3
                +dvds[..., 10]*C4+dvds[..., 11]*C5-dvds[..., 12]*C6)

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 2,
            'z_axis_idx': 7,
        }

class MultiVehicleCollision(Dynamics):
    def __init__(self):
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=9, input_dim=10, control_dim=3, disturbance_dim=0,
            periodic_dims=[6, 7, 8],
            state_mean=[
                0, 0,
                0, 0, 
                0, 0,
                0, 0, 0,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi,
            ],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],           
        ]
        
    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 6])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 6])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 7])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 7])
        dsdt[..., 4] = self.velocity*torch.cos(state[..., 8])
        dsdt[..., 5] = self.velocity*torch.sin(state[..., 8])
        dsdt[..., 6] = control[..., 0]
        dsdt[..., 7] = control[..., 1]
        dsdt[..., 8] = control[..., 2]
        return dsdt
    
    def boundary_fn(self, state):
        boundary_values = torch.norm(state[..., 0:2] - state[..., 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] - state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)
        return boundary_values

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, time, dvds):
        # Compute the hamiltonian for the ego vehicle
        ham = self.velocity*(torch.cos(state[..., 6]) * dvds[..., 0] + torch.sin(state[..., 6]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 6])
        # Hamiltonian effect due to other vehicles
        ham += self.velocity*(torch.cos(state[..., 7]) * dvds[..., 2] + torch.sin(state[..., 7]) * dvds[..., 3]) + self.omega_max * torch.abs(dvds[..., 7])
        ham += self.velocity*(torch.cos(state[..., 8]) * dvds[..., 4] + torch.sin(state[..., 8]) * dvds[..., 5]) + self.omega_max * torch.abs(dvds[..., 8])
        return ham

    def optimal_control(self, state, dvds):
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [
                0, 0, 
                -0.4, 0, 
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4,
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }



#################### Nikhil: Additional Dynamics ####################

class Drone4D(Dynamics): 

    def __init__(self, set_mode:str, gravity: float, min_angle: float, max_angle: float, min_thrust: float, max_thrust: float, 
                max_pos_y_dist: float, max_pos_z_dist: float, max_vel_y_dist: float, max_vel_z_dist: float): 

        # Input bounds
        self.u1_min = min_angle # default -math.pi/8
        self.u1_max = max_angle # default math.pi/8
        self.u2_min = min_thrust # default 5
        self.u2_max = max_thrust # default 13

        # Disturbance bounds 
        self.d1_max = max_pos_y_dist # default 0
        self.d2_max = max_pos_z_dist # default 0
        self.d3_max = max_vel_y_dist # default 0
        self.d4_max = max_vel_z_dist # default 0

        # Boundaries 
        from utils import boundary_functions
        space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                        torch.Tensor([4.0, 2.5, 1.9, 1.9]))
        circle = boundary_functions.Circle([0, 1], 0.5, torch.Tensor([2.0, 1.5]))
        rectangle = boundary_functions.Rectangle([0, 1], torch.Tensor([-2.0, 0.5]), torch.Tensor([0.0, 1.5]))
        self.sdf = boundary_functions.build_sdf(space_boundary, [circle, rectangle])

        # Constants
        self.gravity = gravity # default 9.81

        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=4, input_dim=5, control_dim=2, disturbance_dim=4,
            state_mean=[0., 1.3, 0, 0], # NOTE: might want to change
            state_var=[5., 1.5, 2, 2],  # NOTE: might want to change
            value_mean=0.2,             # NOTE: might want to change
            value_var=0.5,              # NOTE: might want to change
            value_normto=0.02,          # NOTE: might want to change
            deepreach_model="vanilla", # NOTE: Was "exact" before, "vanilla" worked better on the attitude model with these boundary functions 
            periodic_dims=[]
        )


    def state_test_range(self):
        return [
            [-5, 5], 
            [-0.2, 2.8],
            [-1.4, 1.4],
            [-1.4, 1.4]
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state) # no wrapping needed
        return wrapped_state
    
    # Drone 4D dynamics 
    # \dot y = vy + d1
    # \dot z = vz + d2
    # \dot vy = u2 sin(u1) + d3
    # \dot vz = u2 cos(u1) - g + d4
    # state = [y, z, vy, vz]
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] + disturbance[..., 0]
        dsdt[..., 1] = state[..., 3] + disturbance[..., 1]
        dsdt[..., 2] = control[..., 1]*torch.sin(control[..., 0]) + disturbance[..., 2]
        dsdt[..., 3] = control[..., 1]*torch.cos(control[..., 0]) - self.gravity + disturbance[..., 3]
        return dsdt
    
    def boundary_fn(self, state):
        # safety related bounds 
        return self.sdf(state) 
    
    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, time, dvds):
        # Compute the hamiltonian for the drone
        # args: state (tensor), dvds (tensor) - derivative of value function wrt state
        # TODO
        p1 = dvds[..., 0]
        p2 = dvds[..., 1]
        p3 = dvds[..., 2]
        p4 = dvds[..., 3]

        device = p3.device

        u1, u2 = self.optimal_control(state, dvds)
        d1, d2, d3, d4 = self.optimal_disturbance(state, dvds)

        # Compute the Hamiltonian 
        y = state[..., 0]
        z = state[..., 1]
        vy = state[..., 2]
        vz = state[..., 3]

        ham = (p1 * vy) + (p2 * vz) - (p4 * self.gravity) + \
              (p1 * d1) + (p2 * d2) + (p3 * d3) + (p4 * d4) + \
              (p3 * u2 * torch.sin(u1)) + (p4 * u2 * torch.cos(u1))

        return ham
    
    def optimal_control(self, state, dvds):
        p3 = dvds[..., 2]
        p4 = dvds[..., 3]

        device = p3.device

        u1 = torch.ones(p3.shape, device=device) 
        u2 = torch.ones(p4.shape, device=device) 

        # First: Maximize with u1 and u2 
        # Go through cases 
        arctan_p3p4 = torch.atan(p3/(p4 + torch.tensor(torch.finfo(torch.float).eps, device=device)))

        # Case 1: p3 > 0, p4 > 0 
        u1[torch.where(torch.logical_and(p3 > 0, p4 > 0))] = torch.min(arctan_p3p4[torch.where(torch.logical_and(p3 > 0, p4 > 0))], torch.tensor(self.u1_max, device=device))

        # Case 2: p3  > 0, p4 < 0
        u1[torch.where(torch.logical_and(p3 > 0, p4 < 0))] = torch.tensor(self.u1_max, device=device)

        # Case 3: p3 < 0, p4 > 0
        u1[torch.where(torch.logical_and(p3 < 0, p4 > 0))] = torch.max(arctan_p3p4[torch.where(torch.logical_and(p3 < 0, p4 > 0))], torch.tensor(self.u1_min, device=device))

        # Case 4: p3 < 0, p4 < 0
        u1[torch.where(torch.logical_and(p3 < 0, p4 < 0))] = torch.tensor(self.u1_min, device=device)

        # u2: select u2 max if g(u1) > 0 else select u2 min
        g_u1 = p3*torch.sin(u1) + p4*torch.cos(u1)
        u2 = u2 * self.u2_max
        u2[torch.where(g_u1 < 0)] = self.u2_min

        opt_control = torch.cat((u1[..., None], u2[..., None]), dim=-1)
        return opt_control 

    
    def optimal_disturbance(self, state, dvds):
        p1 = dvds[..., 0]
        p2 = dvds[..., 1]
        p3 = dvds[..., 2]
        p4 = dvds[..., 3]

        # Second: Minimize with d1, d2, d3, d4
        d1 = torch.ones(p1.shape) * self.d1_min
        d2 = torch.ones(p2.shape) * self.d2_min
        d3 = torch.ones(p3.shape) * self.d3_min
        d4 = torch.ones(p4.shape) * self.d4_min

        # If pi < 0 then di = di_min else di_max
        d1[torch.where(p1 < 0)] = self.d1_max
        d2[torch.where(p2 < 0)] = self.d2_max   
        d3[torch.where(p3 < 0)] = self.d3_max
        d4[torch.where(p4 < 0)] = self.d4_max

        opt_disturbance = torch.cat((d1[..., None], d2[..., None], d3[..., None], d4[..., None]), dim=-1)
        return opt_disturbance 
    
    def plot_config(self):
        # NOTE: might be incorrect
        return {
            'state_slices': [0, 0, 0, 0],
            'state_labels': ['y', 'z', r'$v_y$', r'$v_z$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': [2, 3],
        }


#################### Nikhil: Parametric Dynamics ####################
"""
Design choices: 
1. Add the parameters as additional inputs at the end of the state 
2. The parameter dimensions and corresponding input names (for hj reachability) are specified with attributes: 
    2.1 parametric_dims: indices of the parametric values in the state vector
    2.2 parametric_names: names of the parametric values - used to specify / vary parameter names for ground truth hj reachability
    2.3 state_dims: indices corresponding to the state / non parametric values of the state vector
3. parameter_test_slices: list of lists: where sublists are each full parameter configuration you want to evaluate
"""

class Quad2DAttitude_parametric(Dynamics):
    def __init__(self, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, set_mode: str='avoid', 
                 max_pos_dist: float = 0.0, max_vel_dist: float = 0.0):
        """
        args: 
            - max_pos_dist: maximum disturbance in position
            - max_vel_dist: maximum disturbance in velocity
        The maximum bounds specified here are the maximum for the parametric disturbance. So these are the worst case the parametric disturbance can take. 

        The max disturbance bounds are parametric and fed in as the last 2 parts of the state vector: 
        max_pos_dist, max_vel_dist 
        """
        import numpy as np 

        self.gravity = gravity
        self.max_angle = max_angle
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust

        self.max_pos_dist = max_pos_dist
        self.max_vel_dist = max_vel_dist

        # Parametric dynamics specific
        self.parametric_dims = [4, 5] # indices of the parametric values in the state vector
        self.parametric_names = ['max_pos_dist', 'max_vel_dist']
        self.coord_parametric_dims = list(np.array(self.parametric_dims) + 1) # indices corresponding to parametric dimensions in input coords - time is added as the 0th input
        self.state_dims = [0, 1, 2, 3] # indices corresponding to the state dimensions
        self.coord_state_dims = list(np.array(self.state_dims) + 1)# indices corresponding to state dimensions in input coords - time is added as the 0th input

        from utils import boundary_functions
        space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                        torch.Tensor([4.0, 2.5, 1.9, 1.9]))
        circle = boundary_functions.Circle([0, 1], 0.5, torch.Tensor([2.0, 1.5]))
        rectangle = boundary_functions.Rectangle([0, 1], torch.Tensor([-2.0, 0.5]), torch.Tensor([0.0, 1.5]))
        self.sdf = boundary_functions.build_sdf(space_boundary, [circle, rectangle])

        # Additional 2 dimensions for the parameteric disturbance bounds 
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=4+2, input_dim=5+2, control_dim=2, disturbance_dim=4,
            state_mean=[0., 1.3, 0, 0,    self.max_pos_dist/2, self.max_vel_dist/2],
            state_var=[5., 1.5, 2, 2,     (self.max_pos_dist/2) + 0.05, (self.max_vel_dist/2) + 0.05], # NOTE: try increasing range to capture 0 and max
            value_mean=0.2,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact", 
            periodic_dims=[],
        )

    def state_test_range(self):
        return [
            [-5, 5], 
            [-0.2, 2.8],
            [-1.4, 1.4],
            [-1.4, 1.4],
            #
            [self.max_pos_dist, self.max_pos_dist], #[0, self.max_pos_dist], # only test in worst case 
            [self.max_vel_dist, self.max_vel_dist] #[0, self.max_vel_dist]
        ]
    
    def parameter_test_slices(self): 
        """
        The parameter slices to evaluate and plot with
        """
        # return [[0., 0., ], 
        #         [self.max_pos_dist/2, self.max_vel_dist/2], 
        #         [self.max_pos_dist/2, self.max_vel_dist], 
        #         [self.max_pos_dist, self.max_vel_dist/2], 
        #         [self.max_pos_dist, self.max_vel_dist]]

        # return [[0., 0., ],]
        return [[0., 0., ], 
                [self.max_pos_dist/2, self.max_vel_dist/2], 
                [self.max_pos_dist/2, self.max_vel_dist], 
                [self.max_pos_dist, self.max_vel_dist]]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state
    
    def dsdt(self, state, control, disturbance, time):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] + disturbance[..., 0]
        dsdt[..., 1] = state[..., 3] + disturbance[..., 1]
        dsdt[..., 2] = self.gravity * control[..., 0] + disturbance[..., 2]
        dsdt[..., 3] = control[..., 1] - self.gravity + disturbance[..., 3]

        # No dynamics on the max disturbance changing 
        dsdt[..., 4] = 0 #dsdt[..., 4]
        dsdt[..., 5] = 0 #dsdt[..., 5]
        return dsdt
    
    def boundary_fn(self, state):
        return self.sdf(state)

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, time, dvds):
        optimal_control = self.optimal_control(state, dvds)
        optimal_disturbance = self.optimal_disturbance(state, dvds)
        flow = self.dsdt(state, optimal_control, optimal_disturbance, time)
        return torch.sum(flow*dvds, dim=-1)
    
    def optimal_control(self, state, dvds):
        if self.set_mode == "avoid":
            # a1 = torch.sign(dvds[..., 2]) * self.max_angle
            # a2 = self.min_thrust + torch.sign(dvds[..., 3]) * (self.max_thrust - self.min_thrust)
            a1 = torch.where(dvds[..., 2] < 0, -self.max_angle, self.max_angle)
            a2 = torch.where(dvds[..., 3] < 0, self.min_thrust, self.max_thrust)
        elif self.set_mode == "reach":
            # a1 = -torch.sign(dvds[..., 2]) * self.max_angle
            # a2 = self.max_thrust - torch.sign(dvds[..., 3]) * (self.max_thrust - self.min_thrust)
            a1 = torch.where(dvds[..., 2] > 0, -self.max_angle, self.max_angle)
            a2 = torch.where(dvds[..., 3] > 0, self.min_thrust, self.max_thrust)
        else:
            raise NotImplementedError("{self.set_mode} is not a valid set mode")
        return torch.cat((a1[..., None], a2[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        
        max_pos_dist = state[..., 4]
        max_vel_dist = state[..., 5]

        if self.set_mode == "avoid":
            # d1 = -torch.sign(dvds[..., 0]) * self.max_pos_dist
            # d2 = -torch.sign(dvds[..., 1]) * self.max_pos_dist
            # d3 = -torch.sign(dvds[..., 2]) * self.max_vel_dist
            # d4 = -torch.sign(dvds[..., 3]) * self.max_vel_dist
            d1 = torch.where(dvds[..., 0] > 0, -max_pos_dist, max_pos_dist)
            d2 = torch.where(dvds[..., 1] > 0, -max_pos_dist, max_pos_dist)
            d3 = torch.where(dvds[..., 2] > 0, -max_vel_dist, max_vel_dist)
            d4 = torch.where(dvds[..., 3] > 0, -max_vel_dist, max_vel_dist)
        elif self.set_mode == "reach":
            # d1 = torch.sign(dvds[..., 0]) * self.max_pos_dist
            # d2 = torch.sign(dvds[..., 1]) * self.max_pos_dist
            # d3 = torch.sign(dvds[..., 2]) * self.max_vel_dist
            # d4 = torch.sign(dvds[..., 3]) * self.max_vel_dist
            d1 = torch.where(dvds[..., 0] < 0, -max_pos_dist, max_pos_dist)
            d2 = torch.where(dvds[..., 1] < 0, -max_pos_dist, max_pos_dist)
            d3 = torch.where(dvds[..., 2] < 0, -max_vel_dist, max_vel_dist)
            d4 = torch.where(dvds[..., 3] < 0, -max_vel_dist, max_vel_dist)
        else:
            raise NotImplementedError("{self.set_mode} is not a valid set mode")
        return torch.cat((d1[..., None], d2[..., None], d3[..., None], d4[..., None]), dim=-1)
    
    def plot_config(self): # TODO: change this 
        return {
            'state_slices': [0, 0, 0, 0,   0, 0], # visualize at the worst case for now 
            'state_labels': ['y', 'z', r'$v_y$', r'$v_z$',   'max_pos_dist', 'max_vel_dist'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': [2, 3],
        }

class InvertedPendulum(Dynamics): 
    # Following dynamics from: https://arxiv.org/pdf/2206.03568 
    # NOTE: want to follow dynamics from: https://arxiv.org/pdf/1903.08792 

    def __init__(self, gravity: float, length: float, mass: float, 
                 unsafe_theta_min: float, unsafe_theta_max: float, min_torque: float, max_torque: float, 
                 max_theta_dist: float, max_thetadot_dist: float, 
                 damping: float=0.0, 
                 tMin: float=0.0, tMax: float=1.0):
        
        import numpy as np 

        self.gravity = gravity 
        self.tMin = tMin 
        self.tMax = tMax 

        # pendulum parameters
        self.gravity = gravity 
        self.length = length 
        self.mass = mass 

        # control and disturbance parameters 
        self.min_torque = min_torque 
        self.max_torque = max_torque 
        self.max_theta_dist = max_theta_dist 
        self.max_thetadot_dist = max_thetadot_dist

        self.unsafe_theta_min = unsafe_theta_min 
        self.unsafe_theta_max = unsafe_theta_max

        self.damping = damping 

        # Boundaries # NOTE: TODO: Add

        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid', 
            state_dim=2, input_dim=3, control_dim=1, disturbance_dim=2, 
            state_mean=[np.pi, 0], # NOTE: TODO: Check this - print a bunch of states and see what the case is ? 
            state_var=[np.pi, 1], # NOTE: TODO: check the angular velocity range 
            value_mean=0.2, # NOTE: TODO: check this ? - ask sander - check all the ones below ...
            value_var=0.5, 
            value_normto=0.02, 
            deepreach_model='vanilla', 
            periodic_dims=[0,1] 
        )
        return 

    def pendulum_sdf(self, state): 
        # TODO: might need to have a wrap around thing here 
        theta = state[..., 0] 
        
        unsafe_val = torch.zeros(theta.shape).to(state.device)
        unsafe_val[torch.where(theta > self.unsafe_theta_max)] = (theta - self.unsafe_theta_max)[torch.where(theta > self.unsafe_theta_max)]
        unsafe_val[torch.where(theta < self.unsafe_theta_min)] = (self.unsafe_theta_min - theta)[torch.where(theta < self.unsafe_theta_min)]
        
        remaining_dims = torch.where((self.unsafe_theta_max > theta) & (theta > self.unsafe_theta_min))
        unsafe_val[remaining_dims] = torch.min(self.unsafe_theta_min - theta, theta - self.unsafe_theta_max)[remaining_dims]

        return unsafe_val

    def state_test_range(self):
        raise NotImplementedError
    
    def equivalent_wrapped_state(self): 
        raise NotImplementedError
    
   
    # Dynamics 
    # d theta  = thetadot
    # d thetadot = (-damping*theta_dot - m*g*l*sin(theta) + dt)/ml^2 + 1/ml^2 u # NOTE: assumes gravity is positive
    def dsdt(self, state, control, disturbance, time):
        theta, thetadot = state
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = thetadot + disturbance[..., 0]
        dsdt[..., 1] = (-self.damping * thetadot + self.mass * self.gravity * self.length * torch.sin(theta) ) / (self.mass * self.length ** 2) + \
                        (control[..., 0]/ (self.mass * self.length**2)) + disturbance[..., 1]
        return dsdt 

    def boundary_fn(self, state): 
        return self.pendulum_sdf(state)
    
    def sample_target_state(self, num_samples): 
        raise NotImplementedError

    def cost_fn(self, state_traj): 
        raise NotImplementedError
    
    def hamiltonian(self, state, time, dvds):
        raise NotImplementedError
    
    def optimal_control(self, state, dvds): 
        raise NotImplementedError
    
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError
    
    def plot_config(self): 
        raise NotImplementedError


    def render(self, state, img_size=[500, 500]):
        """
        Creates an image of a pendulum at a given angle.

        Args:
        - theta (float): Angle of the pendulum (in radians) from the vertical.
        - length (float): Length of the pendulum (normalized for rendering).
        - img_size (int): Size of the square image in pixels.

        Returns:
        - img (numpy.ndarray): Image of the pendulum as a NumPy array with size (img_size, img_size, 3).
        """
        import matplotlib.pyplot as plt
        import numpy as np 
        import imageio
        from PIL import Image, ImageDraw, ImageFont

        theta, thetadot = state
        safe = self.boundary_fn(torch.tensor([[theta.item(), thetadot.item()]])) > 0 

        # Create a blank canvas
        fig, ax = plt.subplots(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
        ax.set_xlim(-self.length - 0.5, self.length + 0.5)
        ax.set_ylim(-self.length - 0.5, self.length + 0.5)
        ax.axis("off")

        # Pendulum coordinates: 0 means on top, np.pi means on bottom, pi/2 on right, -pi/2 on left
        x = self.length * torch.sin(theta)
        y = self.length * torch.cos(theta)  # positive because want flipped #-self.length * torch.cos(theta)  # Negative because y increases downwards

        # Draw pendulum
        ax.plot([0, x], [0, y], color="black", lw=2)  # Rod
        if safe: 
            ax.scatter(x, y, color="green", s=100)  # Pendulum bob
        else: 
            ax.scatter(x, y, color="red", s=500)  # Pendulum bob

        # Add state text near the pendulum bob
        state_text = f"θ = {theta:.2f} rad"
        ax.text(x + 0.1, y, state_text, fontsize=12, color="blue", ha="left", va="center")

        # Save the figure to a NumPy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        img = np.array(img)
        return img
    

class CartPole(Dynamics): 
    # Following dynamics from: https://arxiv.org/pdf/2206.03568 
    # NOTE: want to follow dynamics from: https://arxiv.org/pdf/1903.08792 

    def __init__(self, gravity: float, umax: float, length: float, mass_cart: float, mass_pole: float,
                 unsafe_x_min: float, unsafe_x_max: float, unsafe_vel_max: float, unsafe_theta_min: float, unsafe_theta_max: float, # unsafe bounds
                 x_dist: float, theta_dist: float, vel_dist: float, thetadot_dist: float, # disturbance bound parameters
                 tMin: float=0.0, tMax: float=1.0, unsafe_theta_in_range: float=True):
        """
        args: 
            - unsafe_theta_in_range: 
                - True:  when True the unsafe theta are described in the range
                - False: when False the range describes the safe theta - and out of range is the unsafe theta 
        """
        import numpy as np 
        self.unsafe_theta_in_range = unsafe_theta_in_range

        self.gravity = gravity 
        self.tMin = tMin 
        self.tMax = tMax 

        # cartpole parameters
        self.gravity = gravity 
        self.length = length 
        self.m_c = mass_cart
        self.m = mass_pole 

        # control and disturbance parameters 
        self.umax = umax
        
        self.x_dist = x_dist 
        self.theta_dist = theta_dist
        self.vel_dist = vel_dist 
        self.thetadot_dist = thetadot_dist 

        # Unsafe parameters 
        self.unsafe_x_min = unsafe_x_min
        self.unsafe_x_max = unsafe_x_max 
        self.unsafe_vel_max = unsafe_vel_max 
        self.unsafe_theta_min = unsafe_theta_min 
        self.unsafe_theta_max = unsafe_theta_max 

        self.use_unsafe_theta = True 
        if self.unsafe_theta_max == self.unsafe_theta_min: 
            # Do not use unsafe theta in the sdf
            self.use_unsafe_theta = False 

        # Boundaries # NOTE: TODO: Add

        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid', 
            state_dim=2, input_dim=3, control_dim=1, disturbance_dim=2, 
            state_mean=[np.pi, 0], # NOTE: TODO: Check this - print a bunch of states and see what the case is ? 
            state_var=[np.pi, 1], # NOTE: TODO: check the angular velocity range 
            value_mean=0.2, # NOTE: TODO: check this ? - ask sander - check all the ones below ...
            value_var=0.5, 
            value_normto=0.02, 
            deepreach_model='vanilla', 
            periodic_dims=[0,1] 
        )
        return 

    def state_test_range(self):
        raise NotImplementedError
    
    def equivalent_wrapped_state(self): 
        raise NotImplementedError
    
   
    # Dynamics: TODO: add dynamics equations here 
    def dsdt(self, state, control, disturbance, time):
        raise NotImplementedError

    def cartpole_sdf(self, state): 
        # TODO: Need to add this
        x = state[..., 0]
        theta = state[..., 1]
        xdot = state[..., 2]
        thetadot = state[..., 3]

        # Unsafe x: in range is safe 
        unsafe_x = torch.zeros(x.shape).to(state.device)
        greater_than_x = torch.where(x > self.unsafe_x_max)
        less_than_x = torch.where(x < self.unsafe_x_min)
        in_range_x = torch.where((self.unsafe_x_min < x) & (x < self.unsafe_x_max))
        unsafe_x[greater_than_x] = (self.unsafe_x_max - x)[greater_than_x] # negative unsafe 
        unsafe_x[less_than_x] = (x - self.unsafe_x_min)[less_than_x] # negative unsafe
        unsafe_x[in_range_x] = torch.min(x - self.unsafe_x_min, self.unsafe_x_max - x)[in_range_x]

        # Unsafe velocity: in range is safe 
        unsafe_xdot = torch.zeros(xdot.shape).to(state.device)
        greater_than_xdot = torch.where(xdot > self.unsafe_vel_max)
        less_than_xdot = torch.where(xdot < -self.unsafe_vel_max)
        in_range_xdot = torch.where((-self.unsafe_vel_max < xdot) & (xdot < self.unsafe_vel_max))
        unsafe_xdot[greater_than_xdot] = (self.unsafe_vel_max - xdot)[greater_than_xdot] # negative unsafe
        unsafe_xdot[less_than_xdot] = (xdot - (-1 * self.unsafe_vel_max))[less_than_xdot] # negative unsafe
        unsafe_xdot[in_range_xdot] = torch.min(xdot - (-1 * self.unsafe_vel_max), self.unsafe_vel_max - xdot)[in_range_xdot]

        if self.use_unsafe_theta: 

            unsafe_theta = torch.zeros(theta.shape).to(state.device)
            greater_than_theta = torch.where(theta > self.unsafe_theta_max) 
            less_than_theta = torch.where(theta < self.unsafe_theta_min)
            in_range_theta = torch.where((self.unsafe_theta_min < theta) & (theta < self.unsafe_theta_max))
            
            if self.unsafe_theta_in_range: 
                # Unsafe Theta: in range is unsafe
                unsafe_theta[greater_than_theta] = (theta - self.unsafe_theta_max)[greater_than_theta]
                unsafe_theta[less_than_theta] = (self.unsafe_theta_min - theta)[less_than_theta]
                unsafe_theta[in_range_theta] = torch.min(self.unsafe_theta_min - theta, theta - self.unsafe_theta_max)[in_range_theta] # negative unsafe
            else: 
                # Safe Theta: in range, Unsafe theta: out of range 
                unsafe_theta[greater_than_theta] = (self.unsafe_theta_max - theta)[greater_than_theta]
                unsafe_theta[less_than_theta] = (theta - self.unsafe_theta_min)[less_than_theta]
                unsafe_theta[in_range_theta] = torch.min(theta - self.unsafe_theta_min, self.unsafe_theta_max - theta)[in_range_theta]

            # TODO: NOTE: might need to change 
            unsafe_vals = torch.min(unsafe_x, torch.min(unsafe_xdot, unsafe_theta))
        else: 
            unsafe_vals = torch.min(unsafe_x, unsafe_xdot)
        
        # import pdb; pdb.set_trace()
        return unsafe_vals

    def boundary_fn(self, state): 
        return self.cartpole_sdf(state)
    
    def sample_target_state(self, num_samples): 
        raise NotImplementedError

    def cost_fn(self, state_traj): 
        raise NotImplementedError
    
    def hamiltonian(self, state, time, dvds):
        raise NotImplementedError
    
    def optimal_control(self, state, dvds): 
        raise NotImplementedError
    
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError
    
    def plot_config(self): 
        raise NotImplementedError

    def is_unsafe(self, state): 
        """
        Returns boolean if the cartpole is in the unsafe region
        """
        x = state[0]
        theta = state[1]
        xdot = state[2]

        if x < self.unsafe_x_min or self.unsafe_x_max < x: 
            return True 
        elif xdot < -self.unsafe_vel_max or self.unsafe_vel_max < xdot: 
            return True 
        elif self.use_unsafe_theta:
            if self.unsafe_theta_in_range: 
                return (self.unsafe_theta_min < theta and theta < self.unsafe_theta_max)
            else: 
                return (theta < self.unsafe_theta_min or self.unsafe_theta_max < theta)

        return False 

    def render(self, state):
        """
        Renders the cartpole environment and returns it as a NumPy array.

        Args:
            state (list or np.ndarray): The state of the cartpole [x, theta, x_dot, theta_dot].
        Returns:
            np.ndarray: The rendered image as a NumPy array.
        """
        pole_length = self.length 
        cart_width = self.length/2
        cart_height = self.length/4

        x, theta, _, _ = state
        y=0

        # Ensure theta is between -np.pi and np.pi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-2, 2)  # Set x-axis limits
        ax.set_ylim(-1.5, 1.5)  # Set y-axis limits
        ax.set_aspect('equal')
        ax.set_title(f"Cartpole: {(np.round(x.item(), decimals=3) , np.round(theta.item(), decimals=3))}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        if self.is_unsafe(state): 
            color = 'red'
        else: 
            color = 'blue'

        # Plot the patches 
        rect = plt.Rectangle((x-cart_width/2, y-cart_height/2), cart_width, cart_height, fill=True, color=color)
        line_start = (x, y) # x,y
        # line_end = (x - pole_length*np.sin(theta), y + pole_length*np.cos(theta)) # x,y
        line_end = (x + pole_length*np.sin(theta), y + pole_length*np.cos(theta)) # x,y
        line = plt.Line2D([line_start[0], line_end[0]], [line_start[1], line_end[1]], color=color)
        ax.add_line(line)
        ax.add_patch(rect)

        # Render the figure to a NumPy array
        fig.canvas.draw()  # Render the figure
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        buf.close()

        plt.close(fig)  # Close the figure to release memory
        return img_array


# NOTE: Same as CartPole but the avoid sdf (only important for rendering) can be specified after initialization 
# Unsafe things don't need to be specified when initalizing 
class BaseCartPole(Dynamics): 
    # Following dynamics from: https://arxiv.org/pdf/2206.03568 
    # NOTE: want to follow dynamics from: https://arxiv.org/pdf/1903.08792 

    def __init__(self, gravity: float, umax: float, length: float, mass_cart: float, mass_pole: float,
                 x_dist: float, theta_dist: float, vel_dist: float, thetadot_dist: float, # disturbance bound parameters
                 tMin: float=0.0, tMax: float=1.0):
        """
        args: 
            - unsafe_theta_in_range: 
                - True:  when True the unsafe theta are described in the range
                - False: when False the range describes the safe theta - and out of range is the unsafe theta 
        """
        import numpy as np 

        self.gravity = gravity 
        self.tMin = tMin 
        self.tMax = tMax 

        # cartpole parameters
        self.gravity = gravity 
        self.length = length 
        self.m_c = mass_cart
        self.m = mass_pole 

        # control and disturbance parameters 
        self.umax = umax
        
        self.x_dist = x_dist 
        self.theta_dist = theta_dist
        self.vel_dist = vel_dist 
        self.thetadot_dist = thetadot_dist 


        # Initialize boundary function - start with safe everywhere 
        self.boundary_function = lambda x: torch.ones(x[..., 0].shape).to(x.device)

        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid', 
            state_dim=2, input_dim=3, control_dim=1, disturbance_dim=2, 
            state_mean=[np.pi, 0], # NOTE: TODO: Check this - print a bunch of states and see what the case is ? 
            state_var=[np.pi, 1], # NOTE: TODO: check the angular velocity range 
            value_mean=0.2, # NOTE: TODO: check this ? - ask sander - check all the ones below ...
            value_var=0.5, 
            value_normto=0.02, 
            deepreach_model='vanilla', 
            periodic_dims=[0,1] 
        )
        return 

    def state_test_range(self):
        raise NotImplementedError
    
    def equivalent_wrapped_state(self): 
        raise NotImplementedError
    
   
    # Dynamics: TODO: add dynamics equations here 
    def dsdt(self, state, control, disturbance, time):
        raise NotImplementedError

    def init_boundary_fn(self, func):
        """
        Function to initialize boundary function for cartpole: 
        This is the function that will be used for the boundary function 
            - NOTE: right now honestly only used for rendering 
        args: 
            - func: function that takes in state and returns the avoid value 
        """
        self.boundary_function = func 
        return 
    
    def boundary_fn(self, state): 
        return self.boundary_function(state)
    
    def sample_target_state(self, num_samples): 
        raise NotImplementedError

    def cost_fn(self, state_traj): 
        raise NotImplementedError
    
    def hamiltonian(self, state, time, dvds):
        raise NotImplementedError
    
    def optimal_control(self, state, dvds): 
        raise NotImplementedError
    
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError
    
    def plot_config(self): 
        raise NotImplementedError

    def is_unsafe(self, state): 
        """
        Returns boolean if the cartpole is in the unsafe region
        """
        return self.boundary_fn(state) < 0 

    def render(self, state):
        """
        Renders the cartpole environment and returns it as a NumPy array.

        Args:
            state (list or np.ndarray): The state of the cartpole [x, theta, x_dot, theta_dot].
        Returns:
            np.ndarray: The rendered image as a NumPy array.
        """
        pole_length = self.length 
        cart_width = self.length/2
        cart_height = self.length/4

        x, theta, _, _ = state
        y=0

        # Ensure theta is between -np.pi and np.pi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-2, 2)  # Set x-axis limits
        ax.set_ylim(-1.5, 1.5)  # Set y-axis limits
        ax.set_aspect('equal')
        ax.set_title(f"Cartpole: {(np.round(x.item(), decimals=3) , np.round(theta.item(), decimals=3))}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        if self.is_unsafe(state): 
            color = 'red'
        else: 
            color = 'blue'

        # Plot the patches 
        rect = plt.Rectangle((x-cart_width/2, y-cart_height/2), cart_width, cart_height, fill=True, color=color)
        line_start = (x, y) # x,y
        # line_end = (x - pole_length*np.sin(theta), y + pole_length*np.cos(theta)) # x,y
        line_end = (x + pole_length*np.sin(theta), y + pole_length*np.cos(theta)) # x,y
        line = plt.Line2D([line_start[0], line_end[0]], [line_start[1], line_end[1]], color=color)
        ax.add_line(line)
        ax.add_patch(rect)

        # Render the figure to a NumPy array
        fig.canvas.draw()  # Render the figure
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        buf.close()

        plt.close(fig)  # Close the figure to release memory
        return img_array


if __name__ == "__main__":
    dynamics = Quad2DAttitude(9.81, 0.75, 5.0, 15.0, 0.0, 0.0, 'avoid')
    sample = torch.rand((65000, 4)) * 2 - 1
    sample_pts = dynamics.input_to_coord(sample)
    