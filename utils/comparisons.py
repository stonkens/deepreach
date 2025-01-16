import jax.numpy as jnp
import jax
from torch2jax import t2j, j2t
import numpy as np
from hj_reachability.finite_differences import upwind_first


class GroundTruthHJSolution:
    """
    Ground truth solution using Dynamic Programming, only for 5 state dimensions or less.
    """
    def __init__(self, hj_dynamics):
        import hj_reachability as hj
        import jax.numpy as jnp
        self.hj_dynamics = hj_dynamics
        self.is_parametric = False 
        if hasattr(self.hj_dynamics.torch_dynamics, "parametric_dims"):
            self.is_parametric = True 
            self.non_parametric_state_dims = self.hj_dynamics.torch_dynamics.state_dims

        state_mean = self.hj_dynamics.torch_dynamics.state_mean.detach().cpu().numpy()
        state_var = self.hj_dynamics.torch_dynamics.state_var.detach().cpu().numpy().copy()
        for periodic_dim in self.hj_dynamics.periodic_dims:
            state_var[periodic_dim] = np.pi # Deepreach dynamics might add overlap, ground truth should have pi
        
        if self.is_parametric:
            state_mean = state_mean[self.non_parametric_state_dims]
            state_var = state_var[self.non_parametric_state_dims]
            grid_resolution = tuple([51]) * len(self.non_parametric_state_dims)
        else: 
            grid_resolution = tuple([51]) * self.hj_dynamics.torch_dynamics.state_dim 
        
        state_mean = jnp.array(state_mean) 
        state_var = jnp.array(state_var)
        
        state_hi = state_mean + state_var
        state_lo = state_mean - state_var
        state_domain = hj.sets.Box(lo=state_lo, hi=state_hi)
        
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, grid_resolution, periodic_dims=self.hj_dynamics.periodic_dims)
        
        self.loss_type = self.hj_dynamics.torch_dynamics.loss_type
        self.set_mode = self.hj_dynamics.torch_dynamics.set_mode
        if self.loss_type == 'brt_hjivi':
            # Distinguish between reach and avoid
            self.avoid_values = t2j(self.hj_dynamics.torch_dynamics.boundary_fn(j2t(self.grid.states)))
            brt = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))
            postprocessor = brt(self.avoid_values)
            self.boundary_values = self.avoid_values
        elif self.loss_type == 'brat_ci_hjivi':
            self.avoid_values = t2j(self.hj_dynamics.torch_dynamics.avoid_fn(j2t(self.grid.states)))
            self.reach_values = t2j(self.hj_dynamics.torch_dynamics.reach_fn(j2t(self.grid.states)))
            self.boundary_values = t2j(self.hj_dynamics.torch_dynamics.boundary_fn(j2t(self.grid.states)))
            brt = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))
            postprocessor = brt(self.avoid_values)
        elif self.loss_type == 'brat_hjivi':
            # By convention, we always "max" u and "min" d in hj_reachability, hence some flipping required
            # avoid_fn is defined such that avoid_fn >= 0 <=> in non-avoid region (also in HJR)
            # reach_fn is defined such that reach_fn <= 0 <=> in reach region (flipped in HJR)
            self.avoid_values = t2j(self.hj_dynamics.torch_dynamics.avoid_fn(j2t(self.grid.states)))
            self.reach_values = -t2j(self.hj_dynamics.torch_dynamics.reach_fn(j2t(self.grid.states)))
            self.boundary_values = -t2j(self.hj_dynamics.torch_dynamics.boundary_fn(j2t(self.grid.states)))
            brat = lambda obstacle, target: (lambda t, x: jnp.minimum(jnp.maximum(x, target), obstacle))
            postprocessor = brat(self.avoid_values, self.reach_values)
        

        solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                          value_postprocessor=postprocessor)
        min_time = self.hj_dynamics.tMin
        max_time = -self.hj_dynamics.tMax
        self.times = jnp.linspace(min_time, max_time, 5)  # FIXME: Hardcoded (has to be same  as with QuantifyBinary)
        self.value_functions = hj.solve(solver_settings, self.hj_dynamics, self.grid, self.times, 
                                        self.boundary_values, progress_bar=True)
        # TODO: Maybe negative for brat_hjivi
        self.interpolation_f = jax.vmap(self.grid.interpolate, in_axes=(None, 0))
        self.dsdt_f = jax.vmap(self.hj_dynamics.__call__, in_axes=(0, 0, 0, 0))
        self.optimal_control_and_disturbance_f = jax.vmap(self.hj_dynamics.optimal_control_and_disturbance, in_axes=(0, 0, 0))
        
    def __call__(self, state, time):
        raise NotImplementedError("Broken")
        # Find nearest time
        # def single_compute(state, time):
        #     time_idx = jnp.argmin(jnp.abs(jnp.abs(self.times) - jnp.abs(time)))
        #     return self.grid.interpolate(self.value_functions[time_idx], state)
        # vectorized_compute = jax.vmap(single_compute, in_axes=(0, 0))
        # return vectorized_compute(state, time)
    
    def get_values_gradient(self, states, ts):
        unique_times = jnp.unique(ts)
        alt_values = jnp.zeros((states.shape[0], states.shape[1]))
        if states.shape[0] == 1:
            time_idx = self.get_closest_time_idx(ts)
            grad_values = self.grid.grad_values(self.value_functions[time_idx], upwind_scheme=upwind_first.WENO3)
            alt_values = self.interpolation_f(grad_values, states)
        else:
            for timestep in unique_times:
                mask = (ts == timestep).squeeze()
                time_idx = self.get_closest_time_idx(timestep)
                grad_values = self.grid.grad_values(self.value_functions[time_idx], upwind_scheme=upwind_first.WENO3)
                alt_values = alt_values.at[mask].set(self.interpolation_f(grad_values, states[mask]))
        return alt_values
    
    def get_values(self, states, ts):
        unique_times = jnp.unique(ts)
        alt_values = jnp.zeros((states.shape[0]))
        for timestep in unique_times:
            mask = (ts == timestep).squeeze()
            time_idx = self.get_closest_time_idx(timestep)
            alt_values = alt_values.at[mask].set(self.interpolation_f(self.value_functions[time_idx], states[mask]))
        return alt_values

    def get_closest_time(self, time):
        return self.times(self.get_closest_time_idx(time))
    
    def get_closest_time_idx(self, time):
        return jnp.argmin(jnp.abs(jnp.abs(self.times) - jnp.abs(time)))
    
    def get_values_table(self):
        return self.value_functions.reshape(len(self.times), -1), self.times, self.grid.states.reshape(-1, self.grid.ndim)

    def recompute_values(self, states, times):
        import torch
        if isinstance(states, torch.Tensor):
            states = t2j(states)
            times = t2j(times)
        if np.isclose(states.reshape(*self.grid.shape, self.grid.ndim), self.grid.states, atol=1e-6).all():
            return False
        else:
            return True
    
    def value_gradient_from_coords(self, coordinates):
        ts, states = jnp.split(t2j(coordinates), [1], axis=1)
        if self.is_parametric: 
            states = states[..., self.non_parametric_state_dims]

        value_gradients = self.get_values_gradient(states, ts)
        return value_gradients
    
    def value_from_coords(self, coordinates):
        # FIXME: Temp to convert to cuda, should be fixed further upstream
        ts, states = jnp.split(t2j(coordinates.to('cuda')), [1], axis=1)
        if self.is_parametric: 
            states = states[..., self.non_parametric_state_dims]

        values = self.get_values(states, ts)
        return j2t(values)