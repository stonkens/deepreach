import jax.numpy as jnp
import jax
from torch2jax import t2j, j2t
import numpy as np
from hj_reachability.finite_differences import upwind_first


class GroundTruthHJSolution:
    def __init__(self, hj_dynamics):
        import hj_reachability as hj
        import jax.numpy as jnp
        self.hj_dynamics = hj_dynamics
        state_mean = jnp.array(self.hj_dynamics.torch_dynamics.state_mean.detach().cpu().numpy())
        state_var = jnp.array(self.hj_dynamics.torch_dynamics.state_var.detach().cpu().numpy())
        state_hi = state_mean + state_var
        state_lo = state_mean - state_var
        state_domain = hj.sets.Box(lo=state_lo, hi=state_hi)
        
        grid_resolution = tuple([51]) * self.hj_dynamics.torch_dynamics.state_dim 
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, grid_resolution)
        sdf_values = t2j(self.hj_dynamics.torch_dynamics.boundary_fn(j2t(self.grid.states)))
        backwards_reachable_tube = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))
        solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                          value_postprocessor=backwards_reachable_tube(sdf_values))
        min_time = self.hj_dynamics.tMin
        max_time = -self.hj_dynamics.tMax
        self.times = jnp.linspace(min_time, max_time, 5)  # FIXME: Hardcoded (has to be same  as with QuantifyBinary)
        self.value_functions = hj.solve(solver_settings, self.hj_dynamics, self.grid, self.times, 
                                        sdf_values, progress_bar=True)
        self.interpolation_f = jax.vmap(self.grid.interpolate, in_axes=(None, 0))
        self.dsdt_f = jax.vmap(self.hj_dynamics.__call__, in_axes=(0) * self.grid.ndim)
        self.optimal_control_and_disturbance_f = jax.vmap(self.hj_dynamics.optimal_control_and_disturbance, in_axes=(0, 0, 0))
        
    def __call__(self, state, time):
        # Find nearest time
        def single_compute(state, time):
            time_idx = jnp.argmin(jnp.abs(jnp.abs(self.times) - jnp.abs(time)))
            return self.grid.interpolate(self.value_functions[time_idx], state)
        print(time)
        vectorized_compute = jax.vmap(single_compute, in_axes=(0, 0))
        return vectorized_compute(state, time)
    
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
        value_gradients = self.get_values_gradient(states, ts)
        return value_gradients
    
    def value_from_coords(self, coordinates):
        # FIXME: Temp to convert to cuda, should be fixed further upstream
        ts, states = jnp.split(t2j(coordinates.to('cuda')), [1], axis=1)
        values = self.get_values(states, ts)
        return j2t(values)