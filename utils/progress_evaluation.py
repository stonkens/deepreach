import numpy as np
import torch
import jax
from torch2jax import t2j, j2t
import jax.numpy as jnp


class GroundTruthHJSolution:
    def __init__(self, hj_dynamics):
        import hj_reachability as hj
        import jax.numpy as jnp
        self.hj_dynamics = hj_dynamics
        state_test_range = jnp.array(self.hj_dynamics.torch_dynamics.state_bounds.detach().cpu().numpy())

        state_domain = hj.sets.Box(lo=state_test_range[:, 0], hi=state_test_range[:, 1])
        
        grid_resolution = tuple([51]) * state_test_range.shape[0]  
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, grid_resolution)
        from utils import boundary_functions
        space_boundary = boundary_functions.BoundaryJAX([0, 1, 2, 3], jnp.array([-4.5, 0.0, -3.0, -3.0]), 
                                                    jnp.array([4.5, 2.5, 3.0, 3.0]))
        circle = boundary_functions.CircleJAX([0, 1], 0.5, jnp.array([2.0, 1.5]))
        rectangle = boundary_functions.RectangleJAX([0, 1], jnp.array([-2.0, 0.5]), jnp.array([-1.0, 1.5]))
        sdf_function = boundary_functions.build_sdf_jax(space_boundary, [circle, rectangle])
        sdf_values = hj.utils.multivmap(sdf_function, jnp.arange(self.grid.ndim))(self.grid.states).squeeze()
        backwards_reachable_tube = lambda obstacle: (lambda t, x: jnp.minimum(x, obstacle))
        solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                          value_postprocessor=backwards_reachable_tube(sdf_values))
        min_time = 0.0
        max_time = -1.0
        self.times = jnp.linspace(min_time, max_time, 5)
        self.value_functions = hj.solve(solver_settings, self.hj_dynamics, self.grid, self.times, 
                                        sdf_values, progress_bar=True)
        self.interpolation_f = jax.vmap(self.grid.interpolate, in_axes=(None, 0))
        
    def __call__(self, state, time):
        # Find nearest time
        def single_compute(state, time):
            time_idx = jnp.argmin(jnp.abs(jnp.abs(self.times) - jnp.abs(time)))
            return self.grid.interpolate(self.value_functions[time_idx], state)
        vectorized_compute = jax.vmap(single_compute, in_axes=(0, 0))
        return vectorized_compute(state, time)
    
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
        if (states.reshape(*self.grid.shape, self.grid.ndim) == self.grid.states).all() and (times == self.times).all():
            return False
        else:
            return True


class CompareWithAlternative:
    def __init__(self, orig_dynamics, alt_method, comparsion_metrics, eval_states, eval_times):
        self.orig_dynamics = orig_dynamics
        self.alt_method = alt_method
        self.comparsion_metrics = comparsion_metrics
        self.eval_states = eval_states
        self.eval_times = eval_times
        self.include_plot = True
        # state_mesh, time_mesh = torch.meshgrid(eval_states, eval_times[torch.newaxis], indexing='ij')
        # self.eval_states_flat = state_mesh.reshape(-1, state_mesh.shape[-1])
        # self.eval_times_flat = time_mesh.reshape(-1, time_mesh.shape[-1])
        # self.eval_input = torch.cat((self.eval_times_flat, self.eval_states_flat), dim=1).to('cuda')
    
    def process_in_batches(self, model, ts, eval_states, batch_size):
        # Concatenate ts and eval_states
        full_input = torch.cat((ts, eval_states), dim=-1)
        
        # Split the full input into batches
        num_batches = (full_input.size(0) + batch_size - 1) // batch_size  # Calculate the number of batches
        outputs = []

        for i in range(num_batches):
            batch_input = full_input[i * batch_size:(i + 1) * batch_size]
            batch_output = model({'coords': self.orig_dynamics.coord_to_input(batch_input)})
            outputs.append(batch_output)

        # Initialize an empty dictionary to store concatenated results
        concatenated_output = {key: [] for key in outputs[0].keys()}

        # Concatenate the outputs from all batches for each key
        for output in outputs:
            for key, value in output.items():
                concatenated_output[key].append(value)

        # Convert lists to tensors
        for key in concatenated_output:
            concatenated_output[key] = torch.cat(concatenated_output[key], dim=0)

        return concatenated_output

    def __call__(self, model, add_temporal_data=False):
        if self.alt_method.recompute_values(self.eval_states, self.eval_times):
            alt_values = []
            for timestep in self.eval_times:
                alt_values.append(self.alt_method(self.eval_states, timestep))
            alt_values = torch.cat(alt_values, dim=1)
        else:
            alt_values = j2t(self.alt_method.get_values_table()[0])[..., torch.newaxis]

        with torch.no_grad():
            values = []
            for timestep in j2t(self.eval_times):
                eval_states = j2t(self.eval_states)
                ts = torch.zeros(self.eval_states.shape[0], 1).fill_(timestep).to(eval_states.device)
                output = self.process_in_batches(model, ts, eval_states, 62500)
                vals = self.orig_dynamics.io_to_value(output['model_in'], output['model_out'].squeeze(dim=-1))
                values.append(vals[torch.newaxis])
            values = torch.cat(values, dim=0)[..., torch.newaxis]
            alt_values = alt_values.to(values.device)
            positive_model_states = (values >= 0)
            share_positive_states = torch.sum(positive_model_states) / torch.numel(values)
            positive_alt_states = (alt_values >= 0)
            intersection_mask = torch.logical_and(positive_alt_states, positive_model_states) 
            union_mask = torch.logical_or(positive_alt_states, positive_model_states)
            jaccard_index = torch.sum(intersection_mask) / torch.sum(union_mask)
            false_positive_states = torch.logical_and(positive_model_states, ~positive_alt_states).sum() / torch.numel(values)
            false_negative_states = torch.logical_and(~positive_model_states, positive_alt_states).sum() / torch.numel(values)
            log_dict = {
                'share_of_positive_states': share_positive_states.item(), 
                'jaccard_index': jaccard_index.item(), 
                'false_positive_states': false_positive_states.item(), 
                'false_negative_states': false_negative_states.item()
            }
            if add_temporal_data:
                for i, timestep in enumerate(self.eval_times):
                    values_subset = values[i]
                    alt_values_subset = alt_values[i]
                    positive_model_states_subset = (values_subset >= 0)
                    positive_alt_states_subset = (alt_values_subset >= 0)
                    share_positive_states_subset = torch.sum(positive_model_states_subset) / torch.numel(values_subset)
                    intersection_mask_subset = torch.logical_and(positive_alt_states_subset, positive_model_states_subset)
                    union_mask_subset = torch.logical_or(positive_alt_states_subset, positive_model_states_subset)
                    jaccard_index_subset = torch.sum(intersection_mask_subset) / torch.sum(union_mask_subset)
                    false_positive_states_subset = torch.logical_and(positive_model_states_subset, ~positive_alt_states_subset).sum() / torch.numel(values_subset)
                    false_negative_states_subset = torch.logical_and(~positive_model_states_subset, positive_alt_states_subset).sum() / torch.numel(values_subset)
                    
                    log_dict.update({
                        f'share_of_positive_states_t={timestep}': share_positive_states_subset.item(),
                        f'jaccard_index_t={timestep}': jaccard_index_subset.item(), 
                        f'false_positive_states_t={timestep}': false_positive_states_subset.item(), 
                        f'false_negative_states_t={timestep}': false_negative_states_subset.item()
                    })
        return log_dict


    def get_comparison_plot_data(self, coordinates):
        ts, states = jnp.split(t2j(coordinates), [1], axis=1)
        values = self.alt_method.get_values(states, ts)
        return j2t(values).to(coordinates.device)

if __name__ == "__main__":
    import hj_reachability as hj
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box([0., 0.], [1., 1.]), (3, 3))
    alt_method = None
    times = torch.from_numpy(np.linspace(0, 1, 5).astype(np.float32))
    eval_states = torch.from_numpy(np.array(grid.states))
    comparison = CompareWithAlternative(alt_method, [], eval_states, times)
