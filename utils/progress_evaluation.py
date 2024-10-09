import numpy as np
import torch
import jax
from torch2jax import t2j, j2t
import jax.numpy as jnp
from utils.error_evaluators import SliceSampleGenerator, ValueThresholdValidator
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
        from utils import boundary_functions
        space_boundary = boundary_functions.BoundaryJAX([0, 1, 2, 3], jnp.array([-4.5, 0.0, -2.0, -2.0]), 
                                                    jnp.array([4.5, 2.5, 2.0, 2.0]))
        circle = boundary_functions.CircleJAX([0, 1], 0.5, jnp.array([2.0, 1.5]))
        rectangle = boundary_functions.RectangleJAX([0, 1], jnp.array([-2.0, 0.5]), jnp.array([-1.0, 1.5]))
        sdf_function = boundary_functions.build_sdf_jax(space_boundary, [circle, rectangle])
        # sdf_function = lambda x: jnp.linalg.norm(x[:2]) - self.hj_dynamics.goalR

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
        self.dsdt_f = jax.vmap(self.hj_dynamics.__call__, in_axes=(0) * self.grid.ndim)
        
    def __call__(self, state, time):
        # Find nearest time
        def single_compute(state, time):
            time_idx = jnp.argmin(jnp.abs(jnp.abs(self.times) - jnp.abs(time)))
            return self.grid.interpolate(self.value_functions[time_idx], state)
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
        if (states.reshape(*self.grid.shape, self.grid.ndim) == self.grid.states).all() and (times == self.times).all():
            return False
        else:
            return True


class CompareWithAlternative:
    def __init__(self, orig_dynamics, alt_method, comparsion_metrics, eval_states, eval_times, add_temporal_data=True):
        self.orig_dynamics = orig_dynamics
        self.alt_method = alt_method
        self.comparsion_metrics = comparsion_metrics
        self.eval_states = eval_states
        self.eval_times = eval_times
        self.include_plot = True
        self.add_temporal_data = add_temporal_data
    
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

    def __call__(self, model, **kwargs):
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
                'all_t_share_of_positive_states': share_positive_states.item(), 
                'all_t_jaccard_index': jaccard_index.item(), 
                'all_t_false_positive_states': false_positive_states.item(), 
                'all_t_false_negative_states': false_negative_states.item()
            }
            if self.add_temporal_data:
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
                        f't={timestep}_share_of_positive_states': share_positive_states_subset.item(),
                        f't={timestep}_jaccard_index': jaccard_index_subset.item(), 
                        f't={timestep}_false_positive_states': false_positive_states_subset.item(), 
                        f't={timestep}_false_negative_states': false_negative_states_subset.item()
                    })
        return log_dict


    def get_comparison_plot_data(self, coordinates):
        ts, states = jnp.split(t2j(coordinates), [1], axis=1)
        values = self.alt_method.get_values(states, ts)
        return j2t(values).to(coordinates.device)


class EmpiricalPerformance:
    def __init__(self, dynamics, dt, sample_generator = None, sample_validator = None, violation_validator = None,
                 batch_size=100000, samples_per_round_multiplier=5.0, fixed_vf=True, device='cuda'):
        self.dynamics = dynamics
        self.dt = dt  # dt in forward simulation
        self.sample_generator = sample_generator
        if self.sample_generator is None:
            # Default: Any state can be chosen as a sample
            self.sample_generator = SliceSampleGenerator(dynamics, [None]*dynamics.state_dim)
        self.sample_validator = sample_validator
        if self.sample_validator is None:
            # Default: Valid x0s all have positive values
            self.sample_validator = ValueThresholdValidator(v_min=0.0, v_max=float('inf'))
        self.violation_validator = violation_validator
        if self.violation_validator is None:
            # Default: Violation states are states where the value is negative
            self.violation_validator = ValueThresholdValidator(v_min=float('-inf'), v_max=0.0)
        self.batch_size = batch_size
        self.samples_per_round_multiplier = samples_per_round_multiplier
        self.fixed_vf = fixed_vf  # If fixed_vf is True, we keep the value function fixed (as if it converged)
        self.device = device

    def get_values(self, coordinates, model, **kwargs):
        with torch.no_grad():
            candidate_model_results = model({'coords': self.dynamics.coord_to_input(coordinates.to(self.device))})
            candidate_values = self.dynamics.io_to_value(candidate_model_results['model_in'], 
                                                         candidate_model_results['model_out'].squeeze(dim=-1))
        return candidate_values
    
    def get_optimal_trajectory(self, curr_coords, model, **kwargs):
        rollout_results = model({'coords': self.dynamics.coord_to_input(curr_coords)})
        dvs = self.dynamics.io_to_dv(rollout_results['model_in'], rollout_results['model_out'].squeeze(dim=-1)).detach()
        
        controls = self.dynamics.optimal_control(curr_coords[:, 1:], dvs[..., 1:])
        disturbances = self.dynamics.optimal_disturbance(curr_coords[:, 1:], dvs[..., 1:])
        next_states = (curr_coords[:, 1:] + self.dt * self.dynamics.dsdt(curr_coords[:, 1:], controls, disturbances))
        # you can save some RAM by deleting the model here (and running torch.cuda.empty_cache())
        return next_states, controls, disturbances
    
    def __call__(self, model, vf_times, rollout_times=None):
        """
            model: torch.nn.Module to use for finding valid sample points + optionally performing rollouts (if NN)
            vf_times: float or tuple of floats, timepoints from which we start evaluating the value function
                - If vf_fixed is True, this vf is used for all times in the rollout (considers converged vf)
                - If vf_fixed is False, this vf changes its t with the rollout itself (surfing the time-varying vf)
            rollout_times: float or tuple of floats, length of time for which we perform rollouts
                - Can only be set if vf_fixed is True
        """
        if isinstance(vf_times, float):
            # TODO: Maybe replace with explicit call in run_experiment to distinguish cases
            # Here, we only evaluate the model with rollouts from the maximum time
            times_hi = vf_times
            times_lo = vf_times
        else:
            # We take different times for which we start evaluating
            times_lo, times_hi = vf_times
        
        num_samples = int(self.samples_per_round_multiplier * self.batch_size)
        sample_times = torch.zeros(self.batch_size, )
        sample_states = torch.zeros(self.batch_size, self.dynamics.state_dim)
        sample_values = torch.zeros(self.batch_size, )

        num_scenarios = 0
        while num_scenarios < self.batch_size:
            candidate_sample_times = torch.ceil((torch.rand((num_samples)) * (times_hi - times_lo) + times_lo) / self.dt) * self.dt
            candidate_sample_states = self.dynamics.equivalent_wrapped_state(self.sample_generator.sample(num_samples))
            candidate_sample_coords = torch.cat((candidate_sample_times.unsqueeze(-1), candidate_sample_states), dim=-1)

            candidate_values = self.get_values(candidate_sample_coords, model)

            valid_candidate_idis = torch.where(self.sample_validator.validate(candidate_sample_coords, candidate_values))[0].detach().cpu()
            valid_candidate_idis = valid_candidate_idis[:self.batch_size - num_scenarios]  # Remove any excess
            num_valid_idis = len(valid_candidate_idis)
            sample_times[num_scenarios:num_scenarios + num_valid_idis] = candidate_sample_times[valid_candidate_idis]
            sample_states[num_scenarios:num_scenarios + num_valid_idis] = candidate_sample_states[valid_candidate_idis]
            sample_values[num_scenarios:num_scenarios + num_valid_idis] = candidate_values[valid_candidate_idis]
            num_scenarios += num_valid_idis

        if self.fixed_vf and (rollout_times is not None):
            if isinstance(rollout_times, float):
                traj_times_hi = rollout_times
                traj_times_lo = rollout_times
            else:
                traj_times_lo, traj_times_hi = rollout_times
            # sample the start times
            traj_sample_times = torch.ceil((torch.rand((self.batch_size)) * (traj_times_hi - traj_times_lo) + traj_times_lo) / self.dt) * self.dt
        else: 
            assert rollout_times is None, "Rollout times are not supported when the value function is not fixed"
            traj_times_lo = times_hi
            traj_times_hi = times_hi
            traj_sample_times = sample_times


        state_trajs = torch.zeros(self.batch_size, int((traj_times_hi) / self.dt + 1), self.dynamics.state_dim)
        controls_trajs = torch.zeros(self.batch_size, int((traj_times_hi) / self.dt), self.dynamics.control_dim)
        state_trajs[:, 0] = sample_states

        for k in range(int((traj_times_hi) / self.dt)):
            traj_time = traj_times_hi - k * self.dt
            if self.fixed_vf:
                model_input_time = sample_times.clone().unsqueeze(-1)
            else:
                model_input_time = torch.zeros(self.batch_size, 1).fill_(traj_time)

            curr_coords = torch.cat((model_input_time, state_trajs[:, k]), dim=-1).to(self.device)
            next_states, controls, dists = self.get_optimal_trajectory(curr_coords, model)
            
            not_started_times = traj_sample_times < (traj_time - self.dt / 2)
            started_times = ~not_started_times
            state_trajs[not_started_times, k + 1] = state_trajs[not_started_times, k]
            state_trajs[started_times, k + 1] = next_states[started_times].to('cpu')
            controls_trajs[started_times, k] = controls[started_times].to('cpu')
        batch_scenario_costs = self.dynamics.cost_fn(state_trajs)
        batch_value_errors = batch_scenario_costs - sample_values
        batch_value_mse = torch.mean(batch_value_errors ** 2)
        false_safe_trajectories = torch.logical_and(batch_scenario_costs < 0, sample_values >= 0)
        log_dict = {
            'value_mse': batch_value_mse.item(),
            'false_safe_trajectories': torch.sum(false_safe_trajectories).item() / self.batch_size,
            'trajectories': state_trajs,
            'values': sample_values,
            'controls_trajs': controls_trajs,
            'batch_values': batch_scenario_costs
        }
        return log_dict


class EmpiricalPerformanceHJR(EmpiricalPerformance):
    def __init__(self, dynamics, dt, sample_generator = None, sample_validator = None, violation_validator = None,
                 batch_size=100000, samples_per_round_multiplier=5.0, fixed_vf=True, **kwargs):
        self.hj_dynamics = dynamics
        dynamics = self.hj_dynamics.torch_dynamics
        super().__init__(dynamics, dt, sample_generator, sample_validator, violation_validator, batch_size, samples_per_round_multiplier, fixed_vf)
        self.ground_truth_hj_solution = kwargs['ground_truth_hj_solution']

    def get_values(self, coordinates, model=None, **kwargs):
        coordinates = t2j(coordinates)
        return j2t(self.ground_truth_hj_solution.get_values(coordinates[:, 1:], coordinates[:, 0]))

    def get_optimal_trajectory(self, curr_coords, model=None, **kwargs):
        curr_coords = t2j(curr_coords)
        # Get gradient values
        grad_values = self.ground_truth_hj_solution.get_values_gradient(curr_coords[:, 1:], curr_coords[:, 0])
        # Get optimal control and disturbance
        control, disturbance = self.hj_dynamics.optimal_control_and_disturbance(curr_coords[:, 1:], curr_coords[:, 0], grad_values)
        # Forward simulate
        next_states = curr_coords[:,1:] + self.dt * self.ground_truth_hj_solution.dsdt_f(curr_coords[:, 1:], control, disturbance, curr_coords[:, 0])
        # if self.hj_dynamics.periodic_dims is not None:
        #     raise NotImplementedError("Periodic dimensions not yet supported")
        return j2t(next_states), j2t(control), j2t(disturbance)
