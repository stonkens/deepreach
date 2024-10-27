import numpy as np
import torch
from torch2jax import t2j, j2t
from utils.error_evaluators import SliceSampleGenerator, ValueThresholdValidator
import math
import wandb
from abc import ABC, abstractmethod


class EvaluationMetric(ABC):
    """
    Abstract base class for evaluation metrics. 
    Subclasses must implement the __init__ and __call__ methods.
    """
    @abstractmethod
    def __init__(self, dataset, val_dict):
        """
        Initialize the metric with a dataset and a dictionary of validation parameters.
        """
        pass

    def update_counters(self):
        """
        Called before __call__ to update any counters or state variables.
        """
        pass

    @abstractmethod
    def __call__(self, model_eval, model_eval_grad):
        """
        Evaluate the model with the given model evaluation function and gradient.
        """
        pass



class VisualizeSafeSet2D(EvaluationMetric):
    """
    Purpose: Visualize the safe set of the current model in 2D for different slices of time and other states.
    How to adjust the visualization:
    - Modify val_dict to change the resolution of the grid, the time slices, and the state slices.
    - Modify state_test_range and plot_config in the dynamics function to change the range and state slices.
    """
    def __init__(self, dataset, val_dict):
        self.dataset = dataset
        self.val_dict = val_dict
        self.save_path = val_dict.get('save_path', None)
    
    def __call__(self, model_eval, model_eval_grad, vis_type='imshow'):
        """
        Generate and visualize the safe set for the model using 2D plots. 
        self.dataset.dynamics.plot_config() is used to determine the axes to plot.
        
        Args:
            model_eval: Function to evaluate the model on given coordinates.
            model_eval_grad: Gradient of the model evaluation function (unused here).
            vis_type: Visualization type ('imshow' or 'contourf'). imshow is a binary plot, contourf is a continuous plot.
        
        Steps:
        1) Find the coords corresponding to visualization points.
        2) Create figure with subplots for each time slice and state slice.
        3) In a for loop (over each individual subplot): Evaluate model at coords, plot the values, and plot boundary.
        """
        ########### Set up evaluation coords ###########
        import matplotlib.pyplot as plt
        import matplotlib
        plot_config = self.dataset.dynamics.plot_config()
        x_resolution = self.val_dict['x_resolution']
        y_resolution = self.val_dict['y_resolution']
        time_resolution = self.val_dict['time_resolution']
        z_resolution = self.val_dict['z_resolution']

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        if isinstance(plot_config['z_axis_idx'], list):
            z_min, z_max = list(map(list, zip(*[state_test_range[z_idx] for z_idx in plot_config['z_axis_idx']])))
            for plot_idx, z_idx in enumerate(plot_config['z_axis_idx']):
                if (hasattr(plot_config, 'angle_dims') and 
                    z_idx in plot_config['angle_dims'] and
                    math.isclose(z_max[plot_idx] - z_min[plot_idx], 2.*math.pi, rel_tol=1e-2)):
                    z_max[plot_idx] = z_max[plot_idx] - (z_max[plot_idx] - z_min[plot_idx]) / (z_resolution + 1) 
                else:
                    z_min[plot_idx], z_max[plot_idx] = state_test_range[z_idx]
        else:
            z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        if isinstance(plot_config['z_axis_idx'], list):
            zs = [torch.linspace(z_min[i], z_max[i], z_resolution) for i in range(len(plot_config['z_axis_idx']))]
            zs = torch.cartesian_prod(*zs)
        else:
            zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        ########### Setup visualization ###########
        fig = plt.figure(figsize=(6*len(zs), 5*len(times)))
        gs = matplotlib.gridspec.GridSpec(len(times), len(zs) + 1, width_ratios=[1] * len(zs) + [0.1], wspace=0.2, hspace=0.2)

        ########### Computing and plotting values ###########
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                if isinstance(plot_config['z_axis_idx'], list):
                    for k, z_idx in enumerate(plot_config['z_axis_idx']):
                        coords[:, 1 + z_idx] = zs[j][k]
                else:
                    coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    values = model_eval(coords)
                    sdf_values = self.dataset.dynamics.boundary_fn(coords[:, 1:].to(values.device))

                ax = fig.add_subplot(gs[i, j])

                if isinstance(plot_config['z_axis_idx'], list):
                    ax_title = 't = %0.2f, %s' % (
                        times[i],
                        ', '.join(['%s = %0.2f' % (plot_config['state_labels'][z_idx], zs[j][k].item()) 
                                   for k, z_idx in enumerate(plot_config['z_axis_idx'])])
                    )
                else:
                    ax_title = 't = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j])
                xs_plot = np.linspace(-1, 1, x_resolution)
                ys_plot = np.linspace(-1, 1, y_resolution)
                if vis_type == "imshow":
                    s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                    # Go from xs to (-1, 1) and ys to (-1, 1)
                elif vis_type == "contourf":
                    s = ax.contourf(xs_plot, ys_plot, values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T)
                    
                ax.contour(xs_plot, ys_plot, sdf_values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T, levels=[0], colors='black')
                ax.set_title(ax_title)

            cax = fig.add_subplot(gs[i, -1])
            fig.colorbar(s, cax=cax, orientation='vertical')
        fig.tight_layout()
        if self.save_path is not None:
            if self.save_path.endswith('.png'):
                fig.savefig(self.save_path)
            else:
                fig.savefig(self.save_path + '/model_safe_set.png')
        
        return {"safe_set": wandb.Image(fig)}


class VisualizeSafeSet2DHJR(VisualizeSafeSet2D):
    """
    Purpose: HJ implementation of VisualizeSafeSet2D
    """
    def __init__(self, dataset, val_dict, ground_truth):
        self.ground_truth = ground_truth
        super().__init__(dataset, val_dict)
    
    def __call__(self, model_eval, model_eval_grad):
        """
        Generate and visualize the safe set of the ground truth model (see VisualizeSafeSet2D for details).
        Args:
            model_eval: Function to evaluate the model on given coordinates.
            model_eval_grad: Gradient of the model evaluation function (unused here). 
        """
        model_eval = lambda x: self.ground_truth.value_from_coords(x)
        log_dict = super().__call__(model_eval, model_eval_grad)
        # Modify the keys to include _gt in the key to distinguish from the model
        new_dict = {}
        for key, value in log_dict.items():
            new_dict[key + "_gt"] = value
        return new_dict

class VisualizeValueDifference2D(VisualizeSafeSet2D):
    """
    Purpose: Visualize the difference between the model and the ground truth in 2D for different slices of time and 
             other states.
    See VisualizeSafeSet2D for details on how to adjust the visualization.
    """
    def __init__(self, dataset, val_dict, ground_truth):
        self.ground_truth = ground_truth
        super().__init__(dataset, val_dict)

    def __call__(self, model_eval, model_eval_grad):
        """
        Args:
            model_eval: Function to evaluate the model on given coordinates.
            model_eval_grad: Gradient of the model evaluation function (unused here). 
        """
        new_eval = lambda x: model_eval(x) - self.ground_truth.value_from_coords(x)
        log_dict = super().__call__(new_eval, model_eval_grad, vis_type='contourf')
        new_dict = {}
        for key, value in log_dict.items():
            new_dict[key + "_diff"] = value
        return new_dict

class QuantifyBinarySafety(EvaluationMetric):
    """
    Purpose: Quantifiable metrics on (safe v unsafe) states for the model. 
    How to adjust the visualization:
    - Modify val_dict to change the resolution of the grid and the time slices
        NOTE: Expects different grid and resolution compared to VisualizeSafeSet2D (this should be over all states)
    """
    def __init__(self, dataset, val_dict):
        """
        Same eval_states and eval_times for all evaluations.
        Args:
            dataset: Dataset object
            val_dict: Dictionary of evaluation parameters
        add_temporal_data: Whether to add temporal data (at different value function slices) to the log_dict
        """
        self.dataset = dataset
        time_resolution = val_dict.get('time_resolution', 5)
        self.eval_times = torch.linspace(self.dataset.tMin, self.dataset.tMax, time_resolution).to('cuda')
        grid_resolution = val_dict.get('grid_resolution', 51)
        eval_states = torch.cartesian_prod(*[torch.linspace(-1, 1, grid_resolution) 
                                             for _ in range(dataset.dynamics.state_dim)])
        # FIXME: Does this need to be reshaped?
        self.eval_states = self.dataset.dynamics.state_mean + eval_states * self.dataset.dynamics.state_var
        self.add_temporal_data = val_dict.get('add_temporal_data', True)

    def __call__(self, model_eval, model_eval_grad):
        """
        Evaluate the model on the eval_states and eval_times. These grids are often dense and require batching to not
        exceed GPU memory.
        Args:
            model_eval: Function to evaluate the model on given coordinates.
            model_eval_grad: Gradient of the model evaluation function (unused here). 
        """
        with torch.no_grad():
            values = []
            for timestep in self.eval_times.to(self.eval_states.device):
                coords = torch.cat((torch.zeros(self.eval_states.shape[0], 1).fill_(timestep).to(self.eval_states.device), self.eval_states), dim=-1)
                # Split into different batch sizes
                vals = []
                for i in range(0, coords.shape[0], 62500):  # FIXME: Hardcoded batch size
                    vals.append(model_eval(coords[i:i+62500]))
                vals = torch.cat(vals, dim=0)
                values.append(vals[torch.newaxis])
            values = torch.cat(values, dim=0)
            nbr_states = values.numel()
            positive_model_states = (values >= 0)
            share_positive_states = torch.sum(positive_model_states) / nbr_states
            log_dict = {'share_positive_states': share_positive_states.item()}
            if self.add_temporal_data:
                for i, ts in enumerate(self.eval_times):
                    nbr_states = values[i].numel()
                    share_positive_states = torch.sum(positive_model_states[i]) / nbr_states
                    log_dict.update({f'share_positive_states_t={ts}': share_positive_states.item()})
        return log_dict


class QuantifyBinarySafetyDifference(QuantifyBinarySafety):
    """
    Purpose: Quantifiable metrics on (safe v unsafe) states for the model with a comparison to ground truth.
    Includes IOU (Jaccard index) and false positive and false negative rates.
    See QuantifyBinarySafety for details on how to adjust the visualization.
    FIXME: If the grid is the same, we require the same time_resolution for the ground truth and the validation points.
    """
    def __init__(self, dataset, val_dict, ground_truth=None):
        super().__init__(dataset, val_dict)
        self.ground_truth = ground_truth
        if self.ground_truth.recompute_values(self.eval_states, self.eval_times):
            gt_values = []
            for timestep in self.eval_times:
                gt_values.append(self.ground_truth(self.eval_states, timestep))
            gt_values = torch.cat(gt_values, dim=1)
        else:
            gt_values = j2t(self.ground_truth.get_values_table()[0])
        self.gt_values = gt_values


    def __call__(self, model_eval, model_eval_grad):
        """
        Evaluate the model on the eval_states and eval_times. These grids are often dense and require batching to not
        exceed GPU memory.
        Args:
            model_eval: Function to evaluate the model on given coordinates
            model_eval_grad: Gradient of the model evaluation function (unused here).

        Unlike Visualization functions, this function does not modify model_eval and instead stores ground truth values.
        """
        with torch.no_grad():
            values = []
            for timestep in self.eval_times.to(self.eval_states.device):
                coords = torch.cat((torch.zeros(self.eval_states.shape[0], 1).fill_(timestep).to(self.eval_states.device), self.eval_states), dim=-1)
                vals = []
                for i in range(0, coords.shape[0], 62500):
                    vals.append(model_eval(coords[i:i+62500]))
                vals = torch.cat(vals, dim=0)
                values.append(vals[torch.newaxis])
            values = torch.cat(values, dim=0)

            nbr_states = values.numel()
            positive_model_states = (values >= 0)
            gt_values = self.gt_values.to(values.device)
            share_positive_states = torch.sum(positive_model_states) / nbr_states
            positive_gt_states = (gt_values >= 0)
            intersection_mask = torch.logical_and(positive_gt_states, positive_model_states)
            union_mask = torch.logical_or(positive_gt_states, positive_model_states)
            jaccard_index = torch.sum(intersection_mask) / torch.sum(union_mask)
            false_positive_states = torch.logical_and(positive_model_states, ~positive_gt_states)
            false_negative_states = torch.logical_and(~positive_model_states, positive_gt_states)
            share_false_positive_states = false_positive_states.sum() / nbr_states
            share_false_negative_states = false_negative_states.sum() / nbr_states
            log_dict = {
                'value_v_gt_mse': torch.mean((values - gt_values) ** 2).item(),
                'share_positive_states': share_positive_states.item(),
                'jaccard_index': jaccard_index.item(),
                'share_false_positive_states': share_false_positive_states.item(),
                'share_false_negative_states': share_false_negative_states.item()
            }
            if self.add_temporal_data:
                for i, ts in enumerate(self.eval_times):
                    nbr_states = values[i].numel()
                    share_positive_states = torch.sum(positive_model_states[i]) / nbr_states
                    jaccard_index = torch.sum(intersection_mask[i]) / torch.sum(union_mask[i])
                    share_false_positive_states = false_positive_states[i].sum() / nbr_states
                    share_false_negative_states = false_negative_states[i].sum() / nbr_states
                    log_dict.update({
                        f'value_v_gt_mse_t={ts}': torch.mean((values[i] - gt_values[i]) ** 2).item(),
                        f'share_positive_states_t={ts}': share_positive_states.item(),
                        f'jaccard_index_t={ts}': jaccard_index.item(),
                        f'share_false_positive_states_t={ts}': share_false_positive_states.item(),
                        f'share_false_negative_states_t={ts}': share_false_negative_states.item()
                    })
        return log_dict
 

class RolloutTrajectories(EvaluationMetric):
    """
    Purpose: Evaluate the model with rollouts from the value function.
    Specifically we can do the following rollouts:
    - Time varying rollouts: "Surf" the value function for t seconds (rollout is also t seconds)
    - Time invariant rollouts: Consider a converged value function (no time variation) and rollout for t seconds
    - Fixed samples: Use fixed samples for the rollout (e.g. to provide consistent comparison in training validation)

    How to adjust the metrics:
    - Modify val_dict to change the dt, rollout_batch_size, samples_per_round_multiplier, and the sample_generator.
    """
    def __init__(self, dataset, val_dict, is_time_invariant=True):
        self.dataset = dataset
        self.dynamics = dataset.dynamics
        self.val_dict = val_dict
        self.dt = val_dict['dt']
        self.sample_generator = val_dict.get('sample_generator', SliceSampleGenerator(self.dynamics, 
                                                                                      [None]*self.dynamics.state_dim))
        self.sample_validator = val_dict.get('sample_validator', ValueThresholdValidator(v_min=float(0.0), 
                                                                                         v_max=float('inf')))
        self.violation_validator = val_dict.get('violation_validator', ValueThresholdValidator(v_min=float('-inf'),
                                                                                               v_max=float(0.0)))
        self.batch_size = val_dict.get('rollout_batch_size', 20000)
        self.samples_per_round_multiplier = val_dict.get('samples_per_round_multiplier', 5.0)
        
        self.is_time_invariant = is_time_invariant  # Step through time or vary
        self.device = 'cuda'  # FIXME: Hardcoded for now   

    def generate_samples(self, model_eval, times_lo, times_hi, fixed_samples_validator=None):
        """
        Generate samples for the starting/initial state and times (jointly coords) for the rollout.
        Args:
            model_eval: Function to evaluate the model on given coordinates.
            times_lo: Lower bound of the time interval
            times_hi: Upper bound of the time interval
            fixed_samples_validator: Validator for fixed samples (if applicable), instead of using model_eval!
                - Used for e.g. comparing the model to the ground truth
        """
        num_samples = int(self.samples_per_round_multiplier * self.batch_size)
        sample_times = torch.zeros(self.batch_size, )
        sample_states = torch.zeros(self.batch_size, self.dynamics.state_dim)

        num_scenarios = 0
        max_counter = self.batch_size * 50
        counter = 0 
        while num_scenarios < self.batch_size:
            candidate_sample_times = (torch.ceil((torch.rand((num_samples)) * (times_hi - times_lo) + times_lo) / self.dt) * self.dt).to(self.device)
            candidate_sample_states = self.dynamics.equivalent_wrapped_state(self.sample_generator.sample(num_samples)).to(self.device)
            candidate_sample_coords = torch.cat((candidate_sample_times.unsqueeze(-1), candidate_sample_states), dim=-1).to(self.device)

            if fixed_samples_validator is not None:
                valid_candidate_idis = torch.where(fixed_samples_validator.validate(candidate_sample_coords))[0].detach().cpu()
            else:
                candidate_values = model_eval(candidate_sample_coords)
                valid_candidate_idis = torch.where(self.sample_validator.validate(candidate_sample_coords, candidate_values))[0].detach().cpu()
            valid_candidate_idis = valid_candidate_idis[:self.batch_size - num_scenarios]  # Remove any excess
            num_valid_idis = len(valid_candidate_idis)
            sample_times[num_scenarios:num_scenarios + num_valid_idis] = candidate_sample_times[valid_candidate_idis]
            sample_states[num_scenarios:num_scenarios + num_valid_idis] = candidate_sample_states[valid_candidate_idis]
            num_scenarios += num_valid_idis
            counter += 1
        
            if counter > max_counter: 
                print(f"Could not find enough valid samples after {max_counter} iterations")
                break 

        return sample_times, sample_states

    def get_optimal_trajectory(self, curr_coords, model_eval_grad):
        """
        Implement optimal trajectory for the model over one time step.
        u and d are the optimal control and disturbance associated with the Hamiltonian.
        u = argmax_u model_eval_grad(x) * f(x, u, d)
        d = argmin_d model_eval_grad(x) * f(x, u, d)
        x_{t+1} = x_t + dt * f(x, u, d)
        """
        dvs = model_eval_grad(curr_coords)
        controls = self.dynamics.optimal_control(curr_coords[:, 1:], dvs[..., 1:])
        disturbances = self.dynamics.optimal_disturbance(curr_coords[:, 1:], dvs[..., 1:])
        next_states = (curr_coords[:, 1:] + self.dt * self.dynamics.dsdt(curr_coords[:, 1:], controls, disturbances))
        return next_states, controls, disturbances
        
    def get_coords(self, model_eval, time_interval):
        """
        Generate sample coordinates (initial time) given the model (to validate states) and the time interval.
        This function is modified in subclasses to provide different types of rollouts (for the state specifically).        
        """
        sample_times, sample_states = self.generate_samples(model_eval, time_interval[0], time_interval[1])
        return sample_times, sample_states

    def update_counters(self):
        """
        Update base class functionality. Here we update the vf_times (time at which to evaluate the value function) 
        and the rollout times (time for which to rollout the trajectory).
        """
        curr_t = self.dataset.tMin + (self.dataset.tMax - self.dataset.tMin) * self.dataset.counter / self.dataset.counter_end
        self.vf_times = [max(0, curr_t - 0.02), curr_t]
        if self.is_time_invariant:
            self.rollout_times = self.dataset.tMax
        else:
            self.rollout_times = self.vf_times

    def __call__(self, model_eval, model_eval_grad):
        """
        Evaluate the model with rollouts with controls and disturbances from the value function and its gradient.
        Args:
            model_eval: Function to evaluate the model on given coordinates.
            model_eval_grad: Gradient of the model evaluation function (actually used here).
        Steps:
        1. Generate initial states and times (to eval the value function) for the rollout.
        2. Determine the duration of the rollout (distinguishing time invariant from time varying).
        3. Rollout trajectories
        4. Evaluate performance of the trajectories
        """
        ########### Step 1: Generate initial states and times ###########
        if isinstance(self.vf_times, float):
            times_hi = self.vf_times
            times_lo = self.vf_times
        else:
            # Randomize the start time (self.vf_times is a tuple)
            times_lo, times_hi = self.vf_times

        # Generate samples for the rollout
        sample_times, sample_states = self.get_coords(model_eval, (times_lo, times_hi))

        ########### Step 2: Determine rollout duration ###########
        if self.is_time_invariant:
            if isinstance(self.rollout_times, float):
                traj_times_hi = self.rollout_times
                traj_times_lo = self.rollout_times
            else:
                traj_times_lo, traj_times_hi = self.rollout_times
            # sample the start times
            traj_sample_times = torch.ceil((torch.rand((self.batch_size)) * (traj_times_hi - traj_times_lo) + 
                                            traj_times_lo) / self.dt) * self.dt
        else: 
            assert (self.rollout_times == self.vf_times), "Rollout for finite time has to match the vf times"
            traj_times_lo = times_hi
            traj_times_hi = times_hi
            traj_sample_times = sample_times

        ########### Step 3: Rollout the trajectories ###########
        state_trajs = torch.zeros(self.batch_size, int((traj_times_hi) / self.dt + 1), self.dynamics.state_dim)
        controls_trajs = torch.zeros(self.batch_size, int((traj_times_hi) / self.dt), self.dynamics.control_dim)
        cost_over_trajs = torch.zeros(self.batch_size, int((traj_times_hi) / self.dt + 1))
        values_over_trajs = torch.zeros(self.batch_size, int((traj_times_hi) / self.dt + 1))
        state_trajs[:, 0] = sample_states
        cost_over_trajs[:, 0] = self.dynamics.boundary_fn(sample_states)
        for k in range(int((traj_times_hi) / self.dt)):
            traj_time = traj_times_hi - k * self.dt
            if self.is_time_invariant:
                model_input_time = sample_times.clone().unsqueeze(-1)
            else:
                model_input_time = torch.zeros(self.batch_size, 1).fill_(traj_time)

            curr_coords = torch.cat((model_input_time, state_trajs[:, k]), dim=-1).to(self.device)
            values_over_trajs[:, k] = model_eval(curr_coords)
            next_states, controls, dists = self.get_optimal_trajectory(curr_coords, model_eval_grad)
            
            # Trajectories only get updated if they have "started"
            not_started_times = traj_sample_times < (traj_time - self.dt / 2)
            started_times = ~not_started_times
            state_trajs[not_started_times, k + 1] = state_trajs[not_started_times, k]
            state_trajs[started_times, k + 1] = next_states[started_times].to('cpu')
            controls_trajs[started_times, k] = controls[started_times].to('cpu')
            cost_over_trajs[:, k + 1] = self.dynamics.boundary_fn(state_trajs[:, k + 1])
        curr_coords = torch.cat((model_input_time, state_trajs[:, -1]), dim=-1).to(self.device)
        values_over_trajs[:, -1] = model_eval(curr_coords).squeeze()

        ########### Step 4: Evaluate performance the trajectories ###########
        sample_values = values_over_trajs[:, 0]
        batch_scenario_costs = self.dynamics.cost_fn(state_trajs)
        batch_value_errors = batch_scenario_costs - sample_values
        batch_value_mse = torch.mean(batch_value_errors ** 2)
        false_safe_trajectories = torch.logical_and(batch_scenario_costs < 0, sample_values >= 0)

        if isinstance(values_over_trajs, torch.Tensor):
            values_over_trajs = values_over_trajs.to('cpu').numpy()
        log_dict = {
            'value_v_traj_cost_mse': batch_value_mse.item(),
            'false_safe_trajectories': torch.sum(false_safe_trajectories).item() / self.batch_size,
            'trajectories': state_trajs,
            'values': sample_values,
            'controls_trajs': controls_trajs,
            'actual_values': batch_scenario_costs,
            'values_over_trajs': values_over_trajs,
            'cost_over_trajs': cost_over_trajs,
        }
        return log_dict


class FixedRolloutTrajectories(RolloutTrajectories):
    """
    Purpose: Evaluate the model with rollouts from the value function with fixed samples.
    - Useful for training validation where we want to compare the model over time
    """
    def __init__(self, dataset, val_dict):
        RolloutTrajectories.__init__(self, dataset, val_dict)
        self.fixed_samples = True
        self.fixed_samples_validator = val_dict['fixed_samples_validator']
        _, self.sampling_states = self.generate_samples(None, 1.0, 1.0, self.fixed_samples_validator)

    def get_coords(self, model_eval, time_interval):
        sample_times = torch.ceil((torch.rand((self.batch_size)) * (time_interval[1] - time_interval[0]) + 
                                   time_interval[0]) / self.dt) * self.dt
        sample_states = self.sampling_states
        return sample_times, sample_states


class RolloutTrajectoriesWithVisuals(FixedRolloutTrajectories): 
    """
    Purpose: Evaluate the model with rollouts from the value function and visualize the trajectories with fixed samples.
    Be careful to adjust the batch_size to have the visualization be useful. 20 is a good starting point.
    """       
    def __call__(self, model_eval, model_eval_grad):
        """
        Evaluate the model with rollouts and visualize the trajectories.
        Args:
            model_eval: Function to evaluate the model on given coordinates.
            model_eval_grad: Gradient of the model evaluation function.
        Steps:
        1. Evaluate the model performance on rollouts by calling the parent class.
        2. Visualize the trajectories.
        """
        log_dict = FixedRolloutTrajectories.__call__(self, model_eval, model_eval_grad)
        import matplotlib.pyplot as plt
        plot_config = self.dataset.dynamics.plot_config()
        x_resolution = self.val_dict['x_resolution']
        y_resolution = self.val_dict['y_resolution']
        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        xys = torch.cartesian_prod(xs, ys)
        coords = torch.zeros(x_resolution * y_resolution, self.dynamics.state_dim)
        coords[:, :] = torch.tensor(plot_config['state_slices'])
        coords[:, plot_config['x_axis_idx']] = xys[:, 0]
        coords[:, plot_config['y_axis_idx']] = xys[:, 1]
        sdf_values = self.dynamics.boundary_fn(coords)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.plot(log_dict['trajectories'][:, :, plot_config['x_axis_idx']].T, 
                log_dict['trajectories'][:, :, plot_config['y_axis_idx']].T)
        ax.plot(log_dict['trajectories'][:, :1, plot_config['x_axis_idx']].T, 
                log_dict['trajectories'][:, :1, plot_config['y_axis_idx']].T, 'ro')
        ax.plot(log_dict['trajectories'][:, -1:, plot_config['x_axis_idx']].T, 
                log_dict['trajectories'][:, -1:, plot_config['y_axis_idx']].T, 'go')
        ax.contour(xs, ys, sdf_values.reshape(x_resolution, y_resolution).T, levels=[0], colors='black')
        ax.contourf(xs, ys, sdf_values.reshape(x_resolution, y_resolution).T, alpha=0.5)
        rollout_trajectory_figure = wandb.Image(fig)
        log_dict.update({'rollout_trajectory': rollout_trajectory_figure})

        fig2, ax = plt.subplots(1, 2)
        ax[0].plot(log_dict['values_over_trajs'].T)
        ax[0].set_prop_cycle(None)
        ax[0].plot(log_dict['cost_over_trajs'].T, linestyle='--', lw=2)
        ax[1].plot((log_dict['cost_over_trajs'] - log_dict['values_over_trajs']).T)
        cost_value_figure = wandb.Image(fig2)
        log_dict.update({'value_v_cost': cost_value_figure})
        return log_dict


class RolloutTrajectoriesHJR(RolloutTrajectories):
    """
    Purpose: HJR implementation of RolloutTrajectories
    """
    def __init__(self, dataset, val_dict, ground_truth):
        RolloutTrajectories.__init__(self, dataset, val_dict)
        self.ground_truth = ground_truth

    def get_optimal_trajectory(self, curr_coords, model_eval_grad):
        curr_coords = t2j(curr_coords)
        # Get gradient values
        grad_values = self.ground_truth.get_values_gradient(curr_coords[:, 1:], curr_coords[:, 0])
        # Get optimal control and disturbance
        control, disturbance = self.ground_truth.hj_dynamics.optimal_control_and_disturbance(curr_coords[:, 1:], curr_coords[:, 0], grad_values)
        # Forward simulate
        next_states = curr_coords[:,1:] + self.dt * self.ground_truth.dsdt_f(curr_coords[:, 1:], control, disturbance, curr_coords[:, 0])
        # if self.hj_dynamics.periodic_dims is not None:
        #     raise NotImplementedError("Periodic dimensions not yet supported")
        return j2t(next_states), j2t(control), j2t(disturbance)

    def __call__(self, model_eval, model_eval_grad):
        model_eval = lambda x: self.ground_truth.value_from_coords(x)
        model_eval_grad = lambda x: self.ground_truth.value_gradient_from_coords(x)
        return RolloutTrajectories.__call__(self, model_eval, model_eval_grad)
    

class FixedRolloutTrajectoriesHJR(FixedRolloutTrajectories, RolloutTrajectoriesHJR):
    """
    Purpose: HJR implementation of FixedRolloutTrajectories
    """
    def __init__(self, dataset, val_dict, ground_truth):
        FixedRolloutTrajectories.__init__(self, dataset, val_dict)
        self.ground_truth = ground_truth
    
    def get_coords(self, model_eval, time_interval):
        return FixedRolloutTrajectories.get_coords(self, model_eval, time_interval)

    def get_optimal_trajectory(self, curr_coords, model_eval_grad):
        return RolloutTrajectoriesHJR.get_optimal_trajectory(self, curr_coords, model_eval_grad)
    
    def __call__(self, model_eval, model_eval_grad):
        return RolloutTrajectoriesHJR.__call__(self, model_eval, model_eval_grad)
    

class RolloutTrajectoriesWithVisualsHJR(RolloutTrajectoriesWithVisuals, FixedRolloutTrajectoriesHJR):
    """
    Purpose: HJR implementation of RolloutTrajectoriesWithVisuals
    """
    def __init__(self, dataset, val_dict, ground_truth):
        FixedRolloutTrajectoriesHJR.__init__(self, dataset, val_dict, ground_truth)
    
    def __call__(self, model_eval, model_eval_grad):
        model_eval = lambda x: self.ground_truth.value_from_coords(x)
        return RolloutTrajectoriesWithVisuals.__call__(self, model_eval, model_eval_grad)
