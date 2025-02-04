import matplotlib.gridspec
import wandb
import torch
import os
import shutil
import time
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as spio

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm 
from utils import diff_operators
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, ValueThresholdEvaluatorandValidator, SliceSampleGenerator
from utils.progress_evaluation import *
from utils.comparisons import GroundTruthHJSolution
from dynamics import dynamics_hjr
import inspect

def parameter_list_to_suffix(parameter_list):
    parameter_suffix = ""

    if parameter_list is None: 
        return parameter_suffix 
    
    parameter_suffix += "_"
    for param in parameter_list: 
        parameter_suffix += str(param) + "p"

    return parameter_suffix

class GT_VisualizeSafeSet2D(EvaluationMetric):
    """
    Purpose: Visualize the safe set of the current model in 2D for different slices of time and other states.
    How to adjust the visualization:
    - Modify val_dict to change the resolution of the grid, the time slices, and the state slices.
    - Modify state_test_range and plot_config in the dynamics function to change the range and state slices.
    """
    def __init__(self, dataset, val_dict, parametric=None):
        """
        Args: 
            - dataset
            - val_dict
            - parametric: default None, otherwise list of parameters to evaluate model at, when specified and the dynamics model is parameteric 
        """
        self.dataset = dataset
        self.val_dict = val_dict
        self.save_path = val_dict.get('save_path', None)

        self.parametric = parametric 
        self.isHJR = False 

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
        ################ NEW  ################
        plot_config = self.dataset.dynamics.plot_config()
        times = -1 * torch.tensor(np.array(self.dataset.gt_hj_solution.times), dtype=torch.float32)

        grid_states = torch.tensor(np.array(self.dataset.gt_hj_solution.grid.states), dtype=torch.float32)
        grid_states_shape = grid_states.shape
        if isinstance(plot_config['z_axis_idx'], list):
            z_resolution = 3
            num_zs = len(plot_config['z_axis_idx'])
            zs_size = [grid_states_shape[z_idx] for z_idx in plot_config['z_axis_idx']]
            z_res_offset = 2
            zs_idxs = torch.tensor(np.array([np.linspace(0, zs_size[i]-1, z_resolution + z_res_offset) for i in range(num_zs)]), dtype=torch.int)
            
            zs_idxs = [zs_idxs[i][1:-1] for i in range(num_zs)]
            print("\n\nzs_idxs: ", zs_idxs)

            zs = torch.cartesian_prod(*zs_idxs)
        else: 
            z_resolution = 3
            zs_size = [grid_states_shape[z_idx] for z_idx in plot_config['z_axis_idx']]
            zs = np.linspace(0, zs_size[i]-1, z_resolution)

        fig = plt.figure(figsize=(6*len(zs), 5*len(times)))
        gs = matplotlib.gridspec.GridSpec(len(times), len(zs) + 1, width_ratios=[1] * len(zs) + [0.1], wspace=0.2, hspace=0.2)
        
        x_resolution = grid_states_shape[0]
        y_resolution = grid_states_shape[1]

        for i in range(len(times)): 
            if vis_type == "contourf":
                all_values_in_row = []
                all_titles_in_row = []

            for j in range(len(zs)):
                # Create fixed slices of grid states to use for grid states when creating coords
                grid_states_slices = [slice(None)] * len(grid_states_shape)               
                if isinstance(plot_config['z_axis_idx'], list): 
                    for zi in range(len(zs[j])):
                        grid_states_slices[plot_config['z_axis_idx'][zi]] = zs[j][zi]
                else: 
                    grid_states_slices[plot_config['z_axis_idx']] = zs[j]

                coords = torch.zeros(x_resolution * y_resolution, grid_states_shape[-1] + 1)
                coords[..., 0]  = times[i]
                coords[..., 1:] = torch.flatten(grid_states[grid_states_slices], 0, -2) 

                with torch.no_grad(): 
                    values = model_eval(coords)
                    sdf_values = self.dataset.dynamics.boundary_fn(coords[:, 1:].to(values.device))
                    if self.dataset.dynamics.loss_type == 'brat_hjivi' or self.dataset.dynamics.loss_type == 'gt_brat_hjivi' or self.dataset.dynamics.loss_type == 'brat_ci_hjivi':
                        avoid_values = self.dataset.dynamics.avoid_fn(coords[:, 1:].to(values.device))
                        reach_values = self.dataset.dynamics.reach_fn(coords[:, 1:].to(values.device))

                xs_plot = np.linspace(-1, 1, x_resolution)
                ys_plot = np.linspace(-1, 1, y_resolution)

                # Create title
                individual_coords = coords[0][1:].clone().detach() # individual example of coords - to grab slice values 
                if isinstance(plot_config['z_axis_idx'], list):
                    ax_title = 't = %0.2f, %s' % (
                        times[i],
                        ', '.join(['%s = %0.2f' % (plot_config['state_labels'][z_idx], 
                                                individual_coords[z_idx].item()) # get the actual value from a state
                                for k, z_idx in enumerate(plot_config['z_axis_idx'])])
                    )
                else:
                    ax_title = 't = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], individual_coords[zs[j]].item()) 
                
                if vis_type == "imshow":
                    ax = fig.add_subplot(gs[i, j])
                    s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                    # Go from xs to (-1, 1) and ys to (-1, 1)

                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        ax.contour(xs_plot, ys_plot, sdf_values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T, levels=[0], colors='black')
                    else:
                        ax.contour(xs_plot, ys_plot, avoid_values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T, levels=[0], colors='black')
                        ax.contour(xs_plot, ys_plot, reach_values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T, levels=[0], colors='green')

                    ax.set_title(ax_title)

                elif vis_type == "contourf":
                    # s = ax.contourf(xs_plot, ys_plot, values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T)
                    # store values later for plotting 
                    all_values_in_row.append(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T)
                    all_titles_in_row.append(ax_title)
                    continue 
            
            if vis_type == "imshow":
                cax = fig.add_subplot(gs[i, -1])
                fig.colorbar(s, cax=cax, orientation='vertical')
                cax = fig.add_subplot(gs[i, -1])
                cbar = fig.colorbar(s, cax=cax, orientation='vertical')

            elif vis_type == "contourf":
                # Get the min max value and then plot 
                min_value = min([np.min(row_values) for row_values in all_values_in_row])
                max_value = max([np.max(row_values) for row_values in all_values_in_row])   
                for j in range(len(zs)):
                    ax = fig.add_subplot(gs[i, j])
                    s = ax.contourf(xs_plot, ys_plot, all_values_in_row[j], vmin=min_value, vmax=max_value)

                    individual_coords = coords[0][1:].clone().detach() # individual example of coords - to grab slice values 
                    ax_title = all_titles_in_row[j]
                    
                    ax.set_title(ax_title)
                    fig.colorbar(s, ax=ax)

                

        fig.tight_layout()
        if self.save_path is not None: 
            if self.save_path.endswith('.png'):
                fig.savefig(self.save_path)
            else: 
                fig.savefig(self.save_path + '/model_safe_set.png')
        
        return {"safe_set": wandb.Image(fig)}

class GT_VisualizeSafeSet2DHJR(GT_VisualizeSafeSet2D):
    """
    Implementation for HJR
    """
    def __init__(self, dataset, val_dict, parametric=None): 
        super().__init__(dataset, val_dict, parametric=parametric)
        self.isHJR = True

class GT_VisualizeValueDifference2D(GT_VisualizeSafeSet2D):
    """
    Purpose: Visualize the difference between the model and the ground truth in 2D for different slices of time and 
             other states.
    See VisualizeSafeSet2D for details on how to adjust the visualization.
    """
    def __init__(self, dataset, val_dict, ground_truth, parametric=None):
        self.ground_truth = ground_truth
        super().__init__(dataset, val_dict, parametric=parametric)

    def __call__(self, model_eval, model_eval_grad):
        """
        Args:
            model_eval: Function to evaluate the model on given coordinates.
            model_eval_grad: Gradient of the model evaluation function (unused here). 
        """
        # new_eval = lambda x: model_eval(x) - self.ground_truth.value_from_coords(x)
        def new_eval(x): 

            if hasattr(self.dataset.dynamics, 'parametric_dims') and self.parametric is not None:
                gt_coords = x[:, [0] + self.dataset.dynamics.coord_state_dims] # include time at 0 and shifted state dims 
            else: 
                gt_coords = x

            return model_eval(x) - self.ground_truth.value_from_coords(gt_coords)

        log_dict = super().__call__(new_eval, model_eval_grad, vis_type='contourf')
        new_dict = {}
        for key, value in log_dict.items():
            new_dict[key + "_diff"] = value
        return new_dict

########################################################################################################################################

class GTExperiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb, device, validation_dict={}):
        """
        Sets up the experiment with the model, dataset, and experiment directory.
        For validation it does the following:
        - If state_dim <= 5, it generates ground truth for the validation metrics and logs gt solutions
        - It sets up the validation metrics for every validation step (distinction between visual and non-visual)

        TODO: Ideally we specify the validation metrics to use in the setup, but for now it is hardcoded
        """
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb
        self.device = device

        self.rollout_batch_size = 5000
        self.visual_rollout_batch_size = 20 

        # self.emperical_cost_validation_metric = EmpiricalPerformance(self.dataset.dynamics, 0.01, device=self.device)
        is_parametric = hasattr(self.dataset.dynamics, 'parametric_dims')
        
        # Initialize 
        self.validation_metrics = {} # key: parametric test slice name, value: {"parameter": parameter value list, "metrics": list of callable metrics}
        self.visual_only_validation_metrics = {} # key: parametric test slice name, value: {"parameter": parameter value list, "metrics": list of callable metrics}

        if is_parametric: 
            parametric_test_slices = self.dataset.dynamics.parameter_test_slices()
            for parametric_test_slice in parametric_test_slices: 
                parametric_key = str(parametric_test_slice)
                self.validation_metrics[parametric_key] = {"parameter": parametric_test_slice, "metrics": [GT_VisualizeSafeSet2D(self.dataset, validation_dict, parametric=parametric_test_slice)]}
                self.visual_only_validation_metrics[parametric_key] = {"parameter": parametric_test_slice, "metrics": []}
        else: 
            parametric_test_slices = [None]
            self.validation_metrics[""] = {"parameter": None, "metrics": [GT_VisualizeSafeSet2D(self.dataset, validation_dict, parametric=parametric_test_slices[0])]}
            self.visual_only_validation_metrics[""] = {"parameter": None, "metrics": []}


        for parametric_test_slice_num, parametric_test_slice in enumerate(parametric_test_slices):
            parametric_key = str(parametric_test_slice) if parametric_test_slice is not None else ""
            parametric_suffix = parameter_list_to_suffix(parametric_test_slice) #"0.0p0.0" #"_" + parametric_key if parametric_test_slice is not None else parametric_key 
            print("parameter_suffix: ", parametric_suffix)

            if (hasattr(self.dataset.dynamics, 'parametric_dims') and (self.dataset.dynamics.state_dim - len(self.dataset.dynamics.parametric_dims) <= 5)) or (self.dataset.dynamics.state_dim <= 5):
            # if self.dataset.dynamics.state_dim <= 5:
                # Generate ground truth for the validation metrics
                # TODO: Move this into GroundTruthHJSolution initialization
                dynamics_class_name = self.dataset.dynamics.__class__.__name__
                dynamics_class = getattr(dynamics_hjr, dynamics_class_name)
                dynamics_params = inspect.signature(dynamics_class).parameters
                dynamics_args = {argname: getattr(self.dataset.dynamics, argname) for argname in dynamics_params 
                                if hasattr(self.dataset.dynamics, argname)}
                dynamics_args['tMin'] = self.dataset.tMin
                dynamics_args['tMax'] = self.dataset.tMax

                if parametric_test_slice is not None: 
                    curr_dynamics_args = dynamics_args.copy()
                    for param_num in range(len(self.dataset.dynamics.parametric_dims)): 
                        curr_dynamics_args[self.dataset.dynamics.parametric_names[param_num]] = parametric_test_slice[param_num]

                    print("Starting parameteric test slice: ", parametric_test_slice)
                    hj_dyn = dynamics_class(self.dataset.dynamics, **curr_dynamics_args)
                    gt = GroundTruthHJSolution(hj_dyn)
                    print("Finished parameteric test slice: ", parametric_test_slice)
                else: 
                    hj_dyn = dynamics_class(self.dataset.dynamics, **dynamics_args)
                    print("\n\nJust grabbing solution from dataset\n\n")
                    gt = self.dataset.gt_hj_solution #GroundTruthHJSolution(hj_dyn)
            
                # Visualize 2D safe set for ground truth (for comparison)
                model_eval_gt = lambda coords: gt.value_from_coords(coords)
                log_initial = GT_VisualizeSafeSet2DHJR(self.dataset, validation_dict, parametric=parametric_test_slice)(model_eval_gt, None)
                log_initial = {key + '_gt' + parametric_suffix: value for key, value in log_initial.items()}
                log_initial['step'] = 0

                print("Starting GT rollouts")

                # Rollout of ground truth (large batch size, so no plotting)
                converged_values_validator = ValueThresholdEvaluatorandValidator(eval_fn = gt.value_from_coords, v_min=0.0, v_max=1.0)
                validation_dict['fixed_samples_validator'] = converged_values_validator
                validation_dict['rollout_batch_size'] = self.rollout_batch_size
                gt_rollout = FixedRolloutTrajectoriesHJR(self.dataset, validation_dict, gt)
                gt_rollout.vf_times = self.dataset.tMax  # FIXME: Temp
                gt_rollout.rollout_times = self.dataset.tMax  # FIXME: temp
                trajectory_log = gt_rollout(model_eval_gt, None)
                for item, value in trajectory_log.items():
                    if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        log_initial[item + "_gt" + parametric_suffix] = value

                print("Starting GT visual rollouts")
                
                # Small rollout of ground truth specifically for plotting
                validation_dict['rollout_batch_size'] = self.visual_rollout_batch_size
                gt_rollout_viz = RolloutTrajectoriesWithVisualsHJR(self.dataset, validation_dict, gt)
                gt_rollout_viz.vf_times = self.dataset.tMax  # FIXME: Temp
                gt_rollout_viz.rollout_times = self.dataset.tMax  # FIXME: temp
                trajectory_viz_log = gt_rollout_viz(model_eval_gt, None)
                for item, value in trajectory_viz_log.items():
                    if isinstance(value, wandb.Image):
                        log_initial[item + "_gt" + parametric_suffix] = value
                
                if self.use_wandb:
                    wandb.log(log_initial)
                    print("Wandb should've updated by now...")

                ########### Validation metrics for all iterations ############    
                self.validation_metrics[parametric_key]["metrics"].append(GT_VisualizeValueDifference2D(self.dataset, validation_dict, gt, parametric=parametric_test_slice))
                
                print("Adding Binary Safety Difference Metric")
                safety_metrics = QuantifyBinarySafetyDifference(self.dataset, validation_dict, gt, parametric=parametric_test_slice)
                if is_parametric: 
                    safety_metrics.eval_states[..., self.dataset.dynamics.state_dims] = j2t(gt.grid.states.reshape(-1, gt.grid.ndim)).to(safety_metrics.eval_states.device)
                else: 
                    safety_metrics.eval_states = j2t(gt.grid.states.reshape(-1, gt.grid.ndim))
                self.validation_metrics[parametric_key]["metrics"].append(safety_metrics)
        
            else:
                safety_metrics = QuantifyBinarySafety(self.dataset, validation_dict)
                self.validation_metrics[parametric_key]["metrics"].append(safety_metrics)

            standard_value_validator = ValueThresholdEvaluatorandValidator(eval_fn=self.dataset.dynamics.boundary_fn, v_min=0.0, v_max=2.0)
            validation_dict['fixed_samples_validator'] = standard_value_validator
            validation_dict['rollout_batch_size'] = self.rollout_batch_size
            traj_rollout = FixedRolloutTrajectories(self.dataset, validation_dict, parametric=parametric_test_slice)

            validation_dict['fixed_samples_validator'] = standard_value_validator
            validation_dict['rollout_batch_size'] = self.visual_rollout_batch_size 
            traj_rollout_viz = RolloutTrajectoriesWithVisuals(self.dataset, validation_dict, parametric=parametric_test_slice)

            if (hasattr(self.dataset.dynamics, 'parametric_dims') and (self.dataset.dynamics.state_dim - len(self.dataset.dynamics.parametric_dims) <= 5)) or (self.dataset.dynamics.state_dim <= 5):
                traj_rollout.sampling_states = gt_rollout.sampling_states
                traj_rollout_viz.sampling_states = gt_rollout_viz.sampling_states

            self.validation_metrics[parametric_key]["metrics"].append(traj_rollout)
            self.visual_only_validation_metrics[parametric_key]["metrics"].append(traj_rollout_viz)

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_current.pth')
            self.model.load_state_dict(torch.load(model_path))
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

    def validate(self, epoch):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)
        
        def model_eval(coords):
            results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(self.device))})
            vals = self.dataset.dynamics.io_to_value(results['model_in'], results['model_out'].squeeze(dim=-1).detach())
            return vals.detach()
        
        def model_eval_grad(coords):
            self.model.requires_grad_(True)
            results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(self.device))})
            vals = self.dataset.dynamics.io_to_dv(results['model_in'], results['model_out'].squeeze(dim=-1)).detach()
            self.model.requires_grad_(False)
            return vals
        
        learned_model_eval = model_eval
        wandb_log = {'step': epoch}

        for parametric_key in self.validation_metrics.keys(): 
            parametric_test_slice = self.validation_metrics[parametric_key]["parameter"]
            parametric_suffix = parameter_list_to_suffix(parametric_test_slice) #"_" + parametric_key if parametric_test_slice is not None else parametric_key

            for validation_metric in self.validation_metrics[parametric_key]["metrics"]:
                validation_metric.update_counters()
                log = validation_metric(learned_model_eval, model_eval_grad)
                for key, value in log.items():
                    key = key + parametric_suffix 
                    if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        wandb_log[key] = value
                    elif isinstance(value, wandb.Image):
                        wandb_log[key] = value
            
            for validation_metric in self.visual_only_validation_metrics[parametric_key]["metrics"]:
                validation_metric.update_counters()
                log = validation_metric(learned_model_eval, model_eval_grad)
                for key, value in log.items():
                    key = key + parametric_suffix
                    if isinstance(value, wandb.Image):
                        wandb_log[key] = value

            if self.use_wandb:
                wandb.log(wandb_log)
            plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
    
    def train(
            self, batch_size, epochs, lr, 
            steps_til_summary, epochs_til_checkpoint, 
            loss_fn, clip_grad, use_lbfgs, adjust_relative_grads, 
            val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
            use_CSL, CSL_lr, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size,
        ):

        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')

        train_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Setup Logging
        training_dir = os.path.join(self.experiment_dir, 'training')
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)
        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        writer = SummaryWriter(summaries_dir)

        start_time = time.time()
        train_losses = []
        total_steps = 0

        # TEST start by validating 
        print("Running initial validation before training")
        self.validate(epoch=0)

        # Looping through data 
        for epoch in tqdm(range(epochs)): 
            epoch_steps = 0 

            print("At start of epoch: ", epoch)
            # import pdb; pdb.set_trace()
            for model_coords, gt_value in tqdm(train_dataloader): 
                epoch_steps += 1
                total_steps += 1
                model_coords = model_coords.to(self.device)
                gt_value = gt_value.to(self.device)

                model_results = self.model({"coords": model_coords}) # model_coords normalized during dataset creation. 
                model_value = self.dataset.dynamics.io_to_value(input=model_results["model_in"].detach(), output=model_results['model_out'].squeeze(dim=-1)) # unnormalize the outputted value.
                train_loss = loss_fn(model_value, gt_value)

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                    optim.step()
                else: 
                    raise NotImplementedError

                train_losses.append(train_loss.detach().item())

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d / Epochs %d, Total loss %0.6f, iteration time %0.6f" % (epoch, epochs, train_loss, time.time() - start_time))
                    # pdb.set_trace()
                    print("model value: ", model_value[:100])
                    print("\n\ngt value: ", gt_value[:100])
                    if self.use_wandb:
                        wandb.log({
                            'step': total_steps,
                            'train_loss': train_loss,
                        })

            if not (epoch+1) % epochs_til_checkpoint:
                # Saving the optimizer state is important to produce consistent results
                checkpoint = { 
                    'epoch': epoch+1,
                    'model': self.model.state_dict(),
                    'optimizer': optim.state_dict()}
                torch.save(checkpoint,
                    os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                    np.array(train_losses))
                
                try: 
                    self.validate(epoch=epoch+1)
                except: 
                    print("Validation failed .... likely due to plotting issues ?")
                        
        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)
        

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
        raise NotImplementedError
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)
        raise NotImplementedError
        
class DeepReach(GTExperiment):
    def init_special(self):
        pass