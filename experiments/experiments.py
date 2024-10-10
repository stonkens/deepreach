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
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator
from utils.progress_evaluation import CompareWithAlternative, GroundTruthHJSolution, EmpiricalPerformance
from dynamics import dynamics_hjr
import inspect

class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb, device):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb
        self.device = device
        self.validation_metrics = lambda *args, **kwargs: {} 
        self.emperical_cost_validation_metric = EmpiricalPerformance(self.dataset.dynamics, 0.005, device=self.device)
        if self.dataset.dynamics.state_dim <= 5:
            # Get the name of the dynamics class
            dynamics_class_name = self.dataset.dynamics.__class__.__name__
            dynamics_class = getattr(dynamics_hjr, dynamics_class_name)
            dynamics_params = inspect.signature(dynamics_class).parameters
            dynamics_args = {argname: getattr(self.dataset.dynamics, argname) for argname in dynamics_params 
                             if hasattr(self.dataset.dynamics, argname)}
            hj_dyn = dynamics_class(self.dataset.dynamics, **dynamics_args)
            gt = GroundTruthHJSolution(hj_dyn)
            self.validation_metrics = CompareWithAlternative(self.dataset.dynamics, gt, [], 
                                                             gt.grid.states.reshape(-1, gt.grid.ndim), gt.times)
            self.gt_cost_validation = EmpiricalPerformance(self.dataset.dynamics, 0.002, device=self.device, 
                                                           batch_size=100, fixed_samples=True,
                                                           fixed_samples_validator=gt)

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

    def validate(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

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
        
        fig = plt.figure(figsize=(6*len(zs), 5*len(times)))
        gs = matplotlib.gridspec.GridSpec(len(times), len(zs) + 1, width_ratios=[1] * len(zs) + [0.1], wspace=0.2, hspace=0.2)
        if hasattr(self.validation_metrics, 'include_plot'):
            fig2 = plt.figure(figsize=(6*len(zs), 5*len(times)))
            gs2 = matplotlib.gridspec.GridSpec(len(times), len(zs) + 1, width_ratios=[1] * len(zs) + [0.1], wspace=0.2, hspace=0.2)
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
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(self.device))})
                    if hasattr(self.validation_metrics, 'include_plot'):
                        values_validation = self.validation_metrics.get_comparison_plot_data(coords.to(self.device))
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())

                    sdf_values = self.dataset.dynamics.boundary_fn(coords[:, 1:].to(self.device))

                # ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax = fig.add_subplot(gs[i, j])

                if isinstance(plot_config['z_axis_idx'], list):
                    ax_title = 't = %0.2f, %s' % (
                        times[i],
                        ', '.join(['%s = %0.2f' % (plot_config['state_labels'][z_idx], zs[j][k].item()) 
                                   for k, z_idx in enumerate(plot_config['z_axis_idx'])])
                    )
                else:
                    ax_title = 't = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j])
                s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                # Go from xs to (-1, 1) and ys to (-1, 1)
                xs_plot = np.linspace(-1, 1, x_resolution)
                ys_plot = np.linspace(-1, 1, y_resolution)
                ax.contour(xs_plot, ys_plot, sdf_values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T, levels=[0], colors='black')
                ax.set_title(ax_title)
                if hasattr(self.validation_metrics, 'include_plot'):
                    ax2 = fig2.add_subplot(gs2[i, j])
                    ax2.imshow(1*(values_validation.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                    ax2.contour(xs_plot, ys_plot, sdf_values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T, levels=[0], colors='black')
                    ax2.set_title(ax_title)
                # fig.colorbar(s) 
            cax = fig.add_subplot(gs[i, -1])
            fig.colorbar(s, cax=cax, orientation='vertical')#, shrink=0.5, aspect=20)
            if hasattr(self.validation_metrics, 'include_plot'):
                cax2 = fig2.add_subplot(gs2[i, -1])
                fig2.colorbar(s, cax=cax2, orientation='vertical')
        fig.tight_layout()
        fig.savefig(save_path)

        # Add possible progress evaluation metrics here
        wandb_log = self.validation_metrics(self.model, add_temporal_data=True)
        curr_t = self.dataset.tMin + (self.dataset.tMax - self.dataset.tMin) * self.dataset.counter / self.dataset.counter_end
        added_log = self.emperical_cost_validation_metric(self.model, vf_times=[max(0, curr_t - 0.1), curr_t])
        for key, value in added_log.items():
            if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                wandb_log[key] = value
        if hasattr(self, 'gt_cost_validation'):
            another_log = self.gt_cost_validation(self.model, vf_times=max(0.0, curr_t - 0.1), rollout_times=self.dataset.tMax)
            for key, value in another_log.items():
                if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                    mod_key = 'gt_' + key
                    wandb_log[mod_key] = value
            fig3, ax = plt.subplots()
            ax.plot(another_log['trajectories'][:, :, 0].T, another_log['trajectories'][:, :, 1].T)
            ax.plot(another_log['trajectories'][:,:1, 0].T, another_log['trajectories'][:,:1, 1].T, '*')
            ground_truth = self.validation_metrics.alt_method
            ax.contourf(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[-1][:,:,25,25].T)
            ax.contour(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[0][:,:,25,25].T, levels=[0], colors='k')
            ax.set_xlim([ground_truth.grid.coordinate_vectors[0][0], ground_truth.grid.coordinate_vectors[0][-1]])
            ax.set_ylim([ground_truth.grid.coordinate_vectors[1][0], ground_truth.grid.coordinate_vectors[1][-1]])
            wandb_log['rollouts_plot'] = wandb.Image(fig3)
            fig4, ax = plt.subplots()
            ax.plot(another_log['values_over_trajs'][::10].T)
            ax.plot(torch.zeros_like(another_log['values_over_trajs'][0]), 'k--')
            ax.set_ylim([-1, 1])
            wandb_log['rollouts_values_plot'] = wandb.Image(fig4)
            fig5, ax = plt.subplots()
            ax.plot(another_log['cost_over_trajs'][::10].T)
            ax.plot(torch.zeros_like(another_log['cost_over_trajs'][0]), 'k--')
            ax.set_ylim([-1, 1])
            wandb_log['rollouts_costs_plot'] = wandb.Image(fig5)
        wandb_log['step'] = epoch
        wandb_log['val_plot'] = wandb.Image(fig)
        if hasattr(self.validation_metrics, 'include_plot'):
            wandb_log['val_plot_gt'] = wandb.Image(fig2)
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

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())

        # copy settings from Raissi et al. (2019) and here 
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')
        
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0

        if adjust_relative_grads:
            new_weight = 1

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):
                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt
                
                # self-supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    # model_input contains the states.
                    # gt contains boundary values and dirchelet masks (optionally reach and avoid values)
                    start_time = time.time()
                
                    model_input = {key: value.to(self.device) for key, value in model_input.items()}
                    gt = {key: value.to(self.device) for key, value in gt.items()}

                    model_results = self.model({'coords': model_input['model_coords']})

                    states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
                    dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                    boundary_values = gt['boundary_values']
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']
                    dirichlet_masks = gt['dirichlet_masks']

                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    else:
                        raise NotImplementedError
                    
                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean() 
                            train_loss.backward()
                            return train_loss
                        optim.step(closure)

                    # Adjust the relative magnitude of the losses if required
                    if self.dataset.dynamics.deepreach_model in ['vanilla', 'diff'] and adjust_relative_grads:
                        if losses['diff_constraint_hom'] > 0.01:
                            params = OrderedDict(self.model.named_parameters())
                            # Gradients with respect to the PDE loss
                            optim.zero_grad()
                            losses['diff_constraint_hom'].backward(retain_graph=True)
                            grads_PDE = []
                            for key, param in params.items():
                                grads_PDE.append(param.grad.view(-1))
                            grads_PDE = torch.cat(grads_PDE)

                            # Gradients with respect to the boundary loss
                            optim.zero_grad()
                            losses['dirichlet'].backward(retain_graph=True)
                            grads_dirichlet = []
                            for key, param in params.items():
                                grads_dirichlet.append(param.grad.view(-1))
                            grads_dirichlet = torch.cat(grads_dirichlet)

                            # # Plot the gradients
                            # import seaborn as sns
                            # import matplotlib.pyplot as plt
                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # fig.savefig('gradient_visualization.png')

                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # grads_dirichlet_normalized = grads_dirichlet * torch.mean(torch.abs(grads_PDE))/torch.mean(torch.abs(grads_dirichlet))
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet_normalized.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # ax.set_xlim([-1000.0, 1000.0])
                            # fig.savefig('gradient_visualization_normalized.png')

                            # Set the new weight according to the paper
                            # num = torch.max(torch.abs(grads_PDE))
                            num = torch.mean(torch.abs(grads_PDE))
                            den = torch.mean(torch.abs(grads_dirichlet))
                            new_weight = 0.9*new_weight + 0.1*num/den
                            losses['dirichlet'] = new_weight*losses['dirichlet']
                        writer.add_scalar('weight_scaling', new_weight, total_steps)

                    # import ipdb; ipdb.set_trace()

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_name == 'dirichlet':
                            writer.add_scalar(loss_name, single_loss/new_weight, total_steps)
                        else:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(self.model.state_dict(),
                                os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                        optim.step()

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                        if self.use_wandb:
                            wandb.log({
                                'step': epoch,
                                'train_loss': train_loss,
                                'pde_loss': losses['diff_constraint_hom'],
                            })

                    total_steps += 1

                # cost-supervised learning (CSL) phase
                if use_CSL and not self.dataset.pretrain and (epoch-last_CSL_epoch) >= epochs_til_CSL:
                    last_CSL_epoch = epoch
                    
                    # generate CSL datasets
                    self.model.eval()

                    CSL_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_samples, 100000), sample_batch_size=min(10*num_CSL_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_samples, tStart_generator=lambda n : torch.zeros(n).uniform_(self.dataset.tMin, CSL_tMax)
                    )
                    CSL_coords = torch.cat((CSL_dataset['times'].unsqueeze(-1), CSL_dataset['states']), dim=-1)
                    CSL_costs = CSL_dataset['costs']

                    num_CSL_val_samples = int(0.1*num_CSL_samples)
                    CSL_val_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_val_samples, tStart_generator=lambda n : torch.zeros(n).uniform_(self.dataset.tMin, CSL_tMax)
                    )
                    CSL_val_coords = torch.cat((CSL_val_dataset['times'].unsqueeze(-1), CSL_val_dataset['states']), dim=-1)
                    CSL_val_costs = CSL_val_dataset['costs']

                    CSL_val_tMax_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=self.dataset.tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_val_samples # no tStart_generator, since I want all tMax times
                    )
                    CSL_val_tMax_coords = torch.cat((CSL_val_tMax_dataset['times'].unsqueeze(-1), CSL_val_tMax_dataset['states']), dim=-1)
                    CSL_val_tMax_costs = CSL_val_tMax_dataset['costs']
                    
                    self.model.train()

                    # CSL optimizer
                    CSL_optim = torch.optim.Adam(lr=CSL_lr, params=self.model.parameters())

                    # initial CSL val loss
                    CSL_val_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.to(self.device))})
                    CSL_val_preds = self.dataset.dynamics.io_to_value(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                    CSL_val_errors = CSL_val_preds - CSL_val_costs.to(self.device)
                    CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
                    CSL_initial_val_loss = CSL_val_loss
                    if self.use_wandb:
                        wandb.log({
                            "step": epoch,
                            "CSL_val_loss": CSL_val_loss.item()
                        })

                    # initial self-supervised learning (SSL) val loss
                    # right now, just took code from dataio.py and the SSL training loop above; TODO: refactor all this for cleaner modular code
                    CSL_val_states = CSL_val_coords[..., 1:].to(self.device)
                    CSL_val_dvs = self.dataset.dynamics.io_to_dv(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                    CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(CSL_val_states)
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        CSL_val_reach_values = self.dataset.dynamics.reach_fn(CSL_val_states)
                        CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(CSL_val_states)
                    CSL_val_dirichlet_masks = CSL_val_coords[:, 0].to(self.device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
                    else:
                        NotImplementedError
                    SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
                    if self.use_wandb:
                        wandb.log({
                            "step": epoch,
                            "SSL_val_loss": SSL_val_loss.item()
                        })

                    # CSL training loop
                    for CSL_epoch in tqdm(range(max_CSL_epochs)):
                        CSL_idxs = torch.randperm(num_CSL_samples)
                        for CSL_batch in range(math.ceil(num_CSL_samples/CSL_batch_size)):
                            CSL_batch_idxs = CSL_idxs[CSL_batch*CSL_batch_size:(CSL_batch+1)*CSL_batch_size]
                            CSL_batch_coords = CSL_coords[CSL_batch_idxs]

                            CSL_batch_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_batch_coords.to(self.device))})
                            CSL_batch_preds = self.dataset.dynamics.io_to_value(CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
                            CSL_batch_costs = CSL_costs[CSL_batch_idxs].to(self.device)
                            CSL_batch_errors = CSL_batch_preds - CSL_batch_costs
                            CSL_batch_loss = CSL_loss_weight*torch.mean(torch.pow(CSL_batch_errors, 2))

                            CSL_batch_states = CSL_batch_coords[..., 1:].to(self.device)
                            CSL_batch_dvs = self.dataset.dynamics.io_to_dv(CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
                            CSL_batch_boundary_values = self.dataset.dynamics.boundary_fn(CSL_batch_states)
                            if self.dataset.dynamics.loss_type == 'brat_hjivi':
                                CSL_batch_reach_values = self.dataset.dynamics.reach_fn(CSL_batch_states)
                                CSL_batch_avoid_values = self.dataset.dynamics.avoid_fn(CSL_batch_states)
                            CSL_batch_dirichlet_masks = CSL_batch_coords[:, 0].to(self.device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                            if self.dataset.dynamics.loss_type == 'brt_hjivi':
                                SSL_batch_losses = loss_fn(CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_dirichlet_masks)
                            elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                                SSL_batch_losses = loss_fn(CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_reach_values, CSL_batch_avoid_values, CSL_batch_dirichlet_masks)
                            else:
                                NotImplementedError
                            SSL_batch_loss = SSL_batch_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_batch_dirichlet_masks == False))
                            
                            CSL_optim.zero_grad()
                            SSL_batch_loss.backward(retain_graph=True)
                            if (not use_lbfgs) and clip_grad: # no adjust_relative_grads, because I assume even with adjustment, the diff_constraint_hom remains unaffected and the only other loss (dirichlet) is zero
                                if isinstance(clip_grad, bool):
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                                else:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
                            CSL_batch_loss.backward()
                            CSL_optim.step()
                        
                        # evaluate on CSL_val_dataset
                        CSL_val_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.to(self.device))})
                        CSL_val_preds = self.dataset.dynamics.io_to_value(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                        CSL_val_errors = CSL_val_preds - CSL_val_costs.to(self.device)
                        CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
                    
                        CSL_val_states = CSL_val_coords[..., 1:].to(self.device)
                        CSL_val_dvs = self.dataset.dynamics.io_to_dv(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                        CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(CSL_val_states)
                        if self.dataset.dynamics.loss_type == 'brat_hjivi':
                            CSL_val_reach_values = self.dataset.dynamics.reach_fn(CSL_val_states)
                            CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(CSL_val_states)
                        CSL_val_dirichlet_masks = CSL_val_coords[:, 0].to(self.device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                        if self.dataset.dynamics.loss_type == 'brt_hjivi':
                            SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
                        elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                            SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
                        else:
                            raise NotImplementedError
                        SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
                    
                        CSL_val_tMax_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_tMax_coords.to(self.device))})
                        CSL_val_tMax_preds = self.dataset.dynamics.io_to_value(CSL_val_tMax_results['model_in'], CSL_val_tMax_results['model_out'].squeeze(dim=-1))
                        CSL_val_tMax_errors = CSL_val_tMax_preds - CSL_val_tMax_costs.to(self.device)
                        CSL_val_tMax_loss = torch.mean(torch.pow(CSL_val_tMax_errors, 2))
                        
                        # log CSL losses, recovered_safe_set_fracs
                        if self.dataset.dynamics.set_mode == 'reach':
                            CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(CSL_batch_costs.to(self.device) < 0) / len(CSL_batch_preds)
                            CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds < torch.min(CSL_batch_preds[CSL_batch_costs.to(self.device) > 0])) / len(CSL_batch_preds)
                            CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_costs.to(self.device) < 0) / len(CSL_val_preds)
                            CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds < torch.min(CSL_val_preds[CSL_val_costs.to(self.device) > 0])) / len(CSL_val_preds)
                            CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_tMax_costs.to(self.device) < 0) / len(CSL_val_tMax_preds)
                            CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds < torch.min(CSL_val_tMax_preds[CSL_val_tMax_costs.to(self.device) > 0])) / len(CSL_val_tMax_preds)
                        elif self.dataset.dynamics.set_mode == 'avoid':
                            CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(CSL_batch_costs.to(self.device) > 0) / len(CSL_batch_preds)
                            CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(CSL_batch_preds[CSL_batch_costs.to(self.device) < 0])) / len(CSL_batch_preds)
                            CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_costs.to(self.device) > 0) / len(CSL_val_preds)
                            CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds > torch.max(CSL_val_preds[CSL_val_costs.to(self.device) < 0])) / len(CSL_val_preds)
                            CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_tMax_costs.to(self.device) > 0) / len(CSL_val_tMax_preds)
                            CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds > torch.max(CSL_val_tMax_preds[CSL_val_tMax_costs.to(self.device) < 0])) / len(CSL_val_tMax_preds)
                        else:
                            raise NotImplementedError
                        if self.use_wandb:
                            wandb.log({
                                "step": epoch+(CSL_epoch+1)*int(0.5*epochs_til_CSL/max_CSL_epochs),
                                "CSL_train_batch_loss": CSL_batch_loss.item(),
                                "SSL_train_batch_loss": SSL_batch_loss.item(),
                                "CSL_val_loss": CSL_val_loss.item(),
                                "SSL_val_loss": SSL_val_loss.item(),
                                "CSL_val_tMax_loss": CSL_val_tMax_loss.item(),
                                "CSL_train_batch_theoretically_recoverable_safe_set_frac": CSL_train_batch_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_val_theoretically_recoverable_safe_set_frac": CSL_val_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_val_tMax_theoretically_recoverable_safe_set_frac": CSL_val_tMax_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_train_batch_recovered_safe_set_frac": CSL_train_batch_recovered_safe_set_frac.item(),
                                "CSL_val_recovered_safe_set_frac": CSL_val_recovered_safe_set_frac.item(),
                                "CSL_val_tMax_recovered_safe_set_frac": CSL_val_tMax_recovered_safe_set_frac.item(),
                            })

                        if CSL_val_loss < CSL_loss_frac_cutoff*CSL_initial_val_loss:
                            break

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
                    self.validate(
                        epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        testing_dir = os.path.join(self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
        if os.path.exists(testing_dir):
            overwrite = input("The testing directory %s already exists. Overwrite? (y/n)"%testing_dir)
            if not (overwrite == 'y'):
                print('Exiting.')
                quit()
            shutil.rmtree(testing_dir)
        os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')
            models = os.path.join(self.experiment_dir, 'training', 'checkpoints')
            checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(models) if 'model_epoch' in f]

            for i in tqdm(range(len(checkpoints)), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                raise NotImplementedError

        else:
            import matplotlib.pyplot as plt
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)
            # Get max time of checkpoint from the name (model_epoch_%04d.pth)
            checkpoint_max_time = checkpoint_toload / (self.dataset.tMax - self.dataset.tMin) / self.dataset.counter_end
            curr_t = max(0, checkpoint_max_time)
            if self.dataset.dynamics.state_dim <= 5:
            # Get the name of the dynamics class
                self.cost_validation = EmpiricalPerformance(self.dataset.dynamics, 0.002, device=self.device,
                                                            batch_size=100, fixed_samples=True, 
                                                            fixed_samples_validator=self.validation_metrics.alt_method)
            
            
                results = self.cost_validation(self.model, vf_times=curr_t-0.1, rollout_times=1.0)
                print("Ground truth comparison")
                for key, value in results.items():
                    if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        print('%s: %f' % (key, value))
                
                plt.plot(results['trajectories'][..., 0].T, results['trajectories'][..., 1].T)
                plt.plot(results['trajectories'][:,:1, 0].T, results['trajectories'][:,:1, 1].T, '*')
                ground_truth = self.validation_metrics.alt_method
                plt.contourf(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[-1][:,:,25,25].T)
                plt.contour(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[0][:,:,25,25].T, levels=[0], colors='k')
                plt.xlim([-5, 5])
                plt.ylim([-0.2, 2.8])
                plt.savefig(os.path.join(testing_dir, 'trajectory_guaranteed_safe.png'))

                fig, ax = plt.subplots()
                ax.plot(results['values_over_trajs'][::10].T)
                ax.set_ylim([-1, 1])
                fig.savefig(os.path.join(testing_dir, 'values_over_trajs.png'))

            self.emperical_cost_validation_metric = EmpiricalPerformance(self.dataset.dynamics, 0.001, device=self.device, batch_size=100)
            results = self.emperical_cost_validation_metric(self.model, vf_times=[max(0, curr_t - 0.1), curr_t], rollout_times=1.0)
            for key, value in results.items():
                if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                    print('%s: %f' % (key, value))
            
            plt.plot(results['trajectories'][..., 0].T, results['trajectories'][..., 1].T)
            plt.plot(results['trajectories'][:,:1, 0].T, results['trajectories'][:,:1, 1].T, '*')
            ground_truth = self.validation_metrics.alt_method
            plt.contourf(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[-1][:,:,25,25].T)
            plt.contour(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[0][:,:,25,25].T, levels=[0], colors='k')
            plt.xlim([-5, 5])
            plt.ylim([-0.2, 2.8])
            plt.savefig(os.path.join(testing_dir, 'trajectory.png'))
            pickle.dump(results, open(os.path.join(testing_dir, 'results.pkl'), 'wb'))
            

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

class DeepReach(Experiment):
    def init_special(self):
        pass