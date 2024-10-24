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

class Experiment(ABC):
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
        self.validation_metrics = []
        self.visual_only_validation_metrics = []
        # self.emperical_cost_validation_metric = EmpiricalPerformance(self.dataset.dynamics, 0.01, device=self.device)
        self.safety_plot_validation = VisualizeSafeSet2D(self.dataset, validation_dict)
        self.validation_metrics.append(self.safety_plot_validation)
        if self.dataset.dynamics.state_dim <= 5:
            # Generate ground truth for the validation metrics
            # TODO: Move this into GroundTruthHJSolution initialization
            dynamics_class_name = self.dataset.dynamics.__class__.__name__
            dynamics_class = getattr(dynamics_hjr, dynamics_class_name)
            dynamics_params = inspect.signature(dynamics_class).parameters
            dynamics_args = {argname: getattr(self.dataset.dynamics, argname) for argname in dynamics_params 
                             if hasattr(self.dataset.dynamics, argname)}
            dynamics_args['tMin'] = self.dataset.tMin
            dynamics_args['tMax'] = self.dataset.tMax
            hj_dyn = dynamics_class(self.dataset.dynamics, **dynamics_args)
            gt = GroundTruthHJSolution(hj_dyn)
            
            # Visualize 2D safe set for ground truth (for comparison)
            model_eval_gt = lambda coords: gt.value_from_coords(coords)
            log_initial = VisualizeSafeSet2D(self.dataset, validation_dict)(model_eval_gt, None)
            log_initial = {key + '_gt': value for key, value in log_initial.items()}
            log_initial['step'] = 0

            # Rollout of ground truth (large batch size, so no plotting)
            converged_values_validator = ValueThresholdEvaluatorandValidator(eval_fn = gt.value_from_coords, v_min=0.0, v_max=1.0)
            validation_dict['fixed_samples_validator'] = converged_values_validator
            validation_dict['rollout_batch_size'] = 20000
            gt_rollout = FixedRolloutTrajectoriesHJR(self.dataset, validation_dict, gt)
            gt_rollout.vf_times = self.dataset.tMax  # FIXME: Temp
            gt_rollout.rollout_times = self.dataset.tMax  # FIXME: temp
            trajectory_log = gt_rollout(model_eval_gt, None)
            for item, value in trajectory_log.items():
                if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                    log_initial[item + "_gt"] = value

            # Small rollout of ground truth specifically for plotting
            validation_dict['rollout_batch_size'] = 20
            gt_rollout_viz = RolloutTrajectoriesWithVisualsHJR(self.dataset, validation_dict, gt)
            gt_rollout_viz.vf_times = self.dataset.tMax  # FIXME: Temp
            gt_rollout_viz.rollout_times = self.dataset.tMax  # FIXME: temp
            trajectory_viz_log = gt_rollout_viz(model_eval_gt, None)
            for item, value in trajectory_viz_log.items():
                if isinstance(value, wandb.Image):
                    log_initial[item + "_gt"] = value
            
            if self.use_wandb:
                wandb.log(log_initial)

            ########### Validation metrics for all iterations ############    
            self.validation_metrics.append(VisualizeValueDifference2D(self.dataset, validation_dict, gt))
            
            safety_metrics = QuantifyBinarySafetyDifference(self.dataset, validation_dict, gt)
            safety_metrics.eval_states = j2t(gt.grid.states.reshape(-1, gt.grid.ndim))
            self.validation_metrics.append(safety_metrics)
        
        else:
            safety_metrics = QuantifyBinarySafety(self.dataset, validation_dict)
            self.validation_metrics.append(safety_metrics)

        standard_value_validator = ValueThresholdEvaluatorandValidator(eval_fn=self.dataset.dynamics.sdf, v_min=0.0, v_max=2.0)
        validation_dict['fixed_samples_validator'] = standard_value_validator
        validation_dict['rollout_batch_size'] = 20000
        traj_rollout = FixedRolloutTrajectories(self.dataset, validation_dict)

        validation_dict['fixed_samples_validator'] = standard_value_validator
        validation_dict['rollout_batch_size'] = 20
        traj_rollout_viz = RolloutTrajectoriesWithVisuals(self.dataset, validation_dict)

        if self.dataset.dynamics.state_dim <= 5:
            traj_rollout.sampling_states = gt_rollout.sampling_states
            traj_rollout_viz.sampling_states = gt_rollout_viz.sampling_states

        self.validation_metrics.append(traj_rollout)
        self.visual_only_validation_metrics.append(traj_rollout_viz)

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

        if not isinstance(self.validation_metrics, list):
            self.validation_metrics = [self.validation_metrics]
        for validation_metric in self.validation_metrics:
            validation_metric.update_counters()
            log = validation_metric(learned_model_eval, model_eval_grad)
            for key, value in log.items():
                if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                    wandb_log[key] = value
                elif isinstance(value, wandb.Image):
                    wandb_log[key] = value
        
        for validation_metric in self.visual_only_validation_metrics:
            validation_metric.update_counters()
            log = validation_metric(learned_model_eval, model_eval_grad)
            for key, value in log.items():
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
                    self.validate(epoch=epoch+1)

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)
        raise NotImplementedError
        # FIXME: Rewrite this function to use the new validation metrics
        # testing_dir = os.path.join(self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
        # if os.path.exists(testing_dir):
        #     overwrite = input("The testing directory %s already exists. Overwrite? (y/n)"%testing_dir)
        #     if not (overwrite == 'y'):
        #         print('Exiting.')
        #         quit()
        #     shutil.rmtree(testing_dir)
        # os.makedirs(testing_dir)

        # if checkpoint_toload is None:
        #     print('running cross-checkpoint testing')
        #     models = os.path.join(self.experiment_dir, 'training', 'checkpoints')
        #     checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(models) if 'model_epoch' in f]

        #     for i in tqdm(range(len(checkpoints)), desc='Checkpoint'):
        #         self._load_checkpoint(epoch=checkpoints[i])
        #         raise NotImplementedError

        # else:
        #     import matplotlib.pyplot as plt
        #     print('running specific-checkpoint testing')
        #     self._load_checkpoint(checkpoint_toload)
        #     # Get max time of checkpoint from the name (model_epoch_%04d.pth)
        #     if checkpoint_toload == -1:
        #         checkpoint_toload = self.dataset.counter_end
        #     checkpoint_max_time = checkpoint_toload / (self.dataset.tMax - self.dataset.tMin) / self.dataset.counter_end
            
        #     curr_t = max(0, checkpoint_max_time)
        #     if self.dataset.dynamics.state_dim <= 5:
        #     # Get the name of the dynamics class
        #         self.cost_validation = EmpiricalPerformance(self.dataset.dynamics, 0.002, device=self.device,
        #                                                     batch_size=100, fixed_samples=True, 
        #                                                     fixed_samples_validator=self.validation_metrics.alt_method)
            
            
        #         results = self.cost_validation(self.model, vf_times=curr_t-0.1, rollout_times=1.0)
        #         print("Ground truth comparison")
        #         for key, value in results.items():
        #             if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
        #                 print('%s: %f' % (key, value))
                
        #         plt.plot(results['trajectories'][..., 0].T, results['trajectories'][..., 1].T)
        #         plt.plot(results['trajectories'][:,:1, 0].T, results['trajectories'][:,:1, 1].T, '*')
        #         ground_truth = self.validation_metrics.alt_method
        #         plt.contourf(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[-1][:,:,25,25].T)
        #         plt.contour(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[0][:,:,25,25].T, levels=[0], colors='k')
        #         plt.xlim([-5, 5])
        #         plt.ylim([-0.2, 2.8])
        #         plt.savefig(os.path.join(testing_dir, 'trajectory_guaranteed_safe.png'))

        #         fig, ax = plt.subplots()
        #         ax.plot(results['values_over_trajs'][::10].T)
        #         ax.set_ylim([-1, 1])
        #         fig.savefig(os.path.join(testing_dir, 'values_over_trajs.png'))

        #     self.emperical_cost_validation_metric = EmpiricalPerformance(self.dataset.dynamics, 0.001, device=self.device, batch_size=100)
        #     results = self.emperical_cost_validation_metric(self.model, vf_times=[max(0, curr_t - 0.1), curr_t], rollout_times=1.0)
        #     for key, value in results.items():
        #         if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
        #             print('%s: %f' % (key, value))

        #     # Jaccard index calculation
        #     metrics = self.validation_metrics(self.model, add_temporal_data=True)
        #     for key, value in metrics.items():
        #         # if isinstance(value, float) or (isinstance(value, torch.Tensor) and value.numel() == 1):
        #         print('%s: %f' % (key, value))
            
        #     plt.plot(results['trajectories'][..., 0].T, results['trajectories'][..., 1].T)
        #     plt.plot(results['trajectories'][:,:1, 0].T, results['trajectories'][:,:1, 1].T, '*')
        #     ground_truth = self.validation_metrics.alt_method
        #     plt.contourf(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[-1][:,:,25,25].T)
        #     plt.contour(ground_truth.grid.coordinate_vectors[0], ground_truth.grid.coordinate_vectors[1], ground_truth.value_functions[0][:,:,25,25].T, levels=[0], colors='k')
        #     plt.xlim([-5, 5])
        #     plt.ylim([-0.2, 2.8])
        #     plt.savefig(os.path.join(testing_dir, 'trajectory.png'))
        #     pickle.dump(results, open(os.path.join(testing_dir, 'results.pkl'), 'wb'))

        #     val_x_resolution = 200
        #     val_y_resolution = 200
        #     val_z_resolution = 3
        #     val_time_resolution = 5
        #     self.use_wandb = False
        #     self.validate(0, save_path=testing_dir, x_resolution = val_x_resolution, y_resolution = val_y_resolution, 
        #                   z_resolution=val_z_resolution, time_resolution=val_time_resolution)
        # if was_training:
        #     self.model.train()
        #     self.model.requires_grad_(True)

class DeepReach(Experiment):
    def init_special(self):
        pass