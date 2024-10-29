import torch


# uses real units
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brt_hjivi_loss(state, time, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # only occurs when pretraining (or if all times = 0)
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, time, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value)
        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                # only occurs when pretraining (or if all times = 0)
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss


def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brat_hjivi_loss(state, time, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # only occurs when pretraining (or if all times = 0)
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, time, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
    return brat_hjivi_loss
