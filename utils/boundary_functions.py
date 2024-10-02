import torch
import torch.nn.functional as F
import jax.numpy as jnp


class Obstacle:
    def __init__(self, state_idis, padding, device='cpu') -> None:
        self.state_idis = state_idis
        self.padding = torch.tensor(padding).to(device)

    def to_device(self, device):
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))
        self.device = device


class Circle(Obstacle):
    def __init__(self, state_idis, radius, center, padding=0.0, device='cpu') -> None:
        super().__init__(state_idis, padding, device=device)
        self.radius = torch.tensor(radius).to(device)
        self.center = torch.tensor(center).to(device)[torch.newaxis]

    def obstacle_sdf(self, x):
        self.to_device(x.device)
        obstacle_sdf = torch.norm(self.center - x[..., self.state_idis], dim=-1) - self.radius - self.padding
        return obstacle_sdf

class Rectangle(Obstacle):
    def __init__(self, state_idis, min_val, max_val, padding=0.0, device='cpu') -> None:
        super().__init__(state_idis, padding, device=device)
        self.min_val = torch.tensor(min_val).to(device)[torch.newaxis]
        self.max_val = torch.tensor(max_val).to(device)[torch.newaxis]

    def obstacle_sdf(self, x):
        self.to_device(x.device)
        max_dist_per_dim = torch.max(
            torch.stack([self.min_val - x[..., self.state_idis], x[..., self.state_idis] - self.max_val]), dim=0
        ).values
        outside_obstacle = torch.norm(torch.clamp(max_dist_per_dim, min=0), dim=-1)
        inside_obstacle = torch.max(max_dist_per_dim, dim=-1).values
        obstacle_sdf = (torch.where(torch.all(max_dist_per_dim < 0.0, dim=-1), inside_obstacle, outside_obstacle) 
                        - self.padding)
        return obstacle_sdf

class Boundary(Obstacle):
    def __init__(self, state_idis, min_val, max_val, padding=0.0, device='cpu') -> None:
        super().__init__(state_idis, padding, device=device)
        self.min_val = torch.tensor(min_val).to(device)[torch.newaxis]
        self.max_val = torch.tensor(max_val).to(device)[torch.newaxis]

    def boundary_sdf(self, x):
        # All attributes should move to the device
        self.to_device(x.device)
        max_dist_per_dim = torch.max(
            torch.stack([self.min_val - x[..., self.state_idis], x[..., self.state_idis] - self.max_val]), dim=0
        ).values
        outside_boundary = -torch.norm(torch.clamp(max_dist_per_dim, min=0), dim=-1)
        inside_boundary = -torch.max(max_dist_per_dim, dim=-1).values
        obstacle_sdf = (torch.where(torch.all(max_dist_per_dim < 0.0, dim=-1), inside_boundary, outside_boundary) 
                        - self.padding)
        return obstacle_sdf

def build_sdf(boundary, obstacles):
    def sdf(x):
        sdf_val = boundary.boundary_sdf(x)
        for obstacle in obstacles:
            obstacle_sdf = obstacle.obstacle_sdf(x)
            sdf_val = torch.minimum(sdf_val, obstacle_sdf)
        return sdf_val
    return sdf
    

def build_sdf(boundary, obstacles):
    def sdf(x):
        sdf_val = boundary.boundary_sdf(x)
        for obstacle in obstacles:
            obstacle_sdf = obstacle.obstacle_sdf(x)
            if isinstance(x, torch.Tensor):
                sdf_val = torch.minimum(sdf_val, obstacle_sdf)
            elif isinstance(x, jnp.ndarray):
                sdf_val = jnp.minimum(sdf_val, obstacle_sdf)
        return sdf_val
    return sdf

class ObstacleJAX:
    def __init__(self, state_idis, padding):
        self.state_idis = state_idis
        self.padding = padding

class CircleJAX(ObstacleJAX):
    def __init__(self, state_idis, radius, center, padding=0.0) -> None:
        super().__init__(state_idis, padding)
        self.radius = radius
        self.center = center[jnp.newaxis]

    def obstacle_sdf(self, x):
        obstacle_sdf = jnp.linalg.norm(self.center - x[..., self.state_idis], axis=-1) - self.radius - self.padding
        return obstacle_sdf

class RectangleJAX(ObstacleJAX):
    def __init__(self, state_idis, min_val, max_val, padding=0.0) -> None:
        super().__init__(state_idis, padding)
        self.min_val = min_val[jnp.newaxis]
        self.max_val = max_val[jnp.newaxis]

    def obstacle_sdf(self, x):
        max_dist_per_dim = jnp.max(
            jnp.stack([self.min_val - x[..., self.state_idis], x[..., self.state_idis] - self.max_val]), axis=0
        )
        outside_obstacle = jnp.linalg.norm(jnp.clip(max_dist_per_dim, a_min=0), axis=-1)
        inside_obstacle = jnp.max(max_dist_per_dim, axis=-1)
        obstacle_sdf = (jnp.where(jnp.all(max_dist_per_dim < 0.0, axis=-1), inside_obstacle, outside_obstacle) 
                        - self.padding)
        return obstacle_sdf

class BoundaryJAX(ObstacleJAX):
    def __init__(self, state_idis, min_val, max_val, padding=0.0) -> None:
        super().__init__(state_idis, padding)
        self.min_val = min_val[jnp.newaxis]
        self.max_val = max_val[jnp.newaxis]

    def boundary_sdf(self, x):
        max_dist_per_dim = jnp.max(
            jnp.stack([self.min_val - x[..., self.state_idis], x[..., self.state_idis] - self.max_val]), axis=0
        )
        outside_boundary = -jnp.linalg.norm(jnp.clip(max_dist_per_dim, a_min=0), axis=-1)
        inside_boundary = -jnp.max(max_dist_per_dim, axis=-1)
        obstacle_sdf = (jnp.where(jnp.all(max_dist_per_dim < 0.0, axis=-1), inside_boundary, outside_boundary) 
                        - self.padding)
        return obstacle_sdf


def build_sdf_jax(boundary, obstacles):
    def sdf(x):
        sdf_val = boundary.boundary_sdf(x)
        for obstacle in obstacles:
            obstacle_sdf = obstacle.obstacle_sdf(x)
            sdf_val = jnp.minimum(sdf_val, obstacle_sdf)
        return sdf_val
    return sdf
