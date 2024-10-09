import hj_reachability as hj
from hj_reachability import sets
import jax.numpy as jnp


class Dubins3D(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, goalR: float, velocity: float, omega_max: float, angle_alpha_factor: float, set_mode: str, 
                 freeze_model: bool):
        self.torch_dynamics = torch_dynamics
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        control_space = sets.Box(jnp.array([-self.omega_max]), jnp.array([self.omega_max]))
        disturbance_space = sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__("max", "min", control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, theta = state
        return jnp.array([self.velocity * jnp.cos(theta), self.velocity * jnp.sin(theta), 0.])
    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])
    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])


class Quad2DAttitude(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, 
                 max_pos_dist: float = 0.0, max_vel_dist: float = 0.0):
        self.torch_dynamics = torch_dynamics
        self.gravity = gravity
        control_space = sets.Box(jnp.array([-max_angle, min_thrust]), jnp.array([max_angle, max_thrust]))
        disturbance_space = sets.Box(jnp.array([-max_pos_dist, -max_pos_dist, -max_vel_dist, -max_vel_dist]), 
                                     jnp.array([max_pos_dist, max_pos_dist, max_vel_dist, max_vel_dist]))
        super().__init__("max", "min", control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, vx, vy = state
        return jnp.array([vx, vy, 0., -self.gravity])
    
    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [self.gravity, 0.],
            [0., 1.],
        ])
    
    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    

