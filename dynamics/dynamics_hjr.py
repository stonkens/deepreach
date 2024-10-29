import hj_reachability as hj
from hj_reachability import sets
import jax.numpy as jnp


class ControlandDisturbanceAffineDynamics(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, tMin, tMax, control_space, disturbance_space):
        self.torch_dynamics = torch_dynamics
        self.tMin = tMin
        self.tMax = tMax
        self.periodic_dims = self.torch_dynamics.periodic_dims
        super().__init__("max", "min", control_space, disturbance_space)


class Dubins3D(ControlandDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, goalR: float, velocity: float, omega_max: float, angle_alpha_factor: float, 
                 set_mode: str, freeze_model: bool, tMin: float = 0.0, tMax: float = 1.0):
        self.torch_dynamics = torch_dynamics
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        control_space = sets.Box(jnp.array([-self.omega_max]), jnp.array([self.omega_max]))
        disturbance_space = sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__(torch_dynamics, tMin, tMax, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, theta = state
        return jnp.array([self.velocity * jnp.cos(theta), self.velocity * jnp.sin(theta), 0.])
    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])
    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])


class Quad2DAttitude(ControlandDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, 
                 max_pos_dist: float = 0.0, max_vel_dist: float = 0.0, tMin: float = 0.0, tMax: float = 1.0):
        self.gravity = gravity
        control_space = sets.Box(jnp.array([-max_angle, min_thrust]), jnp.array([max_angle, max_thrust]))
        disturbance_space = sets.Box(jnp.array([-max_pos_dist, -max_pos_dist, -max_vel_dist, -max_vel_dist]), 
                                     jnp.array([max_pos_dist, max_pos_dist, max_vel_dist, max_vel_dist]))
        super().__init__(torch_dynamics, tMin, tMax, control_space, disturbance_space)

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
    

class Air3D(ControlandDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, collisionR:float, evader_speed:float, pursuer_speed:float, evader_omega_max:float,
                 pursuer_omega_max:float, angle_alpha_factor:float, 
                 tMin:float=0.0, tMax:float=1.0):
        self.collisionR = collisionR
        self.evader_speed = evader_speed
        self.pursuer_speed = pursuer_speed
        self.evader_omega_max = evader_omega_max
        self.pursuer_omega_max = pursuer_omega_max
        self.angle_alpha_factor = angle_alpha_factor
        control_space = sets.Box(jnp.array([-self.evader_omega_max]), jnp.array([self.evader_omega_max]))
        disturbance_space = sets.Box(jnp.array([-self.pursuer_omega_max]), jnp.array([self.pursuer_omega_max]))
        super().__init__(torch_dynamics, tMin, tMax, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, psi = state
        v_a, v_b = self.evader_speed, self.pursuer_speed
        return jnp.array([-v_a + v_b * jnp.cos(psi),
                          v_b * jnp.sin(psi),
                          0.0])
    
    def control_jacobian(self, state, time):
        x, y, psi = state
        return jnp.array([[y], [-x], [-1.0]])
    
    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])

