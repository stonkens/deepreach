import hj_reachability as hj
from hj_reachability import sets
import jax.numpy as jnp


class Dubins3D(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, goalR: float, velocity: float, omega_max: float, angle_alpha_factor: float, 
                 set_mode: str, freeze_model: bool, tMin: float = 0.0, tMax: float = 1.0):
        self.torch_dynamics = torch_dynamics
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        self.tMin = tMin
        self.tMax = tMax
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
                 max_pos_dist: float = 0.0, max_vel_dist: float = 0.0, tMin: float = 0.0, tMax: float = 1.0):
        self.torch_dynamics = torch_dynamics
        self.gravity = gravity
        self.tMin = tMin
        self.tMax = tMax
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
 #################### Nikhil: Additional Dynamics ####################

class Air3D(hj.dynamics.ControlAndDisturbanceAffineDynamics): 
    def __init__(self, torch_dynamics, collisionR:float, velocity:float, omega_max:float, angle_alpha_factor:float, 
                tMin: float = 0.0, tMax: float = 1.0):
        self.torch_dynamics = torch_dynamics
        self.collisionR = collisionR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.tMin = tMin
        self.tMax = tMax
        control_space = sets.Box(jnp.array([-omega_max]), jnp.array([omega_max]))
        disturbance_space = sets.Box(jnp.array([-omega_max]), jnp.array([omega_max])) # NOTE: maybe change them to be separate ? 
        super().__init__("max", "min", control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, psi = state 
        return jnp.array([-self.velocity + self.velocity * jnp.cos(psi), 
                          self.velocity * jnp.sin(psi), 
                          0.])        

    def control_jacobian(self, state, time):
        if len(state.shape) == 1:
            x, y, _ = state
            return jnp.array([
                [y],
                [-x],
                [-1.],
            ])
        else: # handle (batch_size, state_dim) case
            return_val =  jnp.array([
                state[..., 1],
                -state[..., 0],
                jnp.array([-1.] * state.shape[0]), 
            ]).T
            return return_val

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """
        Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian.
        Customized to handle state dependent control jacobians
        """
        if len(state.shape) == 1:
            control_direction = grad_value @ self.control_jacobian(state, time)
        else: 
            # Handle (batch_size, state_dim) case - for state dependent control jacobian 
            c_jacobian = self.control_jacobian(state, time)
            control_direction = jnp.einsum("ij,ij->i", grad_value, c_jacobian).reshape(-1, 1)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        return (self.control_space.extreme_point(control_direction),
                self.disturbance_space.extreme_point(disturbance_direction))

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], 
                          [0.0], 
                          [1.0]])


class Drone4D(hj.dynamics.Dynamics):
    def __init__(self, torch_dynamics, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, 
                 max_pos_y_dist: float, max_pos_z_dist: float, max_vel_y_dist: float, max_vel_z_dist: float,
                 tMin: float = 0.0, tMax: float = 1.0):

        self.torch_dynamics = torch_dynamics
        self.gravity = gravity 

        raise NotImplementedError

    def __call__(self, state, control, disturbance, time):
        """Implements the continuous-time dynamics ODE: outputs dx_dt = f(x, u, d, t)."""
        dx_dt = jnp.zeros_like(state)
        dx_dt[..., 0] = state[..., 2] + disturbance[..., 0]
        dx_dt[..., 1] = state[..., 3] + disturbance[..., 1]
        dx_dt[..., 2] = control[..., 1]*jnp.sin(control[..., 0]) + disturbance[..., 2]
        dx_dt[..., 3] = control[..., 1]*jnp.cos(control[..., 0]) - self.gravity + disturbance[..., 3]
        return dx_dt


    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        p1 = grad_value[..., 0]  # partial w.r.t. x1
        p2 = grad_value[..., 1]  # partial w.r.t. x2
        p3 = grad_value[..., 2]  # partial w.r.t. x3
        p4 = grad_value[..., 3]  # partial w.r.t. x4

        u1 = jnp.ones(p3.shape) 
        u2 = jnp.ones(p4.shape)

        # Optimal Control 
        # Maximize with u1 and u2 
        arctan_p3p4 = jnp.atan(p3/(p4 + jnp.finfo(jnp.float32).eps))

        # Case 1: p3 > 0, p4 > 0
        u1 = jnp.where((p3 > 0) & (p4 > 0), jnp.minimum(arctan_p3p4, self.u1_max), u1)

        # Case 2: p3 > 0, p4 < 0
        u1 = jnp.where((p3 > 0) & (p4 < 0), self.u1_max, u1)

        # Case 3: p3 < 0, p4 > 0
        u1 = jnp.where((p3 < 0) & (p4 > 0), jnp.maximum(arctan_p3p4, self.u1_min), u1)

        # Case 4: p3 < 0, p4 < 0
        u1 = jnp.where((p3 < 0) & (p4 < 0), self.u1_min, u1)

        # u2: select u2 max if g(u1) > - else select u2 min 
        g_u1 = p3*jnp.sin(u1) + p4*jnp.cos(u1)
        u2 = jnp.where(g_u1 < 0, self.u2_min, self.u2_max)

        # Optimal Disturbance 
        d1 = jnp.where(p1 < 0, self.d1_max, self.d1_min)
        d2 = jnp.where(p2 < 0, self.d2_max, self.d2_min)
        d3 = jnp.where(p3 < 0, self.d3_max, self.d3_min)
        d4 = jnp.where(p4 < 0, self.d4_max, self.d4_min)

        opt_control = jnp.stack([u1, u2], axis=-1)
        opt_disturbance = jnp.stack([d1, d2, d3, d4], axis=-1)

        return (opt_control, opt_disturbance)
        

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""

        raise NotImplementedError

    # Don't need to re-define the below functions 
    # def optimal_control(self, state, time, grad_value):
    #     """Computes the optimal control realized by the HJ PDE Hamiltonian."""
    #     return self.optimal_control_and_disturbance(state, time, grad_value)[0]
    #
    # def optimal_disturbance(self, state, time, grad_value):
    #     """Computes the optimal disturbance realized by the HJ PDE Hamiltonian."""
    #     return self.optimal_control_and_disturbance(state, time, grad_value)[1]
    #
    # def hamiltonian(self, state, time, value, grad_value):
    #     """Evaluates the HJ PDE Hamiltonian."""
    #     del value  # unused
    #     control, disturbance = self.optimal_control_and_disturbance(state, time, grad_value)
    #     return grad_value @ self(state, control, disturbance, time)


#################### Nikhil: Parametric Dynamics ####################

class Quad2DAttitude_parametric(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, 
                 max_pos_dist: float = 0.0, max_vel_dist: float = 0.0, tMin: float = 0.0, tMax: float = 1.0):
        self.torch_dynamics = torch_dynamics
        self.gravity = gravity
        self.tMin = tMin
        self.tMax = tMax
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