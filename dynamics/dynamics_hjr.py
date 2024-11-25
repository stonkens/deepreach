import hj_reachability as hj
from hj_reachability import sets
import jax.numpy as jnp


class ControlandDisturbanceAffineDynamics(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, tMin, tMax, control_space, disturbance_space):
        self.torch_dynamics = torch_dynamics
        self.tMin = tMin
        self.tMax = tMax
        self.state_dim = self.torch_dynamics.state_dim
        self.control_dim = self.torch_dynamics.control_dim
        self.disturbance_dim = self.torch_dynamics.disturbance_dim
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


class Quad2DAttitude_parametric(ControlandDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, gravity: float, max_angle: float, min_thrust: float, max_thrust: float, 
                 max_pos_dist: float = 0.0, max_vel_dist: float = 0.0, tMin: float = 0.0, tMax: float = 1.0):
        self.torch_dynamics = torch_dynamics
        self.gravity = gravity
        self.tMin = tMin
        self.tMax = tMax
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


class InvertedPendulum(ControlandDisturbanceAffineDynamics): 
    # Following dynamics from: https://arxiv.org/pdf/2206.03568 
    # NOTE: want to follow dynamics from: https://arxiv.org/pdf/1903.08792 

    def __init__(self, torch_dynamics, gravity: float, length: float, mass: float, 
                 unsafe_theta_min: float, unsafe_theta_max: float, min_torque: float, max_torque: float, 
                 max_theta_dist: float, max_thetadot_dist: float, 
                 damping: float=0.0, 
                 tMin: float=0.0, tMax: float=1.0):
        """
        args: 
        - gravity: default 9.8
        - length: default 1.0
        - mass: default 1.0 
        - unsafe_theta_min: angle in radians describing start of unsafe region
        - unsafe_theta_max: angle in radians describing end of unsafe region 
        - min_torque: min input torque
        - max_torque: max input torque 
        - max_theta_dist: max  disturbance on positional angle in radians 
        - max_thetadot_dist: max disturbance on angular velocity in radians 
        - damping: spring damping on the hinge 
        - tMin
        - tMax 
        """    
        self.torch_dynamics = torch_dynamics
        self.gravity = gravity 
        self.tMin = tMin 
        self.tMax = tMax 

        # pendulum parameters
        self.gravity = gravity 
        self.length = length 
        self.mass = mass 

        # control and disturbance parameters 
        self.min_torque = min_torque 
        self.max_torque = max_torque 
        self.max_theta_dist = max_theta_dist 
        self.max_thetadot_dist = max_thetadot_dist

        self.damping = damping 
        
        control_space = sets.Box(jnp.array([min_torque]), jnp.array([max_torque]))
        disturbance_space = sets.Box(jnp.array([-max_theta_dist, -max_thetadot_dist]), 
                                     jnp.array([ max_theta_dist,  max_thetadot_dist]))
        
        super().__init__(torch_dynamics, tMin, tMax, control_space, disturbance_space)

   
    # Dynamics 
    # d theta  = thetadot
    # d thetadot = (-damping*theta_dot - m*g*l*sin(theta) + dt)/ml^2 + 1/ml^2 u # NOTE: assumes gravity is positive
    def open_loop_dynamics(self, state, time):
        theta, thetadot = state
        return jnp.array([
            thetadot, 
            (-self.damping * thetadot + self.mass * self.gravity * self.length * jnp.sin(theta)) / (self.mass * self.length ** 2)
        ])
    
    def control_jacobian(self, state, time): 
        return jnp.array([
            [0.0], 
            [1/(self.mass * self.length**2)]
        ])
    
    def disturbance_jacobian(self, state, time):
        return jnp.array([[1.0, 0.0], [0.0, 1.0]])
    

class CartPole(ControlandDisturbanceAffineDynamics):
    def __init__(self, torch_dynamics, gravity: float, umax: float, length: float, mass_cart: float, mass_pole: float,
                 unsafe_x_min: float, unsafe_x_max: float, unsafe_vel_max: float, unsafe_theta_min: float, unsafe_theta_max: float, # unsafe bounds
                 x_dist: float, theta_dist: float, vel_dist: float, thetadot_dist: float, # disturbance bound parameters
                 tMin: float=0.0, tMax: float=1.0,
                 unsafe_theta_in_range: float=True,
                 control_mode="max", disturbance_mode="min", control_space=None, disturbance_space=None):
        
        self.torch_dynamics = torch_dynamics 
        self.tMin = tMin 
        self.tMax = tMax 

        self.umax = umax # 10.0
        self.length = length # 0.5  # length of the pole
        self.m_c = mass_cart # 1.0  # mass of cart
        self.m = mass_pole # 0.1 # mass of pole
        self.gravity = gravity # -9.8

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-self.umax]), jnp.array([self.umax]))
        if disturbance_space is None:
            # disturbance_space = hj.sets.Box(jnp.array([0.0, 0.0, -0.2, -0.2]), jnp.array([0.0, 0.0, 0.2, 0.2]))
            disturbance_space = hj.sets.Box(jnp.array([-x_dist, -theta_dist, -vel_dist, -thetadot_dist]), jnp.array([x_dist, theta_dist, vel_dist, thetadot_dist]))
        # super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)
        super().__init__(torch_dynamics, tMin, tMax, control_space, disturbance_space)

    # # NOTE: old one - might be the correct one 
    # def open_loop_dynamics(self, state, time):
    #     x, theta, xdot, thetadot = state
    #     return jnp.array([
    #         xdot,
    #         thetadot,
    #         self.m * (self.length * thetadot ** 2 + self.gravity * jnp.cos(theta)) * jnp.sin(theta) / (self.m_c + self.m * jnp.sin(theta) ** 2),
    #         - ((self.m + self.m_c) * self.gravity * jnp.sin(theta) + self.m * self.length * thetadot ** 2 * jnp.sin(theta) * jnp.cos(theta)) / (self.length * (self.m_c + self.m * jnp.sin(theta) ** 2))
    #     ])

    # def control_jacobian(self, state, time):
    #     return jnp.array([[0.0], 
    #                       [0.0], 
    #                       [1.0 / (self.m_c + self.m * jnp.sin(state[1]) ** 2)], 
    #                       [jnp.cos(state[1]) / (self.length * (self.m_c + self.m * jnp.sin(state[1]) ** 2))]
    #                       ])


    # Dynamics from: https://courses.ece.ucsb.edu/ECE594/594D_W10Byl/hw/cartpole_eom.pdf 
    # NOTE: new one - theta goes clockwise however ISSUE: negative thetadot causes positive increase in theta
    def open_loop_dynamics(self, state, time):
        x, theta, xdot, thetadot = state
        # Multiply by -1 to flip theta conventions: now theta increases clockwise
        theta = -theta 
        thetadot = -thetadot
        return jnp.array([
            xdot,
            -1 * thetadot,
            -(self.m * self.length * jnp.sin(theta) * thetadot**2) + (self.m * self.gravity * jnp.sin(theta) * jnp.cos(theta)) / (self.m_c + self.m*jnp.sin(theta)**2),
            -1 * (-(self.m * self.length * jnp.cos(theta) * jnp.sin(theta) * thetadot**2) + ((self.m_c + self.m)*self.gravity*jnp.sin(theta))  / (self.length * (self.m_c + self.m * jnp.sin(theta) ** 2)))
        ])
    
    def control_jacobian(self, state, time):
        # Multiply by -1 to flip theta conventions: now theta increases clockwise
        x, theta, xdot, thetadot = state 
        theta = -theta 
        thetadot = -thetadot
        return jnp.array([[0.0], 
                          [-1 * 0.0], 
                          [1.0 / (self.m_c + self.m * jnp.sin(state[1]) ** 2)], 
                          [-1 *jnp.cos(state[1]) / (self.length * (self.m_c + self.m * jnp.sin(state[1]) ** 2))]
                          ])
    
    def disturbance_jacobian(self, state, time):
        return jnp.eye(4)