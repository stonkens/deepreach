import numpy as np 
import torch 

try: 
    from deepreach.utils import boundary_functions
except: 
    from utils import boundary_functions


class Quad2DAttitude_envs():
    def __init__(self, config_num, problem_type):
        """""
        Args: 
            - config_num: int: configuration number for the environment 
            - problem_type: str: problem type to generate sdfs ["reach", "avoid", "reach_avoid", "reach_avoid_ci"]
        """
        try: 
            from deepreach.utils import boundary_functions
        except: 
            from utils import boundary_functions
            
        viable_obstacle_configs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        viable_problem_types = ["reach", "avoid", "reach_avoid", "reach_avoid_ci"]

        self.config_num = config_num
        self.problem_type = problem_type

        assert config_num in viable_obstacle_configs, "Invalid configuration number for the Quad2DAttitude environment."
        assert problem_type in viable_problem_types, "Invalid problem type for the Quad2DAttitude environment."
        
        # Sign Conventions
        # SDF Avoid: Negative = Unsafe, Positive = Safe
        # SDF Reach: Negative = Outside Reach/Unsafe, Positive = Inside Reach/Safe

        if self.config_num == 1: 
            # Obstacle Config: 1
            # Avoid Config 
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            circle = boundary_functions.Circle([0, 1], 0.5, torch.Tensor([2.0, 1.5]))
            rectangle = boundary_functions.Rectangle([0, 1], torch.Tensor([-2.0, 0.5]), torch.Tensor([0.0, 1.5]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [circle, rectangle])
            # Reach Config 
            ellipse = boundary_functions.Ellipse([0, 1, 2, 3], 1.0, [0.75, 1.0, 0.0, 0.0], [2.0, 1.0, 3.0, 3.0])
            self.sdf_reach = ellipse.boundary_sdf #lambda x: -1 * ellipse.boundary_sdf(x)


        elif self.config_num == 2: 
            # Obstacle Config: 2
            # Avoid Config
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            circle = boundary_functions.Circle([0, 1], 0.5, torch.Tensor([2.0, 1.5]))
            rectangle = boundary_functions.Rectangle([0, 1], torch.Tensor([-2.0, 0.0]), torch.Tensor([0.0, 1.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [circle, rectangle])
            # Reach Config 
            circle = boundary_functions.Circle([0, 1], 0.5, torch.tensor([-3.0, 1.75]))
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x) #circle.obstacle_sdf
        
        elif self.config_num == 3:
            # Obstacle Config: 3
            # Avoid Config
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [])
            circle = boundary_functions.Circle([0, 1, 2, 3], 0.5, torch.Tensor([0.0, 1.25, 0.0, 0.0]))
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x) #circle.obstacle_sdf

        elif self.config_num == 4:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            circle = boundary_functions.Circle([0, 1], 0.6, torch.Tensor([0.0, 1.75]))
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)
        elif self.config_num == 5:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            ellipse = boundary_functions.Ellipse([0, 1, 2, 3], 0.75, [0.0, 2.1, 0.0, 0.0], [0.2, 5.0, 3.0, 3.0], 
                                                 slope_change='outside', slope_change_type='ln')
            self.sdf_reach = ellipse.boundary_sdf #lambda x: -1 * ellipse.boundary_sdf(x)
        elif self.config_num == 6:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            ellipse = boundary_functions.Ellipse([0, 1, 2, 3], 0.75, [0.0, 2.1, 0.0, 0.0], [0.2, 5.0, 3.0, 3.0])
            self.sdf_reach = ellipse.boundary_sdf #lambda x: -1 * ellipse.boundary_sdf(x)
        elif self.config_num == 7:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            ellipse = boundary_functions.Ellipse([0, 1, 2, 3], 0.75, [0.0, 2.1, 0.0, 0.0], [0.2, 5.0, 3.0, 3.0],
                                                 min_val=-2.0)
            self.sdf_reach = ellipse.boundary_sdf #lambda x: -1 * ellipse.boundary_sdf(x)
        elif self.config_num == 8:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            circle = boundary_functions.Circle([0, 1, 2, 3], 0.6, torch.Tensor([0.0, 1.75, 0.0, 0.0]))
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)   
                    
        elif self.config_num == 9:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            ellipse = boundary_functions.TanhEllipse([0, 1, 2, 3], 0.75, [-0.5, 2.0, 0.0, 0.0], [0.2, 5.0, 3.0, 3.0])
            self.sdf_reach = ellipse.obstacle_sdf  #lambda x: -1 * circle.obstacle_sdf(x)               
        elif self.config_num == 10:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))     
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            ellipse = boundary_functions.EllipseNorm([0, 1], 0.6, [0.0, 1.75], [1.0, 1.0])
            self.sdf_reach = ellipse.obstacle_sdf  #lambda x: -1 * circle.obstacle_sdf(x)              
        elif self.config_num == 11:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            ellipse = boundary_functions.EllipseNorm([0, 1, 2, 3], 0.5, [-0.5, 2.2, 0.0, 0.0], [0.2, 2.0, 1.0, 1.0])
            self.sdf_reach = ellipse.obstacle_sdf  #lambda x: -1 * circle.obstacle_sdf(x)
        elif self.config_num == 12:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            ellipse = boundary_functions.EllipseNorm([0, 1, 2, 3], 0.5, [-0.5, 2.2, 0.0, 0.0], [0.2, 2.0, 1.0, 1.0])
            self.sdf_reach = lambda x: 2 * ellipse.obstacle_sdf(x)  #lambda x: -1 * circle.obstacle_sdf(x)  
        elif self.config_num == 13:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            # Reach Config
            ellipse = boundary_functions.EllipseNormFixed([0, 1, 2, 3], 0.6, [-0.5, 2.2, 0.0, 0.0], [0.1, 5.0, 0.5, 0.5])
            self.sdf_reach = lambda x: ellipse.obstacle_sdf(x) / 2.0  # sqrt(4)->2  #lambda x: -1 * circle.obstacle_sdf(x)  
        elif self.config_num == 14:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            ellipse = boundary_functions.EllipseNormFixed([0, 1, 2, 3], 0.6, [0.0, 1.75, 0.0, 0.0], [1.0, 1.0, 0.2, 0.2])
            self.sdf_reach = lambda x: ellipse.obstacle_sdf(x) # Didn't add the sqrt(4) factor here

        elif self.config_num == 15:
            space_boundary = boundary_functions.Boundary([0, 1, 2, 3], torch.Tensor([-4.0, 0.0, -1.9, -1.9]),
                                                            torch.Tensor([4.0, 2.5, 1.9, 1.9]))
            # rectangle1 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.5, 0.0]), torch.Tensor([-2.9, 1.5]))
            rectangle2 = boundary_functions.Rectangle([0, 1], torch.Tensor([-3.1, 0.0]), torch.Tensor([-1.3, 1.5]))
            rectangle3 = boundary_functions.Rectangle([0, 1], torch.Tensor([0.0, 0.0]), torch.Tensor([1.2, 1.0]))
            rectangle4 = boundary_functions.Rectangle([0, 1], torch.Tensor([2.0, 0.0]), torch.Tensor([3.2, 2.0]))
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [rectangle2, rectangle3, rectangle4])
            rectangle = boundary_functions.Rectangle([0, 1, 2, 3], torch.Tensor([-2.5, 1.7, -1.0, -1.0]), torch.Tensor([1.5, 2.3, 1.0, 1.0]))
            self.sdf_reach = lambda x: -1 * rectangle.obstacle_sdf(x) # Didn't add the sqrt(4) factor here
        self.configure_reach_avoid_fns()
        return 

    def configure_reach_avoid_fns(self, ): 
        """
        Function to properly change the sign of the reach and avoid functions so that they align
        with deepreach conventions according to the problem we want to solve

        NOTE: Starting with standard convention: 
            - Avoid: Negative = Unsafe, Positive = Safe
            - Reach: Negative = Outside Reach/Unsafe, Positive = Inside Reach/Safe

        Returns: None - function adjusts the parameters: 
            - sdf_reach
            - sdf_avoid

        For convention reference see: https://www.notion.so/Deepreach-Standard-Conventions-1a82da3e70b280589e4bfcee23e398a4  
        """
        
        if self.problem_type == "avoid": 
            # Negative Unsafe, Positive Safe
            self.avoid_fn = self.sdf_avoid 
            self.reach_fn = None 
            self.boundary_fn = self.sdf_avoid 
        elif self.problem_type == "reach": 
            # WRONG: Negative Outside Reach/Unsafe, Positive Inside Reach/Safe
            # Positive Unsafe/Outside reach set, Negative Safe/Inside reach set
            self.avoid_fn = None 
            self.reach_fn = lambda x: -1 * self.sdf_reach(x) #lambda x: -1 * self.sdf_reach(x) 
            self.boundary_fn = lambda x: -1 * self.sdf_reach(x)
        elif self.problem_type == "reach_avoid" or self.problem_type == "reach_avoid_ci": 
            # Avoid: Negative Unsafe, Positive Safe
            self.avoid_fn = self.sdf_avoid
            # Reach: Positive Unsafe/Outside reach set, Negative Safe/Inside reach set
            self.reach_fn = lambda x: -1 * self.sdf_reach(x)
            # Boundary Function: Positive Unsafe, Negative Safe
            self.boundary_fn = lambda x: torch.maximum(self.reach_fn(x), -self.avoid_fn(x))
        else: 
            raise ValueError("Invalid problem type for the Quad2DAttitude environment.")
    
    # def avoid_fn(self, x):
    #     return self.sdf_avoid(x)
    
    # def reach_fn(self, x): 
    #     return self.sdf_reach(x)

    # def boundary_fn(self, x):  
    #     return self.boundary_function(x)


# TODO: NOTE: Need to consolidate this with the above environment configurations - have the actual initialization 
# be localized to a function call or something 

########################################################### Quad 10d ###########################################################

class Quad10d_envs(): 
    """
    Environment Configurations for 10 d quadcopter
    """

    def __init__(self, config_num, problem_type):
        """
        Args: 
            - config_num: int: configuration number for the environment 
            - problem_type: str: problem type to generate sdfs ["reach", "avoid", "reach_avoid", "reach_avoid_ci"]
        """
        viable_obstacle_configs = [1,3,4, 5, 6]
        viable_problem_types = ["reach", "avoid", "reach_avoid", "reach_avoid_ci"]

        self.config_num = config_num 
        self.problem_type = problem_type 

        assert config_num in viable_obstacle_configs, "Invalid configuration number for the Quad10d environment."
        assert problem_type in viable_problem_types, "Invalid problem type for the Quad10d environment."

        # Sign Conventions: 
        # SDF Avoid: Negative = Unsafe, Positive = Safe
        # SDF Reach: Negative = Outside Reach/Unsafe, Positive = Inside Reach/Safe

        state_slices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        state_labels = ['x', 'v_x', r'$\theta_x$', r'$\omega_x$', 'y', 'v_y', r'$\theta_y$', r'$\omega_y$', 'z', 'v_z']

        # State: [0,  1 ,   2    ,    3   , 4,  5 ,   6    ,    7   , 8,  9]
        # State: [x, v_x, theta_x, omega_x, y, v_y, theta_y, omega_y, z, v_z]
        if self.config_num == 1:
            # Obstacle Config: 1
            # Avoid Config 
            space_boundary = boundary_functions.Boundary(
                state_idis=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                min_val=[-4.0, -1.9, -(np.pi/4  - np.pi/16), -(np.pi  - np.pi/16), -1.9, -1.9, -(np.pi/4  - np.pi/16), -(np.pi  - np.pi/16), 0, -1.9], 
                max_val=[4.0, 1.9, (np.pi/4  - np.pi/16), (np.pi  - np.pi/16), 1.9, 1.9, (np.pi/4  - np.pi/16), (np.pi  - np.pi/16), 2.5, 1.9]
                # min_val=[-4.0, -1.9, -(np.pi/2  - np.pi/8), -(np.pi/2  - np.pi/8), -2.5, -1.9, -(np.pi/2  - np.pi/8), -(np.pi/2  - np.pi/8), 0, -1.9], 
                # max_val=[4.0, 1.9, (np.pi/2  - np.pi/8), (np.pi/2  - np.pi/8), 2.5, 1.9, (np.pi/2  - np.pi/8), (np.pi/2  - np.pi/8), 2.5, 1.9]
                )
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8],
                radius=0.5,  
                center=torch.Tensor([2.0, 0.0, 1.5])
            )
            rectangle = boundary_functions.Rectangle(
                state_idis=[0, 4, 8], 
                min_val=torch.Tensor([-2.0, -2.0, 0.0]), 
                max_val=torch.Tensor([0.0, 0.0, 1.0])
            )
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [circle, rectangle])

            # For 3d visualization: only obstacles no boundary
            self.visualize_avoid_obstacle_sdf = lambda x: torch.min([obstacle_sdf(x) for obstacle_sdf in [circle.obstacle_sdf, rectangle.obstacle_sdf]])
            
            # Reach Config
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.75])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)
            # NOTE: Combine with the above class and just have different init functions for the different dynamics classes or something

            # Define Plot config: 
            state_slices[4] = -0.1 # adjust y
            self.plot_config = {
                'state_slices': state_slices, 
                'state_labels': state_labels, 
                'x_axis_idx': 0, # x-axis index
                'y_axis_idx': 8, # z-axis index
                'z_axis_idx': [1, 9], # vx, vz
            }

        elif self.config_num == 3:
            # Obstacle Config: 3
            # Avoid Config 
            space_boundary = boundary_functions.Boundary(
                state_idis=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                min_val=[-4.0, -1.9, -(np.pi/4  - np.pi/16), -(np.pi  - np.pi/16), -1.9, -1.9, -(np.pi/4  - np.pi/16), -(np.pi  - np.pi/16), 0, -1.9], 
                max_val=[4.0, 1.9, (np.pi/4  - np.pi/16), (np.pi  - np.pi/16), 1.9, 1.9, (np.pi/4  - np.pi/16), (np.pi  - np.pi/16), 2.5, 1.9]
                # min_val=[-4.0, -1.9, -(np.pi/2  - np.pi/8), -(np.pi/2  - np.pi/8), -2.5, -1.9, -(np.pi/2  - np.pi/8), -(np.pi/2  - np.pi/8), 0, -1.9], 
                # max_val=[4.0, 1.9, (np.pi/2  - np.pi/8), (np.pi/2  - np.pi/8), 2.5, 1.9, (np.pi/2  - np.pi/8), (np.pi/2  - np.pi/8), 2.5, 1.9]
                )
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8],
                radius=0.5,  
                center=torch.Tensor([2.5, 0.0, 1.5])
            )
            rectangle = boundary_functions.Rectangle(
                state_idis=[0, 4, 8], 
                min_val=torch.Tensor([-1.5, -1.5, 0.0]), 
                max_val=torch.Tensor([0.5, 0.5, 1.0])
            )
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [circle, rectangle])

            # For 3d visualization: only obstacles no boundary
            self.visualize_avoid_obstacle_sdf = lambda x: torch.min(torch.tensor([obstacle_sdf(x) for obstacle_sdf in [circle.obstacle_sdf, rectangle.obstacle_sdf]]))

            # Reach Config
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.75])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)
            
            # Define Plot config: 
            self.plot_config = {
                'state_slices': state_slices, 
                'state_labels': state_labels, 
                'x_axis_idx': 0, # x-axis index
                'y_axis_idx': 8, # z-axis index
                'z_axis_idx': [1, 9], # vx, vz
            }

        elif self.config_num == 4: 
            """
            Description: Single Cylindrical obstacle in environment center
            """
            # Cylindrical obstacle 
            self.sdf_avoid = lambda state: torch.norm(state[..., [0,4]], dim=-1) - 0.5 

            # For 3d visualization: only obstacles no boundary
            self.visualize_avoid_obstacle_sdf = self.sdf_avoid

            # Reach Config 
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.75])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)

            # Define Plot config: 
            state_slices[8] = 0.5 # adjust z slice
            self.plot_config = {
                'state_slices': state_slices,
                'state_labels': state_labels,
                'x_axis_idx': 0, # x axis 
                'y_axis_idx': 4, # y axis
                'z_axis_idx': [1, 5], # vx and vy
            }
        elif self.config_num == 5:
            """
            Description: Single Cylindrical obstacle in environment center with space boundary
            """
            # Space Boundary
            space_boundary = boundary_functions.Boundary(
                state_idis=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                min_val=[-4.0, -1.9, -(np.pi/4  - np.pi/16), -(np.pi  - np.pi/16), -1.9, -1.9, -(np.pi/4  - np.pi/16), -(np.pi  - np.pi/16), 0, -1.9], 
                max_val=[4.0, 1.9, (np.pi/4  - np.pi/16), (np.pi  - np.pi/16), 1.9, 1.9, (np.pi/4  - np.pi/16), (np.pi  - np.pi/16), 2.5, 1.9]
                )
            # Cylindrical Obstacle
            cylindrical_obstacle_sdf = lambda state: torch.norm(state[..., [0,4]], dim=-1) - 0.5 
            # Avoid SDF 
            self.sdf_avoid = lambda x: torch.minimum(space_boundary.boundary_sdf(x), cylindrical_obstacle_sdf(x))

            # For 3d visualization: only obstacles no boundary
            self.visualize_avoid_obstacle_sdf = cylindrical_obstacle_sdf

            # Reach 
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.75])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)

            # Define Plot config: 
            state_slices[8] = 0.5 # adjust z slice
            self.plot_config = {
                'state_slices': state_slices,
                'state_labels': state_labels,
                'x_axis_idx': 0, # x axis 
                'y_axis_idx': 4, # y axis
                'z_axis_idx': [1, 5], # vx and vy
            }

        elif self.config_num == 6:
            """
            Description: Multiple Cylindrical obstacles in environment 
            """

            def create_cylinder_sdf(state, center, radius): 
                if isinstance(state, torch.Tensor):
                    center = center.to(state.device)
                return torch.norm(state[..., [0, 4]] - center, dim=-1) - radius
            
            def combine_sdfs(x, sdf_list): 
                sdf_val = sdf_list[0](x)
                for sdf in sdf_list[1:]:
                    sdf_val = torch.minimum(sdf_val, sdf(x))
                return sdf_val
            
            center_0 = torch.tensor([-3., -1., ])
            radius_0 = 0.5
            cylinder_sdf_0 = lambda state: create_cylinder_sdf(state, center=center_0, radius=radius_0) 

            center_1 = torch.tensor([3., 1.])
            radius_1 = 0.5
            cylinder_sdf_1 = lambda state: create_cylinder_sdf(state, center=center_1, radius=radius_1)

            center_2 = torch.tensor([-3., 1.]) 
            radius_2 = 0.5
            cylinder_sdf_2 = lambda state: create_cylinder_sdf(state, center=center_2, radius=radius_2)

            center_3 = torch.tensor([3., -1.])
            radius_3 = 0.5
            cylinder_sdf_3 = lambda state: create_cylinder_sdf(state, center=center_3, radius=radius_3)

            self.sdf_avoid = lambda state: combine_sdfs(x=state, sdf_list=[cylinder_sdf_0, cylinder_sdf_1, cylinder_sdf_2, cylinder_sdf_3])
            
            # For 3d visualization: only obstacles no boundary
            self.visualize_avoid_obstacle_sdf = self.sdf_avoid

            # Reach Config 
            # Reach 
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8], 
                radius=0.75, 
                center=torch.Tensor([0, 0, 1.25])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)

            # Define Plot config: 
            state_slices[8] = 1.25 # adjust z slice
            self.plot_config = {
                'state_slices': state_slices,
                'state_labels': state_labels,
                'x_axis_idx': 0, # x axis 
                'y_axis_idx': 4, # y axis
                'z_axis_idx': [1, 5], # vx and vy
            }

        else: 
            raise ValueError("Invalid configuration number for the Quad10d environment.")
        
        self.configure_reach_avoid_fns()
        return 

    def configure_reach_avoid_fns(self, ): 
        """
        Function to properly change the sign of the reach and avoid functions so that they align
        with deepreach conventions according to the problem we want to solve

        NOTE: Starting with standard convention: 
            - Avoid: Negative = Unsafe, Positive = Safe
            - Reach: Negative = Outside Reach/Unsafe, Positive = Inside Reach/Safe

        Returns: None - function adjusts the parameters: 
            - sdf_reach
            - sdf_avoid

        For convention reference see: https://www.notion.so/Deepreach-Standard-Conventions-1a82da3e70b280589e4bfcee23e398a4  
        """
        
        if self.problem_type == "avoid": 
            # Negative Unsafe, Positive Safe
            self.avoid_fn = self.sdf_avoid 
            self.reach_fn = None 
            self.boundary_fn = self.sdf_avoid 
        elif self.problem_type == "reach": 
            # WRONG: Negative Outside Reach/Unsafe, Positive Inside Reach/Safe
            # Positive Unsafe/Outside reach set, Negative Safe/Inside reach set
            self.avoid_fn = None 
            self.reach_fn = lambda x: -1 * self.sdf_reach(x) #lambda x: -1 * self.sdf_reach(x) 
            self.boundary_fn = lambda x: -1 * self.sdf_reach(x)
        elif self.problem_type == "reach_avoid" or self.problem_type == "reach_avoid_ci": 
            # Avoid: Negative Unsafe, Positive Safe
            self.avoid_fn = self.sdf_avoid
            # Reach: Positive Unsafe/Outside reach set, Negative Safe/Inside reach set
            self.reach_fn = lambda x: -1 * self.sdf_reach(x)
            # Boundary Function: Positive Unsafe, Negative Safe
            self.boundary_fn = lambda x: torch.maximum(self.reach_fn(x), -self.avoid_fn(x))
        else: 
            raise ValueError("Invalid problem type for the Quad2DAttitude environment.")
        

########################################################### Quad 6d ###########################################################

class Quad6d_envs(): 
    """
    Environment Configurations for 10 d quadcopter
    """

    def __init__(self, config_num, problem_type):
        """
        Args: 
            - config_num: int: configuration number for the environment 
            - problem_type: str: problem type to generate sdfs ["reach", "avoid", "reach_avoid", "reach_avoid_ci"]
        """
        viable_obstacle_configs = [1,3,4, 5, 6]
        viable_problem_types = ["reach", "avoid", "reach_avoid", "reach_avoid_ci"]

        self.config_num = config_num 
        self.problem_type = problem_type 

        assert config_num in viable_obstacle_configs, "Invalid configuration number for the Quad10d environment."
        assert problem_type in viable_problem_types, "Invalid problem type for the Quad10d environment."

        # Sign Conventions: 
        # SDF Avoid: Negative = Unsafe, Positive = Safe
        # SDF Reach: Negative = Outside Reach/Unsafe, Positive = Inside Reach/Safe

        state_slices = [0, 0, 0, 0, 0, 0]
        state_labels = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']

        # State: [0, 1, 2,  3,   4,   5 ]
        # State: [x, y, z, v_x, v_y, v_z]
        if self.config_num == 1:
            # Obstacle Config: 1
            # Avoid Config 
            circle = boundary_functions.Circle(
                state_idis=[0, 1, 2],
                radius=0.5,  
                center=torch.Tensor([2.0, 0.0, 1.5])
            )
            self.sdf_avoid = circle.obstacle_sdf

            # For 3d visualization: only obstacles no boundary
            self.visualize_avoid_obstacle_sdf = circle.obstacle_sdf
            
            # Reach Config
            circle = boundary_functions.Circle(
                state_idis=[0, 1, 2], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.25])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)
            # NOTE: Combine with the above class and just have different init functions for the different dynamics classes or something

            # Define Plot config: 
            state_slices[3] = 1.25 # adjust z 
            self.plot_config = {
                'state_slices': state_slices, 
                'state_labels': state_labels, 
                'x_axis_idx': 0, # x-axis index
                'y_axis_idx': 2, # z-axis index
                'z_axis_idx': [3, 4], # vx, vy
            }
        elif self.config_num == 2:
            # Obstacle Config: 1
            # Avoid Config 
            space_boundary = boundary_functions.Boundary(
                state_idis=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                min_val=[-4.0, -1.9, 0 -1.9, -1.9, -1.9], 
                max_val=[4.0, 1.9, 2.5, 1.9, 1.9, 1.9]
                )
            circle = boundary_functions.Circle(
                state_idis=[0, 1, 2],
                radius=0.5,  
                center=torch.Tensor([2.0, 0.0, 1.5])
            )
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [circle])

            # For 3d visualization: only obstacles no boundary
            self.visualize_avoid_obstacle_sdf = circle.obstacle_sdf
            
            # Reach Config
            circle = boundary_functions.Circle(
                state_idis=[0, 1, 2], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.25])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)
            # NOTE: Combine with the above class and just have different init functions for the different dynamics classes or something

            # Define Plot config: 
            state_slices[3] = 1.25 # adjust z 
            self.plot_config = {
                'state_slices': state_slices, 
                'state_labels': state_labels, 
                'x_axis_idx': 0, # x-axis index
                'y_axis_idx': 2, # z-axis index
                'z_axis_idx': [3, 4], # vx, vy
            }
        else: 
            raise ValueError("Invalid configuration number for the Quad 6D environment.")
        
        self.configure_reach_avoid_fns()
        return 

    def configure_reach_avoid_fns(self, ): 
        """
        Function to properly change the sign of the reach and avoid functions so that they align
        with deepreach conventions according to the problem we want to solve

        NOTE: Starting with standard convention: 
            - Avoid: Negative = Unsafe, Positive = Safe
            - Reach: Negative = Outside Reach/Unsafe, Positive = Inside Reach/Safe

        Returns: None - function adjusts the parameters: 
            - sdf_reach
            - sdf_avoid

        For convention reference see: https://www.notion.so/Deepreach-Standard-Conventions-1a82da3e70b280589e4bfcee23e398a4  
        """
        
        if self.problem_type == "avoid": 
            # Negative Unsafe, Positive Safe
            self.avoid_fn = self.sdf_avoid 
            self.reach_fn = None 
            self.boundary_fn = self.sdf_avoid 
        elif self.problem_type == "reach": 
            # WRONG: Negative Outside Reach/Unsafe, Positive Inside Reach/Safe
            # Positive Unsafe/Outside reach set, Negative Safe/Inside reach set
            self.avoid_fn = None 
            self.reach_fn = lambda x: -1 * self.sdf_reach(x) #lambda x: -1 * self.sdf_reach(x) 
            self.boundary_fn = lambda x: -1 * self.sdf_reach(x)
        elif self.problem_type == "reach_avoid" or self.problem_type == "reach_avoid_ci": 
            # Avoid: Negative Unsafe, Positive Safe
            self.avoid_fn = self.sdf_avoid
            # Reach: Positive Unsafe/Outside reach set, Negative Safe/Inside reach set
            self.reach_fn = lambda x: -1 * self.sdf_reach(x)
            # Boundary Function: Positive Unsafe, Negative Safe
            self.boundary_fn = lambda x: torch.maximum(self.reach_fn(x), -self.avoid_fn(x))
        else: 
            raise ValueError("Invalid problem type for the Quad2DAttitude environment.")