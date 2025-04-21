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
            
        viable_obstacle_configs = [1, 2, 3, 4, 5, 6, 7, 8]
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
        viable_obstacle_configs = [1,2,3]
        viable_problem_types = ["reach", "avoid", "reach_avoid", "reach_avoid_ci"]

        self.config_num = config_num 
        self.problem_type = problem_type 

        assert config_num in viable_obstacle_configs, "Invalid configuration number for the Quad10d environment."
        assert problem_type in viable_problem_types, "Invalid problem type for the Quad10d environment."

        # Sign Conventions: 
        # SDF Avoid: Negative = Unsafe, Positive = Safe
        # SDF Reach: Negative = Outside Reach/Unsafe, Positive = Inside Reach/Safe

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
            # Reach Config
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.75])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)
            # NOTE: Combine with the above class and just have different init functions for the different dynamics classes or something

        ############################################################################
        elif self.config_num == 2:
            # TODO: NOTE: REMOVE THIS CONFIG TEST CONFIG - FOR TOY ENVIRONMENT VIZ TESTING
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
                center=torch.Tensor([2.5, 0.0, 1.5])
            )
            rectangle = boundary_functions.Rectangle(
                state_idis=[0, 4, 8], 
                min_val=torch.Tensor([-1.5, -1.5, 0.0]), 
                max_val=torch.Tensor([0.5, 0.5, 1.0])
            )
            space_boundary.boundary_sdf = lambda x: torch.tensor([0.0])
            self.sdf_avoid = boundary_functions.build_sdf(space_boundary, [circle, rectangle])
            # Reach Config
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.75])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)
            # NOTE: Combine with the above class and just have different init functions for the different dynamics classes or something
        ############################################################################
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
            # Reach Config
            circle = boundary_functions.Circle(
                state_idis=[0, 4, 8], 
                radius=0.5, 
                center=torch.Tensor([-3.0, 0.0, 1.75])
            )
            self.sdf_reach = lambda x: -1 * circle.obstacle_sdf(x)
            # NOTE: Combine with the above class and just have different init functions for the different dynamics classes or something
        ############################################################################
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