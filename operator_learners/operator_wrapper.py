import gymnasium as gym
import robosuite as suite
import numpy as np
from stable_baselines3 import SAC
#from domain_specific.detector import RoboSuite_PickPlace_Detector
from detector import RoboSuite_PickPlace_Detector

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

class PickWrapper(gym.Wrapper):
    def __init__(self, env, dense_reward=True):
        # Run super method
        super().__init__(env=env)
        # Define needed variables
        self.obj_body = self.env.sim.model.body_name2id('Can_main')
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.env.obj_to_use = 'can'
        self.dense_reward = dense_reward
        self.count_step = 0
        self.activate_pos = np.array(self.env.target_pos) # activate open door area (center pos)
        self.ray_bins = {'pick': 0.3, 'drop': 0.3, 'activate': 0.1}
        self.door_blocking = False # door is not blocking the way to the drop area initially
        self.door_locked = False # door configuration is unlocked initially
        self.in_activate_area = False # gripper is not in activate area initially
        self.env.single_object_mode = 2 # 2: Can only
        self.area_pos = {'pick': env.bin1_pos, 'drop': env.bin2_pos, 'activate': self.activate_pos}
        print(self.env.sim.model.body_names)


    def render(self, mode='human'):
        self.env.viewer.render()

    def reset(self, seed=None):
        # Reset the environment for the pick task
        try:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset()
            info = {}
        self.goal = self._sample_pick_goal()
        self.env.sim.forward()

        # Print out the body names to check the name of the gripper body
        print("Reseting environment...")

        # Return the initial observation
        return obs, info

    def _sample_pick_goal(self):
        # Get the current position of the 'Can' object
        object_pos = self.env.sim.data.body_xpos[self.obj_body]

        # Set the target goal to be directly above the current position of the 'Can' object
        goal = np.array([object_pos[0], object_pos[1], object_pos[2] + 0.2])  # target goal is 10cm above the current position

        return goal

    def step(self, action):
        # Perform the pick step
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        truncated = self.env.done or truncated
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        dist_to_button = np.linalg.norm(gripper_pos - self.activate_pos)
        if dist_to_button < self.ray_bins['activate'] and not self.in_activate_area:
            self.door_locked = not self.door_locked # toggle door state when the robot enters the activate area (does not toggle when it was already there)
            self.in_activate_area = True
        elif self.in_activate_area and dist_to_button > self.ray_bins['activate']:
            self.in_activate_area = False
        # Compute the reward based on the distance between the object and the target goal
        reward, terminated = self.compute_reward()
        
        return obs, reward, terminated, truncated, info

    def compute_reward(self):
        # Compute the reward based on the distance between the object and the target goal
        achieved_goal = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)
        can_dropped = not(achieved_goal['grasped({})'.format(self.obj_to_use)])
        treshold_distance = self.ray_bins["drop"]
        dist_to_target = achieved_goal['at({},drop)'.format(self.obj_to_use)]

        if self.dense_reward:
            # Compute a dense reward based on the distance to the target goal
            reward = -dist_to_target - 1.0 * can_dropped
        else:
            # Create a mask for the conditions (Can dropped or distance to drop off area > 0.1)
            mask_can_dropped = can_dropped
            mask_distance = dist_to_target < treshold_distance
            # Assign rewards based on the conditions
            reward = np.where(mask_can_dropped, -2.0, np.where(mask_distance, 1000.0, -1.0))

        done = (achieved_goal['at({},drop)'.format(self.obj_to_use)] <= treshold_distance)

        return reward.astype(np.float32), done


class ReachWrapper(gym.Wrapper):
    def __init__(self, env, pick_policy_path=None, dense_reward=True, render_init=False, augmented_obs=False):
        # Run super method
        super().__init__(env=env)
        # Define needed variables
        self.obj_body = self.env.sim.model.body_name2id('Can_main')
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.env.obj_to_use = 'can'
        self.dense_reward = dense_reward
        self.pick_policy_path = pick_policy_path
        self.count_step = 0
        # Adjust the positions to its referential (the bin position)
        self.pole1_xy_door = self.env.door_pos[:2] + np.array([0.1, 0.0]) + self.env.bin1_pos[:2]
        self.pole2_xy_door = self.env.door_pos[:2] - np.array([0.12, 0.0]) + self.env.bin1_pos[:2]
        self.center_xy_plate = self.env.plate_pos[:2] + np.array([0.1, 0.0]) + self.env.bin1_pos[:2]
        self.center_xy_cylinder = self.env.cylinder_pos[:2] + self.env.bin1_pos[:2]
        self.activate_pos = np.array(self.env.activate_pos) + self.env.bin1_pos
        self.light_switch_pos = np.array(self.env.lightswitch_pos) + self.env.bin1_pos
        # Environment parameters
        self.ray_bins = {'pick': 0.3, 'drop': 0.15, 'activate': 0.15, 'lightswitch': 0.1}
        self.door_blocking = False # door is not blocking the way to the drop area initially
        self.in_activate_area = False # gripper is not in activate area initially
        self.near_lightswitch = False # gripper is not near lightswitch initially
        self.env.single_object_mode = 2 # 2: Can only
        self.detector = RoboSuite_PickPlace_Detector(self)
        self.area_pos = {'pick': env.bin1_pos, 'drop': env.bin2_pos, 'activate': env.activate_pos, 'lightswitch': env.lightswitch_pos}
        self.render_init = render_init
        self.reset_door_locked = self.env.door.lock
        self.reset_light_on = self.env.light_on
        self.max_distance = 10
        self.augmented_obs = augmented_obs
        self.gripper_on = False
        #TODO: Change action space so that it fits the environment definition and fixes the following error: 
        # File "robosuite/environments/robot_env.py", line 575, in _pre_action
        # assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format("
        # self.env.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # self.env.action_dim = 3

        # set up observation space
        self.obs_dim = self.env.obs_dim + 6 # 6 extra dimensions for the distance to objects/areas

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        if augmented_obs:
            self.init_state = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
            #goal_low = np.zeros(len(self.init_state))
            goal_high = self.max_distance * np.ones(len(self.init_state))
            self.obs_dim = self.env.obs_dim
            high = np.concatenate((high, goal_high))
            low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    def pick_reset(self):
        """
        Resets the environment to a state where the gripper is holding the object
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

        self.reset_step_count = 0
        #print("Moving up...")
        for _ in range(10):
            self.env.step([0,0,0.5,0])
            self.env.render() if self.render_init else None

        #print("Opening gripper...")
        while not state['open(gripper)']:
            self.env.step([0,0,0,-0.1])
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 100:
                return False
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving gripper over object...")
        while not state['over(gripper,{})'.format(self.obj_to_use)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_body])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            self.env.step(action)
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 200:
                return False
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(self.obj_to_use)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_body])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]])
            self.env.step(action)
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 400:
                return False
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Closing gripper...")
        while not state['grasped({})'.format(self.obj_to_use)]:
            self.env.step([0,0,0,0.1])
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 30:
                return False
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Lifting object...")
        while not state['picked_up({})'.format(self.obj_to_use)]:
            self.env.step([0,0,0.2,0])
            self.env.render() if self.render_init else None
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 100:
                return False
        self.reset_step_count = 0
        self.env.time_step = 0

        return True

    def reset(self, seed=None):
        # Reset the environment for the reach task
        success = False
        buffer_light_on = False
        if not(self.env.light_on):
            # Temporally turn on the light to reset the environment
            self.env.light_on = True
            buffer_light_on = True
        while not success:
            try:
                obs, info = self.env.reset(seed=seed)
            except:
                obs = self.env.reset(seed=seed)
                info = {}
            done = False
            if self.pick_policy_path is not None:
                self.pick_policy = SAC.load(self.pick_policy_path)
                while not done:
                    action, _states = self.pick_policy.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                success = True
            else:
                success = self.pick_reset()
        if buffer_light_on:
            # Restore the light state
            self.env.light_on = False
        
        self.env.door.lock = self.reset_door_locked
        self.env.light_on = self.reset_light_on
        self.in_activate_area = False
        self.near_lightswitch = False

        obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations()
        # Flatten the observation
        obs = self.env._flatten_obs(obs)

        self._sample_reach_goal()
        self.sim.forward()

        # Compute the xy distance of the end effector to objects (lidar-like)
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        distance_to_door1 = np.linalg.norm(gripper_pos[:2] - self.pole1_xy_door)
        distance_to_door2 = np.linalg.norm(gripper_pos[:2] - self.pole2_xy_door)
        distance_to_plate = np.linalg.norm(gripper_pos[:2] - self.center_xy_plate)
        distance_to_cylinder = np.linalg.norm(gripper_pos[:2] - self.center_xy_cylinder)
        distance_to_activate = np.linalg.norm(gripper_pos - self.activate_pos)
        distance_to_lightswitch = np.linalg.norm(gripper_pos - self.light_switch_pos)

        #print("Distances: {}, {}, {}, {}".format(distance_to_door1, distance_to_door2, distance_to_plate, distance_to_cylinder))

        if self.env.light_on:
           obs = np.concatenate([obs, np.asarray([distance_to_door1, distance_to_door2, distance_to_plate, distance_to_cylinder, distance_to_activate, distance_to_lightswitch])])
        else:
            # If the light is off, set the distances to max_distance but for the lightswitch distance
            obs = np.concatenate([obs, np.asarray([self.max_distance,self.max_distance,self.max_distance,self.max_distance,self.max_distance,distance_to_lightswitch])])
        # Return the initial observation

        if self.augmented_obs:
            achieved_goal = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
            obs = np.concatenate((obs, achieved_goal))

        return obs, info

    def _sample_reach_goal(self):
        # Get the current target position from the PickPlace instance
        target_pos = self.sim.data.body_xpos[self.sim.model.body_name2id('VisualCan_main')]

        # Add 0.1 to the z-axis of the target position
        goal_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.2])

        # Set the new goal position
        self.goal = goal_pos

    def step(self, action):
        # Perform the reach step
        if not self.gripper_on:
            action[3] = 0.0 #TODO: Remove this line when the lock gripper action is fixed, working fine for now
        truncated = False
        # Checks if the robot end effector is in the activate area
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        # Compute the xy distance of the end effector to objects (lidar-like)
        distance_to_door1 = np.linalg.norm(gripper_pos[:2] - self.pole1_xy_door)
        distance_to_door2 = np.linalg.norm(gripper_pos[:2] - self.pole2_xy_door)
        distance_to_plate = np.linalg.norm(gripper_pos[:2] - self.center_xy_plate)
        distance_to_cylinder = np.linalg.norm(gripper_pos[:2] - self.center_xy_cylinder)
        distance_to_activate = np.linalg.norm(gripper_pos - self.activate_pos)
        distance_to_lightswitch = np.linalg.norm(gripper_pos - self.light_switch_pos)
        if distance_to_activate < self.ray_bins['activate'] and not self.in_activate_area:
            self.env.door.lock = False # toggle door state when the robot enters the activate area (does not toggle when it was already there)
            self.in_activate_area = True # set the flag to true so that the door state is not toggled again
            self.env.open_door("Door")
            print("Door unlocked!")
        distance_to_lightswitch = np.linalg.norm(gripper_pos - self.light_switch_pos)
        if distance_to_lightswitch < self.ray_bins['lightswitch'] and not self.near_lightswitch:
            self.near_lightswitch = True 
            self.env.light_on = not self.env.light_on
            print("Lightswitch toggled!")
        elif self.near_lightswitch and distance_to_lightswitch > self.ray_bins['lightswitch']:
            self.near_lightswitch = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        truncated = truncated or self.env.done

        #print("Distance to lightswitch: {}".format(distance_to_lightswitch))
        #print("Light on: {}".format(self.env.light_on))


        if self.env.light_on:
            obs = np.concatenate([obs, np.asarray([distance_to_door1, distance_to_door2, distance_to_plate, distance_to_cylinder, distance_to_activate, distance_to_lightswitch])])
            #print("Distances: {}, {}, {}, {}, {}, {}".format(distance_to_door1, distance_to_door2, distance_to_plate, distance_to_cylinder, distance_to_activate, distance_to_lightswitch))
        else:
            # If the light is off, set the distances to 0 but for the lightswitch distance
            obs = np.concatenate([obs, np.asarray([self.max_distance,self.max_distance,self.max_distance,self.max_distance,self.max_distance,distance_to_lightswitch])])
            #print("Distances: {}, {}, {}, {}, {}, {}".format(0, 0, 0, 0, 0, distance_to_lightswitch))

        achieved_goal = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)
        treshold_distance = self.ray_bins["drop"]
        condition = (achieved_goal['at({},drop)'.format(self.obj_to_use)] <= treshold_distance)

        if self.augmented_obs:
            achieved_goal = self.detector.dict_to_array(achieved_goal)
            obs = np.concatenate((obs, achieved_goal))

        # Compute the reward based on the distance between the object and the target goal
        reward, _ = self.compute_reward()
        info['is_success'] = condition
        #if condition:
        #    print("Success!")
        
        return obs, reward, terminated, truncated, info

    def compute_reward(self):
        # Compute the reward based on the distance between the object and the target goal
        achieved_goal = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)
        can_dropped = not(achieved_goal['grasped({})'.format(self.obj_to_use)])
        treshold_distance = self.ray_bins["drop"]
        dist_to_target = achieved_goal['at({},drop)'.format(self.obj_to_use)]

        if self.dense_reward:
            # Compute a dense reward based on the distance to the target goal
            reward = -dist_to_target - 1.0 * can_dropped
        else:
            # Create a mask for the conditions (Can dropped or distance to drop off area > 0.1)
            mask_can_dropped = can_dropped
            mask_distance = dist_to_target < treshold_distance
            # Assign rewards based on the conditions
            #reward = np.where(mask_can_dropped, -2.0, np.where(mask_distance, 1000.0, -1.0))
            reward = np.where(mask_distance, 1000.0, -1.0)

        done = (achieved_goal['at({},drop)'.format(self.obj_to_use)] <= treshold_distance)

        return reward, done


class TaskWrapper(gym.Wrapper):
    def __init__(self, env, pick_policy_path=None, dense_reward=True, render_init=False, augmented_obs=False):
        # Run super method
        super().__init__(env=env)
        # Define needed variables
        self.obj_body = self.env.sim.model.body_name2id('Can_main')
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.env.obj_to_use = 'can'
        self.dense_reward = dense_reward
        self.pick_policy_path = pick_policy_path
        self.count_step = 0
        # Adjust the positions to its referential (the bin position)
        self.pole1_xy_door = self.env.door_pos[:2] + np.array([0.1, 0.0]) + self.env.bin1_pos[:2]
        self.pole2_xy_door = self.env.door_pos[:2] - np.array([0.12, 0.0]) + self.env.bin1_pos[:2]
        self.center_xy_plate = self.env.plate_pos[:2] + np.array([0.1, 0.0]) + self.env.bin1_pos[:2]
        self.center_xy_cylinder = self.env.cylinder_pos[:2] + self.env.bin1_pos[:2]
        self.activate_pos = np.array(self.env.activate_pos) + self.env.bin1_pos
        self.light_switch_pos = np.array(self.env.lightswitch_pos) + self.env.bin1_pos
        # Environment parameters
        self.ray_bins = {'pick': 0.3, 'drop': 0.15, 'activate': 0.15, 'lightswitch': 0.1}
        self.door_blocking = False # door is not blocking the way to the drop area initially
        self.in_activate_area = False # gripper is not in activate area initially
        self.near_lightswitch = False # gripper is not near lightswitch initially
        self.env.single_object_mode = 2 # 2: Can only
        self.detector = RoboSuite_PickPlace_Detector(self)
        self.area_pos = {'pick': env.bin1_pos, 'drop': env.bin2_pos, 'activate': env.activate_pos, 'lightswitch': env.lightswitch_pos}
        self.render_init = render_init
        self.reset_door_locked = self.env.door.lock
        self.reset_light_on = self.env.light_on
        self.max_distance = 10
        self.augmented_obs = augmented_obs
        self.gripper_on = False

        # set up observation space
        self.obs_dim = self.env.obs_dim + 6 # 6 extra dimensions for the distance to objects/areas

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        if augmented_obs:
            self.init_state = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
            #goal_low = np.zeros(len(self.init_state))
            goal_high = self.max_distance * np.ones(len(self.init_state))
            self.obs_dim = self.env.obs_dim
            high = np.concatenate((high, goal_high))
            low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    def set_gripper(self, val):
        self.gripper_on = val

    def reset(self, seed=None):
        # Reset the environment for the task
        try:
            obs, info = self.env.reset(seed=seed)
        except:
            obs = self.env.reset(seed=seed)
            info = {}
        self.env.door.lock = self.reset_door_locked
        self.env.light_on = self.reset_light_on
        self.in_activate_area = False
        self.near_lightswitch = False

        obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations()
        # Flatten the observation
        obs = self.env._flatten_obs(obs)

        self._sample_reach_goal()
        self.sim.forward()

        # Compute the xy distance of the end effector to objects (lidar-like)
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        distance_to_door1 = np.linalg.norm(gripper_pos[:2] - self.pole1_xy_door)
        distance_to_door2 = np.linalg.norm(gripper_pos[:2] - self.pole2_xy_door)
        distance_to_plate = np.linalg.norm(gripper_pos[:2] - self.center_xy_plate)
        distance_to_cylinder = np.linalg.norm(gripper_pos[:2] - self.center_xy_cylinder)
        distance_to_activate = np.linalg.norm(gripper_pos - self.activate_pos)
        distance_to_lightswitch = np.linalg.norm(gripper_pos - self.light_switch_pos)

        #print("Distances: {}, {}, {}, {}".format(distance_to_door1, distance_to_door2, distance_to_plate, distance_to_cylinder))

        if self.env.light_on:
           obs = np.concatenate([obs, np.asarray([distance_to_door1, distance_to_door2, distance_to_plate, distance_to_cylinder, distance_to_activate, distance_to_lightswitch])])
        else:
            # If the light is off, set the distances to max_distance but for the lightswitch distance
            obs = np.concatenate([obs, np.asarray([self.max_distance,self.max_distance,self.max_distance,self.max_distance,self.max_distance,distance_to_lightswitch])])
        # Return the initial observation

        if self.augmented_obs:
            achieved_goal = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
            obs = np.concatenate((obs, achieved_goal))

        return obs, info

    def _sample_reach_goal(self):
        # Get the current target position from the PickPlace instance
        target_pos = self.sim.data.body_xpos[self.sim.model.body_name2id('VisualCan_main')]

        # Add 0.1 to the z-axis of the target position
        goal_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.2])

        # Set the new goal position
        self.goal = goal_pos

    def step(self, action):
        # Perform the reach step
        if not self.gripper_on:
            action[3] = 0.0 #TODO: Remove this line when the lock gripper action is fixed, working fine for now
        truncated = False
        # Checks if the robot end effector is in the activate area
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
        # Compute the xy distance of the end effector to objects (lidar-like)
        distance_to_door1 = np.linalg.norm(gripper_pos[:2] - self.pole1_xy_door)
        distance_to_door2 = np.linalg.norm(gripper_pos[:2] - self.pole2_xy_door)
        distance_to_plate = np.linalg.norm(gripper_pos[:2] - self.center_xy_plate)
        distance_to_cylinder = np.linalg.norm(gripper_pos[:2] - self.center_xy_cylinder)
        distance_to_activate = np.linalg.norm(gripper_pos - self.activate_pos)
        distance_to_lightswitch = np.linalg.norm(gripper_pos - self.light_switch_pos)
        if distance_to_activate < self.ray_bins['activate'] and not self.in_activate_area:
            self.env.door.lock = False # toggle door state when the robot enters the activate area (does not toggle when it was already there)
            self.in_activate_area = True # set the flag to true so that the door state is not toggled again
            self.env.open_door("Door")
            print("Door unlocked!")
        distance_to_lightswitch = np.linalg.norm(gripper_pos - self.light_switch_pos)
        if distance_to_lightswitch < self.ray_bins['lightswitch'] and not self.near_lightswitch:
            self.near_lightswitch = True 
            self.env.light_on = not self.env.light_on
            print("Lightswitch toggled!")
        elif self.near_lightswitch and distance_to_lightswitch > self.ray_bins['lightswitch']:
            self.near_lightswitch = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, terminated, info = self.env.step(action)
        truncated = truncated or self.env.done

        if self.env.light_on:
            obs = np.concatenate([obs, np.asarray([distance_to_door1, distance_to_door2, distance_to_plate, distance_to_cylinder, distance_to_activate, distance_to_lightswitch])])
            #print("Distances: {}, {}, {}, {}, {}, {}".format(distance_to_door1, distance_to_door2, distance_to_plate, distance_to_cylinder, distance_to_activate, distance_to_lightswitch))
        else:
            # If the light is off, set the distances to 0 but for the lightswitch distance
            obs = np.concatenate([obs, np.asarray([self.max_distance,self.max_distance,self.max_distance,self.max_distance,self.max_distance,distance_to_lightswitch])])
            #print("Distances: {}, {}, {}, {}, {}, {}".format(0, 0, 0, 0, 0, distance_to_lightswitch))

        achieved_goal = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)
        treshold_distance = self.ray_bins["drop"]

        condition = (achieved_goal['at({},drop)'.format(self.obj_to_use)] <= treshold_distance)

        if self.augmented_obs:
            achieved_goal = self.detector.dict_to_array(achieved_goal)
            obs = np.concatenate((obs, achieved_goal))

        # Compute the reward based on the distance between the object and the target goal
        reward, _ = self.compute_reward()
        info['is_success'] = condition
        #if condition:
        #    print("Success!")
        
        return obs, reward, terminated, truncated, info

    def compute_reward(self):
        # Compute the reward based on the distance between the object and the target goal
        achieved_goal = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)
        can_dropped = not(achieved_goal['grasped({})'.format(self.obj_to_use)])
        treshold_distance = self.ray_bins["drop"]
        dist_to_target = achieved_goal['at({},drop)'.format(self.obj_to_use)]
        gripper_name = 'gripper'
        dist_xy_can = achieved_goal['at_grab_level({},{})'.format(gripper_name,self.obj_to_use)]
        dist_z_can = achieved_goal['over({},{})'.format(gripper_name,self.obj_to_use)]
        dist_to_can = np.sqrt(dist_xy_can**2 + dist_z_can**2)

        if self.dense_reward:
            if (dist_to_target < treshold_distance):
                if can_dropped:
                    # If the can is dropped and the distance to the drop area is less than the treshold, return a high reward
                    return 10000, True
                else:
                    # If the can is not dropped and the distance to the drop area is less than the treshold, return a low reward
                    return -1, False
            elif can_dropped:
                # If the can is dropped and the distance to the drop area is greater than the treshold, return a low reward
                return -dist_to_can - 30, False
            else:
                # If the can is not dropped and the distance to the drop area is greater than the treshold, return a low reward
                return -dist_to_target - 3, False
        else:
            # Create a mask for the conditions (Can dropped or distance to drop off area > 0.1)
            mask_can_dropped = can_dropped
            mask_distance = (dist_to_target < treshold_distance) and can_dropped 
            # Assign rewards based on the conditions
            reward = np.where(mask_can_dropped, -2.0, np.where(mask_distance, 1000.0, -1.0))

        done = (achieved_goal['at({},drop)'.format(self.obj_to_use)] <= treshold_distance)
        done = done and can_dropped

        return reward, done