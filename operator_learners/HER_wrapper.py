import gymnasium as gym
from gymnasium import spaces
import numpy as np
#from domain_specific.detector import RoboSuite_PickPlace_Detector
from detector import RoboSuite_PickPlace_Detector
from copy import deepcopy

class No_end(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, False, truncated, info

class HERWrapper(gym.Wrapper):
    def __init__(self, env, goal="at(can,drop)", dense_reward=False, augmented_obs=False, symbolic_goal=True, factor=1, return_int=False):
        """
        HER wrapper for RoboSuite environments using the detector function to augment the goal space with symbolic and high level goals
        """
        super().__init__(env)
        self.desired_goal_key = 'desired_goal'
        self.achieved_goal_key = 'achieved_goal'
        self.desired_goal = None
        self.achieved_goal = None
        self.detector = RoboSuite_PickPlace_Detector(env, factor=factor, return_int=return_int)
        self.dense_reward = dense_reward
        self.init_state = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
        self.goal = goal
        self.factor = factor
        
        # set up observation space
        self.augmented_obs = augmented_obs
        self.symbolic_goal = symbolic_goal
        self.obs_dim = self.env.obs_dim
        if symbolic_goal:
            goal_low = np.zeros(len(self.init_state))
            goal_high = self.env.max_distance * self.factor * np.ones(len(self.init_state))
        else:
            goal_low = low
            goal_high = high
        if self.augmented_obs:
            high = np.concatenate((np.inf * np.ones(self.obs_dim), goal_high))
            low = -high
        else:
            high = np.inf * np.ones(self.obs_dim)
            low = -high

        observation_space = gym.spaces.Box(low, high, dtype=np.float64)

        self.observation_space = spaces.Dict({
            'observation': observation_space,
            self.desired_goal_key: spaces.Box(goal_low, goal_high, dtype=np.float64),
            self.achieved_goal_key: spaces.Box(goal_low, goal_high, dtype=np.float64)
        })

    def set_goal(self, goal):
        self.goal = goal

    def reset(self, seed=None, **kwargs):
        try:
            obs, info = self.env.reset(seed=seed, **kwargs)
            print("except1")
        except TypeError:
            obs, info = self.env.reset()
            print("except2")
        except:
            obs = self.env.reset(seed=seed, **kwargs)
            print("except3")
            info = {}
        achieved_goal = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
        if self.augmented_obs:
            obs = np.concatenate((obs, achieved_goal))
        print(type(obs))
        print(obs)
        desired_goal = self.compute_desired_goal(obs=obs)
        if not self.symbolic_goal:
            achieved_goal = obs
        return {
            'observation': obs,
            self.desired_goal_key: desired_goal,
            self.achieved_goal_key: achieved_goal
        }, info

    def step(self, action):
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
            terminated = done
        truncated = truncated or self.env.done
        achieved_goal = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)
        desired_goal = self.compute_desired_goal(obs=obs)
        treshold_distance = self.ray_bins["drop"] * self.factor
        condition = (achieved_goal['at({},drop)'.format(self.obj_to_use)] < treshold_distance)
        achieved_goal = self.detector.dict_to_array(achieved_goal)
        info['is_success'] = condition
        # Computes the termination condition (distance to drop off area <= 0.1)
        terminated = terminated or condition
        if self.augmented_obs:
            obs = np.concatenate((obs, achieved_goal))
        if not self.symbolic_goal:
            achieved_goal = obs
        reward = self.compute_reward(achieved_goal, desired_goal, info=info)
        if condition:
            print("Success")
            print("Reward: ", reward)
        return {
            'observation': obs,
            self.desired_goal_key: desired_goal,
            self.achieved_goal_key: achieved_goal
        }, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute the reward based on the distance between the object and the target goal
        treshold_distance = self.ray_bins["drop"]
        if not self.symbolic_goal:
            if self.dense_reward:
                try:
                    # Vectorized version (Batch of states through achieved_goal=vector of achieved goals, desired_goal=vector of desired goals and info=vector of info dicts)
                    # Select only the position of the object for each state in the batch
                    achieved_pos = achieved_goal[:, 40:43]
                    desired_pos = desired_goal[:, 40:43]
                    return -np.linalg.norm(achieved_pos-desired_pos, axis=1)
                except:
                    achieved_pos = achieved_goal[40:43]
                    desired_pos = desired_goal[40:43]
                    return -np.linalg.norm(achieved_pos-desired_pos)
            else:
                try:
                    # Vectorized version (Batch of states through achieved_goal=vector of achieved goals, desired_goal=vector of desired goals and info=vector of info dicts)
                    # Select only the position of the object for each state in the batch
                    achieved_pos = achieved_goal[:, 40:43]
                    desired_pos = desired_goal[:, 40:43]
                    reward = -np.float32(np.linalg.norm(achieved_pos-desired_pos, axis=1) > treshold_distance)
                    # transform 0 to 1000 and let -1 as it is
                    reward = np.where(reward == 0, 1000.0, -1.0)
                    return reward
                except:
                    achieved_pos = achieved_goal[40:43]
                    desired_pos = desired_goal[40:43]
                    reward = -np.float32(np.linalg.norm(achieved_pos-desired_pos) > treshold_distance)
                    reward = 1000.0 if reward == 0 else -1.0
                    return reward
        try:
            # Vectorized version select only the first element (index 0) of the array for each state in the batch
            distance = np.linalg.norm(achieved_goal[:, 0:1] - desired_goal[:, 0:1], axis=1)
        except:
            # Non-vectorized version
            distance = np.linalg.norm(achieved_goal[0] - desired_goal[0])

        if not self.dense_reward:
            # Create a mask 
            mask_distance = distance < treshold_distance * self.factor
            # Assign rewards based on the conditions
            rewards = np.where(mask_distance, 1000.0, -1.0)
        else:
            # Compute the reward based on the distance to drop off area
            rewards = -distance if distance > treshold_distance  * self.factor else 1000.0
        return rewards.astype(np.float32)

    def compute_desired_goal(self, obs=None):
        if not self.symbolic_goal:
            #return np.asarray(self.area_pos['drop'])
            obs_copy = deepcopy(obs)
            obs_copy[40:43] = self.area_pos['drop']
            return obs_copy

        # Compute the goal based on the current state of the environment
        current_state = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)

        # Compute environment distances
        dist_drop_pick = np.linalg.norm(self.area_pos['drop'] - self.area_pos['pick'])
        dist_pick_activate = np.linalg.norm(self.area_pos['pick'] - self.area_pos['activate'])

        if type(self.goal) is list:
            self.goal = self.goal[0]
        if type(self.goal) is str or type(self.goal) is np.str_:
            # Compute the desired goal from the current state
            desired_goal = deepcopy(current_state)
            desired_goal[self.goal] = 0.0
            desired_goal['at({},pick)'.format(self.goal.split(',')[0][3:])] = dist_drop_pick * self.factor
            desired_goal['at_gripper(gripper,pick)'] = dist_drop_pick * self.factor
            desired_goal['at_gripper(gripper,activate)'] = dist_pick_activate * self.factor
            desired_goal['at_gripper(gripper,drop)'] = 0.0
        else:
            desired_goal = deepcopy(current_state)
            for key, value in self.goal.items():
                desired_goal[key] = value
        #print("Desired goal: ", desired_goal)
        return np.asarray([v for k, v in sorted(desired_goal.items())])
    

class HERWrapper_RL(gym.Wrapper):
    def __init__(self, env, goal="at(can,drop)", dense_reward=False, augmented_obs=False, symbolic_goal=True):
        """
        HER wrapper for RoboSuite environments using the detector function to augment the goal space with symbolic and high level goals
        """
        super().__init__(env)
        self.desired_goal_key = 'desired_goal'
        self.achieved_goal_key = 'achieved_goal'
        self.desired_goal = None
        self.achieved_goal = None
        self.detector = RoboSuite_PickPlace_Detector(env)
        self.dense_reward = dense_reward
        self.init_state = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
        self.goal = goal
        
        # set up observation space
        self.augmented_obs = augmented_obs
        self.symbolic_goal = symbolic_goal
        if self.augmented_obs:
            high = np.concatenate((np.inf * np.ones(self.obs_dim), goal_high))
            low = -high
        else:
            high = np.inf * np.ones(self.obs_dim)
            low = -high
        if symbolic_goal:
            goal_low = np.zeros(len(self.init_state))
            goal_high = self.env.max_distance * np.ones(len(self.init_state))
        else:
            #goal_low = np.zeros(3)
            #goal_high = self.env.max_distance * np.ones(3)
            goal_low = low
            goal_high = high
        observation_space = gym.spaces.Box(low, high, dtype=np.float64)

        self.observation_space = spaces.Dict({
            'observation': observation_space,
            self.desired_goal_key: spaces.Box(goal_low, goal_high, dtype=np.float64),
            self.achieved_goal_key: spaces.Box(goal_low, goal_high, dtype=np.float64)
        })

    def reset(self, seed=None, **kwargs):
        try:
            obs, info = self.env.reset(seed=seed, **kwargs)
        except:
            obs = self.env.reset(seed=seed, **kwargs)
            info = {}
        achieved_goal = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
        if self.augmented_obs:
            obs = np.concatenate((obs, achieved_goal))
        desired_goal = self.compute_desired_goal(obs=obs)
        if not self.symbolic_goal:
            #object_pos = self.env.sim.data.body_xpos[self.obj_body]
            #achieved_goal = object_pos#
            achieved_goal = obs
        return {
            'observation': obs,
            self.desired_goal_key: desired_goal,
            self.achieved_goal_key: achieved_goal
        }, info

    def set_goal(self, goal):
        self.goal = goal

    def step(self, action):
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
            terminated = done
        truncated = truncated or self.env.done
        achieved_goal = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)
        desired_goal = self.compute_desired_goal(obs=obs)
        can_dropped = not(achieved_goal['grasped({})'.format(self.obj_to_use)])
        at_drop = achieved_goal['at({},drop)'.format(self.obj_to_use)]
        treshold_distance = self.ray_bins["drop"]
        condition = (achieved_goal['at({},drop)'.format(self.obj_to_use)] <= treshold_distance)
        achieved_goal = self.detector.dict_to_array(achieved_goal)
        #print(condition)
        info = {'can_dropped': can_dropped, 'at_drop': at_drop, 'is_success': condition}
        #print(info)
        # Computes the termination condition (distance to drop off area <= 0.1)
        terminated = terminated or condition
        if self.augmented_obs:
            obs = np.concatenate((obs, achieved_goal))
        if not self.symbolic_goal:
            #object_pos = self.env.sim.data.body_xpos[self.obj_body]
            #achieved_goal = object_pos
            achieved_goal = obs
        reward = self.compute_reward(achieved_goal, desired_goal, info=info)
        #print("Reward: {}".format(reward))
        return {
            'observation': obs,
            self.desired_goal_key: desired_goal,
            self.achieved_goal_key: achieved_goal
        }, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute the reward based on the distance between the object and the target goal
        treshold_distance = self.ray_bins["drop"]
        if not self.symbolic_goal:
            if self.dense_reward:
                try:
                    # Vectorized version (Batch of states through achieved_goal=vector of achieved goals, desired_goal=vector of desired goals and info=vector of info dicts)
                    # Select only the position of the object for each state in the batch
                    achieved_pos = achieved_goal[:, 40:43]
                    desired_pos = desired_goal[:, 40:43]
                    return -np.linalg.norm(achieved_pos-desired_pos, axis=1)
                except:
                    achieved_pos = achieved_goal[40:43]
                    desired_pos = desired_goal[40:43]
                    return -np.linalg.norm(achieved_pos-desired_pos)
            else:
                try:
                    # Vectorized version (Batch of states through achieved_goal=vector of achieved goals, desired_goal=vector of desired goals and info=vector of info dicts)
                    # Select only the position of the object for each state in the batch
                    achieved_pos = achieved_goal[:, 40:43]
                    desired_pos = desired_goal[:, 40:43]
                    reward = -np.float32(np.linalg.norm(achieved_pos-desired_pos, axis=1) > treshold_distance)
                    return reward
                except:
                    achieved_pos = achieved_goal[40:43]
                    desired_pos = desired_goal[40:43]
                    reward = -np.float32(np.linalg.norm(achieved_pos-desired_pos) > treshold_distance)
                    return reward
        groundings_keys = list(self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False).keys())
        can_dropped_index = groundings_keys.index('grasped({})'.format(self.obj_to_use))
        at_drop_index = groundings_keys.index('at({},drop)'.format(self.obj_to_use))
        try:
            # Vectorized version
            can_dropped = achieved_goal[:, can_dropped_index]
            achieved_at_drop = achieved_goal[:, at_drop_index]
            #desired_at_drop = desired_goal[:, at_drop_index]
            desired_at_drop = np.zeros(achieved_at_drop.shape)
            distance = np.array(achieved_at_drop - desired_at_drop)
        except:
            # Non-vectorized version
            can_dropped = info['can_dropped']
            achieved_at_drop = info['at_drop']
            #desired_at_drop = desired_goal[at_drop_index]
            desired_at_drop = 0.0
            distance = np.linalg.norm(achieved_at_drop - desired_at_drop)

        if not self.dense_reward:
            # Create a mask for the conditions (Can dropped or distance to drop off area > 0.1)
            mask_can_dropped = can_dropped
            mask_distance = distance < treshold_distance
            # Assign rewards based on the conditions
            rewards = np.where(mask_can_dropped, -2.0, np.where(mask_distance, 1000.0, -1.0))
        else:
            # Create a mask for the condition (Can dropped)
            mask = can_dropped
            # Compute the reward based on the distance to drop off area
            rewards = -distance - 1.0 * mask
    
        return rewards.astype(np.float32)

    def compute_desired_goal(self, obs=None):
        if not self.symbolic_goal:
            #return np.asarray(self.area_pos['drop'])
            obs_copy = deepcopy(obs)
            obs_copy[40:43] = self.area_pos['drop']
            return obs_copy

        # Compute the goal based on the current state of the environment
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.obj_body])
        current_state = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)

        # Compute environment distances
        dist_drop_pick = np.linalg.norm(self.area_pos['drop'] - self.area_pos['pick'])
        dist_drop_activate = np.linalg.norm(self.area_pos['drop'] - self.area_pos['activate'])
        dist_pick_activate = np.linalg.norm(self.area_pos['pick'] - self.area_pos['activate'])

        # Compute the desired goal from the current state
        desired_goal = deepcopy(current_state)
        desired_goal[self.goal] = 0.0
        #desired_goal['at({},drop)'.format(self.obj_to_use)] = 0.0
        desired_goal['at({},pick)'.format(self.goal.split(',')[0][3:])] = dist_drop_pick
        #desired_goal['at({},activate)'.format(self.goal.split(',')[0][3:])] = dist_pick_activate
        desired_goal['at_gripper(gripper,pick)'] = dist_drop_pick
        desired_goal['at_gripper(gripper,activate)'] = dist_pick_activate
        desired_goal['at_gripper(gripper,drop)'] = 0.0

        return np.asarray([v for k, v in sorted(desired_goal.items())])



class HERSave(gym.Wrapper):
    def __init__(self, env, goal="at(can,drop)", dense_reward=False, augmented_obs=False, symbolic_goal=True):
        """
        HER wrapper for RoboSuite environments using the detector function to augment the goal space with symbolic and high level goals
        """
        super().__init__(env)
        self.desired_goal_key = 'desired_goal'
        self.achieved_goal_key = 'achieved_goal'
        self.desired_goal = None
        self.achieved_goal = None
        self.detector = RoboSuite_PickPlace_Detector(env)
        self.dense_reward = dense_reward
        self.init_state = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
        self.goal = goal
        
        # set up observation space
        self.augmented_obs = augmented_obs
        self.symbolic_goal = symbolic_goal
        self.obs_dim = self.env.obs_dim
        if self.augmented_obs:
            high = np.concatenate((np.inf * np.ones(self.obs_dim), goal_high))
            low = -high
        else:
            high = np.inf * np.ones(self.obs_dim)
            low = -high
        if symbolic_goal:
            goal_low = np.zeros(len(self.init_state))
            goal_high = self.env.max_distance * np.ones(len(self.init_state))
        else:
            #goal_low = np.zeros(3)
            #goal_high = self.env.max_distance * np.ones(3)
            goal_low = low
            goal_high = high
        observation_space = gym.spaces.Box(low, high, dtype=np.float64)

        self.observation_space = spaces.Dict({
            'observation': observation_space,
            self.desired_goal_key: spaces.Box(goal_low, goal_high, dtype=np.float64),
            self.achieved_goal_key: spaces.Box(goal_low, goal_high, dtype=np.float64)
        })

    def reset(self, seed=None, **kwargs):
        try:
            obs, info = self.env.reset(seed=seed, **kwargs)
        except:
            obs = self.env.reset(seed=seed, **kwargs)
            info = {}
        achieved_goal = self.detector.get_groundings(as_dict=False, binary_to_float=True, return_distance=True)
        if self.augmented_obs:
            obs = np.concatenate((obs, achieved_goal))
        desired_goal = self.compute_desired_goal(obs=obs)
        if not self.symbolic_goal:
            #object_pos = self.env.sim.data.body_xpos[self.obj_body]
            #achieved_goal = object_pos#
            achieved_goal = obs
        return {
            'observation': obs,
            self.desired_goal_key: desired_goal,
            self.achieved_goal_key: achieved_goal
        }, info

    def step(self, action):
        truncated = False
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except:
            obs, reward, done, info = self.env.step(action)
            terminated = done
        truncated = truncated or self.env.done
        achieved_goal = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)
        desired_goal = self.compute_desired_goal(obs=obs)
        can_dropped = not(achieved_goal['grasped({})'.format(self.obj_to_use)])
        at_drop = achieved_goal['at({},drop)'.format(self.obj_to_use)]
        treshold_distance = self.ray_bins["drop"]
        condition = (achieved_goal['at({},drop)'.format(self.obj_to_use)] <= treshold_distance)
        achieved_goal = self.detector.dict_to_array(achieved_goal)
        #print(condition)
        info = {'can_dropped': can_dropped, 'at_drop': at_drop, 'is_success': condition}
        #print(info)
        # Computes the termination condition (distance to drop off area <= 0.1)
        terminated = terminated or condition
        if self.augmented_obs:
            obs = np.concatenate((obs, achieved_goal))
        if not self.symbolic_goal:
            #object_pos = self.env.sim.data.body_xpos[self.obj_body]
            #achieved_goal = object_pos
            achieved_goal = obs
        reward = self.compute_reward(achieved_goal, desired_goal, info=info)
        #print("Reward: {}".format(reward))
        return {
            'observation': obs,
            self.desired_goal_key: desired_goal,
            self.achieved_goal_key: achieved_goal
        }, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute the reward based on the distance between the object and the target goal
        treshold_distance = self.ray_bins["drop"]
        if not self.symbolic_goal:
            if self.dense_reward:
                try:
                    # Vectorized version (Batch of states through achieved_goal=vector of achieved goals, desired_goal=vector of desired goals and info=vector of info dicts)
                    # Select only the position of the object for each state in the batch
                    achieved_pos = achieved_goal[:, 40:43]
                    desired_pos = desired_goal[:, 40:43]
                    return -np.linalg.norm(achieved_pos-desired_pos, axis=1)
                except:
                    achieved_pos = achieved_goal[40:43]
                    desired_pos = desired_goal[40:43]
                    return -np.linalg.norm(achieved_pos-desired_pos)
            else:
                try:
                    # Vectorized version (Batch of states through achieved_goal=vector of achieved goals, desired_goal=vector of desired goals and info=vector of info dicts)
                    # Select only the position of the object for each state in the batch
                    achieved_pos = achieved_goal[:, 40:43]
                    desired_pos = desired_goal[:, 40:43]
                    reward = -np.float32(np.linalg.norm(achieved_pos-desired_pos, axis=1) > treshold_distance)
                    return reward
                except:
                    achieved_pos = achieved_goal[40:43]
                    desired_pos = desired_goal[40:43]
                    reward = -np.float32(np.linalg.norm(achieved_pos-desired_pos) > treshold_distance)
                    return reward
        groundings_keys = list(self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False).keys())
        can_dropped_index = groundings_keys.index('grasped({})'.format(self.obj_to_use))
        at_drop_index = groundings_keys.index('at({},drop)'.format(self.obj_to_use))
        try:
            # Vectorized version
            can_dropped = achieved_goal[:, can_dropped_index]
            achieved_at_drop = achieved_goal[:, at_drop_index]
            #desired_at_drop = desired_goal[:, at_drop_index]
            desired_at_drop = np.zeros(achieved_at_drop.shape)
            distance = np.array(achieved_at_drop - desired_at_drop)
        except:
            # Non-vectorized version
            can_dropped = info['can_dropped']
            achieved_at_drop = info['at_drop']
            #desired_at_drop = desired_goal[at_drop_index]
            desired_at_drop = 0.0
            distance = np.linalg.norm(achieved_at_drop - desired_at_drop)

        if not self.dense_reward:
            # Create a mask for the conditions (Can dropped or distance to drop off area > 0.1)
            mask_distance = distance < treshold_distance
            # Assign rewards based on the conditions
            rewards = np.where(mask_distance, 1000.0, -1.0)
        else:
            # Create a mask for the condition (Can dropped)
            mask = can_dropped
            # Compute the reward based on the distance to drop off area
            rewards = -distance - 1.0 * mask
    
        return rewards.astype(np.float32)

    def compute_desired_goal(self, obs=None):
        if not self.symbolic_goal:
            #return np.asarray(self.area_pos['drop'])
            obs_copy = deepcopy(obs)
            obs_copy[40:43] = self.area_pos['drop']
            return obs_copy

        # Compute the goal based on the current state of the environment
        gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
        obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.obj_body])
        current_state = self.detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=True)

        # Compute environment distances
        dist_drop_pick = np.linalg.norm(self.area_pos['drop'] - self.area_pos['pick'])
        dist_drop_activate = np.linalg.norm(self.area_pos['drop'] - self.area_pos['activate'])
        dist_pick_activate = np.linalg.norm(self.area_pos['pick'] - self.area_pos['activate'])

        # Compute the desired goal from the current state
        desired_goal = deepcopy(current_state)
        desired_goal[self.goal] = 0.0
        #desired_goal['at({},drop)'.format(self.obj_to_use)] = 0.0
        desired_goal['at({},pick)'.format(self.goal.split(',')[0][3:])] = dist_drop_pick
        #desired_goal['at({},activate)'.format(self.goal.split(',')[0][3:])] = dist_pick_activate
        desired_goal['at_gripper(gripper,pick)'] = dist_drop_pick
        desired_goal['at_gripper(gripper,activate)'] = dist_pick_activate
        desired_goal['at_gripper(gripper,drop)'] = 0.0

        return np.asarray([v for k, v in sorted(desired_goal.items())])