import numpy as np
from domain_specific.detector import RoboSuite_PickPlace_Detector
from executor import Executor

class Executor_Pick(Executor):
    def __init__(self, id, mode="Coded", I=None, Beta=None, Circumstance=None, basic=False):
        super().__init__(id, mode, I, Beta, Circumstance, basic)
        self.id = id
        self.I = I
        self.Circumstance = Circumstance
        self.Beta = Beta
        self.basic = basic
        self.mode = mode

    def execute(self, env, operator, render=False, obs=None):
        """
        Sets the environment to a state where the gripper is holding the object
        """
        env.set_gripper(True)
        env.light_on = True
        detector = env.detector
        state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        #print(state)

        self.reset_step_count = 0
        print("Moving up...")
        for _ in range(10):
            env.step([0,0,0.5,0])
            env.render() if render else None

        print("Opening gripper...")
        while not state['open(gripper)']:
            env.step([0,0,0,-0.1])
            env.render() if render else None
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 100:
                return False
        self.reset_step_count = 0
        env.time_step = 0

        print("Moving gripper over object...")
        while not state['over(gripper,{})'.format(env.obj_to_use)]:
            gripper_pos = np.asarray(env.sim.data.body_xpos[env.gripper_body])
            object_pos = np.asarray(env.sim.data.body_xpos[env.obj_body])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            env.step(action)
            env.render() if render else None
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 200:
                return False
        self.reset_step_count = 0
        env.time_step = 0

        print("Moving gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(env.obj_to_use)]:
            gripper_pos = np.asarray(env.sim.data.body_xpos[env.gripper_body])
            object_pos = np.asarray(env.sim.data.body_xpos[env.obj_body])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]])
            env.step(action)
            env.render() if render else None
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 400:
                return False
        self.reset_step_count = 0
        env.time_step = 0

        print("Closing gripper...")
        while not state['grasped({})'.format(env.obj_to_use)]:
            env.step([0,0,0,0.1])
            env.render() if render else None
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 30:
                return False
        self.reset_step_count = 0
        env.time_step = 0

        print("Lifting object...")
        while not state['picked_up({})'.format(env.obj_to_use)]:
            obs = env.step([0,0,0.2,0])
            obs = obs[0]
            env.render() if render else None
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 100:
                return False
        self.reset_step_count = 0
        env.reset_step_count = 0
        env.time_step = 0
        env.set_gripper(False)

        return obs, True


class Executor_Drop(Executor):
    def __init__(self, id, mode="Coded", I=None, Beta=None, Circumstance=None, basic=False):
        super().__init__(id, mode, I, Beta, Circumstance, basic)
        self.id = id
        self.I = I
        self.Circumstance = Circumstance
        self.Beta = Beta
        self.basic = basic
        self.mode = mode

    def execute(self, env, operator, render, obs=None):
        """
        Sets the environment to a state where the gripper is not holding the object
        """
        env.set_gripper(True)
        detector = RoboSuite_PickPlace_Detector(env)
        state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

        self.reset_step_count = 0
        print("Opening gripper...")
        while not state['open(gripper)']:
            obs = env.step([0,0,0,-0.1])
            obs = obs[0]
            env.render() if render else None
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.reset_step_count += 1
            if self.reset_step_count > 100:
                return False
        self.reset_step_count = 0
        env.time_step = 0
        env.set_gripper(False)
        return obs, True