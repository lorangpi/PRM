import numpy as np
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from robosuite.utils.mjcf_utils import find_elements
import matplotlib.pyplot as plt

class RoboSuite_PickPlace_Detector:
    def __init__(self, env, single_object_mode=True, object_to_use='can', grid_size=None):
        self.env = env  # The environment in which the detector operates
        self.object_to_use = object_to_use  # The object to be used in single object mode

        # If in single object mode, only use the specified object and the door. Otherwise, use all objects.
        if single_object_mode:
            self.objects = [object_to_use, 'door']
        else:
            self.objects = ['cereal', 'milk', 'can', 'bread', 'door']

        # ENVIRONMENT SPECIFIC VARIABLES
        # Mapping of object names to their Robosuite IDs
        self.object_id = {'cereal': 'Cereal_main', 'milk': 'Milk_main', 'can': 'Can_main', 'bread': 'Bread_main', 'door': 'Door_main'}
        # Areas where objects can be located
        self.object_areas = ['pick', 'drop']
        # Areas where grippers can be located
        self.grippers_areas = ['pick', 'drop', 'activate', 'lightswitch']
        # List of grippers
        self.grippers = ['gripper']
        # Mapping of area names to their positions in the environment
        self.area_pos = {'pick': env.bin1_pos, 'drop': env.bin2_pos, 'activate': self.env.activate_pos, 'lightswitch': self.env.lightswitch_pos}
        # Size of the area from the environment
        self.area_size = self.env.ray_bins
        # List of active objects
        self.active_objs = [self.env.obj_to_use]
        # Maximum distance for the lidar detection in meters
        self.max_distance = 10 
        self.grid_size = grid_size

        # OBJECT SPECIFIC VARIABLES FOR PDDL GENERATION
        # Mapping of object names to their types
        self.types = {"location":"physobj", "object":"physobj", "gripper":"physobj", "door":"physobj"}  # Types of entities in the environment
        self.obj_types = {obj: "object" for obj in ["can"]}
        self.obj_types.update({"door":"door"})
        self.obj_types.update({area: "location" for area in self.grippers_areas})
        self.obj_types.update({gripper: "gripper" for gripper in self.grippers})

        # Predicate mappings
        self.predicates = {"at": self.at, "at_gripper": self.at_gripper, "grasped": self.grasped, "picked_up": self.picked_up, "dropped_off": self.dropped_off, "open": self.open, "door_locked": self.door_locked, "over": self.over, "at_grab_level": self.at_grab_level, "door_collision": self.door_collision}
        self.shift = {"at": 1.0, "at_gripper": 1.0, "grasped": 1.0, "picked_up": 1.0, "dropped_off": 1.0, "open": 1.0, "door_locked": 1.0, "over": 1.0, "at_grab_level": 1.0, "door_collision": 1.0}
        self.predicate_type = {"at": "num", "at_gripper": "num", "grasped": "bool", "picked_up": "bool", "dropped_off": "bool", "open": "num", "door_locked": "bool", "over": "num", "at_grab_level": "num", "door_collision": "bool", "light_off": "bool", "locked": "bool", "open_gripper": "num"}
        self.useful_predicates = ["at", "at_gripper", "grasped", "open", "locked", "light_off", "open_gripper"]
        self.object_generalization = {'can': True, 'door': False, 'pick': True, 'drop': True, 'activate': False, 'lightswitch': False, 'gripper': True}

    def get_ranges(self, space_dict):
        ranges = []
        for key in sorted(space_dict.keys()):
            key = key.split('(')[0]
            if self.predicate_type[key] == 'num':
                ranges.append((0, self.grid_size//self.max_distance))
            else:
                ranges.append((0, 1))
        return ranges

    def at(self, obj, area, return_distance=False):
        if obj in ['cereal', 'milk', 'can', 'bread']:
            obj_pos = self.env.sim.data.body_xpos[self.env.obj_body]
            if area == 'pick':
                dist = np.linalg.norm(obj_pos - self.area_pos['pick'])
            elif area == 'drop':
                dist = np.linalg.norm(obj_pos - self.area_pos['drop'])
            elif area == 'activate':
                dist = np.linalg.norm(obj_pos - self.area_pos['activate'])
            else:
                return None
        elif obj == 'door':
            return None
        else:
            return None

        if return_distance:
            return dist
        else:
            return bool(dist < self.area_size[area])

    def at_gripper(self, gripper, area, return_distance=False):
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            if area == 'pick':
                dist = np.linalg.norm(gripper_pos - self.area_pos['pick'])
            elif area == 'drop':
                dist = np.linalg.norm(gripper_pos - self.area_pos['drop'])
            elif area == 'activate':
                dist = np.linalg.norm(gripper_pos - self.area_pos['activate'])
            elif area == 'lightswitch':
                dist = np.linalg.norm(gripper_pos - self.area_pos['lightswitch'])
            else:
                raise ValueError('Invalid area.')
        else:
            raise ValueError('Invalid object.')

        if return_distance:
            return dist
        else:
            return bool(dist < self.area_size[area])

    def grasped(self, obj):
        if obj == 'door':
            return None
        active_obj = self.select_object(obj)

        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True

    def picked_up(self, obj, return_distance=False):
        if obj == 'door':
            return None
        active_obj = self.select_object(obj)
        z_target = self.env.bin1_pos[2] + 0.25
        object_z_loc = self.env.sim.data.body_xpos[self.env.obj_body_id[active_obj.name]][2]
        z_dist = z_target - object_z_loc
        #if return_distance:
        #    return z_dist
        #else:
        return bool(z_dist < 0.15)

    def dropped_off(self):
        """
        Returns True if the object is in the correct bin, False otherwise.
        """
        gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        for i, obj in enumerate(self.env.objects):
            obj_str = obj.name
            obj_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj_str]]
            dist = np.linalg.norm(gripper_site_pos - obj_pos)
            r_reach = 1 - np.tanh(10.0 * dist)
            self.env.objects_in_bins[i] = float((not self.env.not_in_bin(obj_pos, i)) and r_reach < 0.6)

        # returns True if a single object is in the correct bin
        if self.env.single_object_mode in {1, 2}:
            return bool(np.sum(self.env.objects_in_bins) > 0)

        # returns True if all objects are in correct bins
        return bool(np.sum(self.env.objects_in_bins) == len(self.env.objects))

    def open(self, obj, return_distance=False):
        """
        Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
        ignoring the caps from the first letter in the self.obj_names.
        """
        if obj == 'door':
            """
            Returns True if the door is open (i.e., unlocked and open), False otherwise.
            """
            # Get the hinge joint element from the XML
            hinge = find_elements(root=self.env.door.worldbody, tags="joint", attribs={"name": self.env.door.hinge_joint}, return_first=True)

            # Check if the hinge joint is unlocked and open
            if self.env.door.lock:
                return 0 if return_distance else False
            else:
                # If the door is not locked, check if the hinge joint is at its minimum position
                # Get the current position of the hinge joint
                qpos = self.env.sim.data.qpos[self.env.sim.model.get_joint_qpos_addr(f"{obj.capitalize()}_hinge")]

                # Get the closed position of the hinge joint
                qpos_min = hinge.get("range").split(" ")[0]

                # Calculate the relative door aperture as a percentage of the range between closed and maximum positions
                qpos_max = hinge.get("range").split(" ")[1]
                relative_aperture = ((float(qpos) - float(qpos_min)) / (float(qpos_max) - float(qpos_min))) * 100
                return relative_aperture / 100 if return_distance else bool(relative_aperture > 10)

        elif obj == 'gripper':
            """
            Returns True if the gripper is open, False otherwise.
            """
            gripper = self.env.robots[0].gripper
            # Print gripper aperture
            left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
            right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
            aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
            #print(f'Gripper aperture: {aperture}')
            return bool(aperture > 0.13)
        return None

    def door_locked(self):
        """
        Returns True if the door is locked, False otherwise.
        """
        if self.env.door.lock and not(self.open('door')):
            return True
        else:
            return False

    def over(self, gripper, obj, return_distance=False):
        """
        Returns True if the gripper is over the object, False otherwise.
        """
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
            if return_distance:
                return dist_xy
            else:
                return bool(dist_xy < 0.02)
        else:
            return None

    def at_grab_level(self, gripper, obj, return_distance=False):
        """
        Returns True if the gripper is at the same height as the object, False otherwise.
        """
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_z = np.linalg.norm(gripper_pos[2] - obj_pos[2])
            if return_distance:
                return dist_z
            else:
                return bool(dist_z < 0.005)
        else:
            return None

    def door_collision(self):
        """
        Returns True if the gripper is colliding with the door, False otherwise.
        """
        active_obj = self.select_object('door')
        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms
        o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        g_group = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        
        return self.env.check_contact(g_group, o_geoms)

    def select_object(self, obj_name):
        """
        Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
        ignoring the caps from the first letter in the self.obj_names.
        """
        obj_name = obj_name.lower().capitalize()
        for obj, name in zip(self.env.objects, self.env.obj_names):
            if name.startswith(obj_name):
                return obj
        return None

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False, as_grid=False):
        
        groundings = {}

        # Check if each object is in each area
        for obj in self.objects:
            if obj == 'door':
                continue
            for area in self.object_areas:
                at_value = self.at(obj, area, return_distance=return_distance)
                if not(self.env.light_on):
                    at_value = self.max_distance if return_distance else False
                #if return_distance:
                #    at_value = at_value / self.max_distance  # Normalize distance
                if binary_to_float:
                    at_value = float(at_value)
                groundings[f'at({obj},{area})'] = at_value

        # Check if the gripper is in each area and if it's free
        for gripper in self.grippers:
            for area in self.grippers_areas:
                at_gripper_value = self.at_gripper(gripper, area, return_distance=return_distance)
                if not(self.env.light_on):
                    at_gripper_value = self.max_distance if return_distance else False
                #if return_distance:
                #    at_gripper_value = at_gripper_value / self.max_distance  # Normalize distance
                if binary_to_float:
                    at_gripper_value = float(at_gripper_value)
                groundings[f'at_gripper({gripper},{area})'] = at_gripper_value

            # Check if the gripper is grasping each object
            for obj in self.objects:
                if obj == 'door':
                    continue
                grasped_value = self.grasped(obj)
                if not(self.env.light_on):
                    grasped_value = False
                if binary_to_float:
                    grasped_value = float(grasped_value)
                groundings[f'grasped({obj})'] = grasped_value

        # Check if the door is open
        door_open_value = self.open('door', return_distance=return_distance)
        if not(self.env.light_on):
            door_open_value = self.max_distance if return_distance else False
        if binary_to_float:
            door_open_value = float(door_open_value)
        groundings['open(door)'] = door_open_value

        # Check if the door is locked
        door_locked_value = self.door_locked()
        if not(self.env.light_on):
            door_locked_value = True
        if binary_to_float:
            door_locked_value = float(door_locked_value)
        groundings['locked(door)'] = door_locked_value

        # Check if the gripper is open
        gripper_open_value = self.open('gripper')
        if binary_to_float:
            gripper_open_value = float(gripper_open_value)
        groundings['open(gripper)'] = gripper_open_value

        # Check if the gripper is colliding with the door
        door_collision_value = self.door_collision()
        if binary_to_float:
            door_collision_value = float(door_collision_value)
        groundings['door_collision'] = door_collision_value

        # Check if an object has been picked up
        for obj in self.objects:
            if obj == 'door':
                continue
            picked_up_value = self.picked_up(obj, return_distance=return_distance)
            #if return_distance:
            #    picked_up_value = picked_up_value / self.max_distance  # Normalize distance
            if binary_to_float:
                picked_up_value = float(picked_up_value)
            groundings[f'picked_up({obj})'] = picked_up_value

        # Check if an object has been dropped off
        dropped_off_value = self.dropped_off()
        if binary_to_float:
            dropped_off_value = float(dropped_off_value)
        groundings['dropped_off'] = dropped_off_value

        # Check if the gripper is over each object
        for gripper in self.grippers:
            for obj in self.objects:
                if obj == 'door':
                    continue
                over_value = self.over(gripper, obj, return_distance=return_distance)
                if not(self.env.light_on):
                    over_value = self.max_distance if return_distance else False
                #if return_distance:
                #    over_value = over_value / self.max_distance
                if binary_to_float:
                    over_value = float(over_value)
                groundings[f'over({gripper},{obj})'] = over_value

        # Check if the gripper is at the same height as each object
        for gripper in self.grippers:
            for obj in self.objects:
                if obj == 'door':
                    continue
                at_grab_level_value = self.at_grab_level(gripper, obj, return_distance=return_distance)
                #if return_distance:
                #    at_grab_level_value = at_grab_level_value / self.max_distance
                if binary_to_float:
                    at_grab_level_value = float(at_grab_level_value)
                groundings[f'at_grab_level({gripper},{obj})'] = at_grab_level_value
        
        if not(self.env.light_on):
            groundings['light_off'] = 1.0 if binary_to_float else True
        else:
            groundings['light_off'] = 0.0 if binary_to_float else False

        if self.grid_size != None and as_grid:
            # Decomposes the state space for each numerical predicate into a grid of grid_size x grid_size i.e., factors all numerical predicates
            # by self.max_distance divided by self.grid_size
            copy_groundings = groundings.copy()
            for predicate in groundings:
                if predicate.split('(')[0] not in self.useful_predicates:
                    # Remove all predicates that are not useful
                    del copy_groundings[predicate]
                elif self.predicate_type[predicate.split('(')[0]] == 'num':
                    copy_groundings[predicate] = int(groundings[predicate] // (self.max_distance / self.grid_size))
            return dict(sorted(copy_groundings.items())) if as_dict else np.asarray([v for k, v in sorted(copy_groundings.items())])
        return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])

    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])

    def get_static_predicates(self, as_dict=True, binary_to_float=False):
        """
        Returns the static predicates of the environment.
        """
        predicates = {}

        # Check if each object is of type "physobj"
        for obj in self.objects:
            if obj == 'door':
                continue
            type_value = self.obj_types[obj]
            if binary_to_float:
                type_value = float(type_value)
            predicates[f'type({obj},physobj)'] = type_value

        return dict(sorted(predicates.items())) if as_dict else np.asarray([v for k, v in sorted(predicates.items())])

    def get_ungrounded_predicates(self, groundings=None):
        predicates = {}
        if groundings is None:
            groundings = self.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)

        for grounding in groundings:
            predicate, *args = grounding.strip().split('(')[0].split(',')
            args[-1] = args[-1].strip(')')

            if predicate not in predicates:
                predicates[predicate] = [self.obj_types[arg] if arg in self.obj_types else arg for arg in args]

        return predicates