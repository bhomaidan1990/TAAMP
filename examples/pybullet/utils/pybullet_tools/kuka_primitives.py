import time

from itertools import count
import random
import math
import os
import json

import numpy as np
import transformations as tfs
import open3d as o3d

#from .pr2_utils import get_top_grasps
from .utils import get_pose, set_pose, get_movable_joints, \
    set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, pairwise_collisions, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, wait_if_gui, flatten, multiply, \
    grasp_from_body_ee, stable_z, invert, interpolate_poses, Euler, unit_pose, approximate_as_prism, point_from_pose, body_from_end_effector
from.meiying_collision_checking import _decompose_homogeneous_transformation_matrix, _get_pcd, _get_homogeneous_transformation_matrix_from_quaternion, is_pcd_collide

# TODO: deprecate

GRASP_INFO = {
    'top': GraspInfo(lambda body: get_top_grasps(body, under=True, tool_pose=Pose(), max_width=INF,  grasp_length=0),
                     approach_pose=Pose(0.1*Point(z=1))),
}

TOOL_FRAMES = {
    'iiwa14': 'iiwa_link_ee_kuka', # iiwa_link_ee | iiwa_link_ee_kuka
}

DEBUG_FAILURE = False

##################################################

TOOL_POSE = Pose(euler=Euler(pitch=np.pi/2))
MAX_GRASP_WIDTH = np.inf
GRASP_LENGTH = 0.

def get_top_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                   max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    # TODO: rename the box grasps
    center, (w, l, h) = approximate_as_prism(body, body_pose=body_pose)
    reflect_z = Pose(euler=[0, math.pi, 0])
    translate_z = Pose(point=[0, 0, h / 2 - grasp_length])
    translate_center = Pose(point=point_from_pose(body_pose)-center)
    grasps = []
    while True:
        #theta = random.uniform(0., np.pi * 2)
        theta = 0.
        rotate_z = Pose(euler=[0, 0, theta])
        yield multiply(tool_pose, translate_z, rotate_z,
                       reflect_z, translate_center, body_pose)
    #if w <= max_width:
        #for i in range(1 + under):
            #rotate_z = Pose(euler=[0, 0, math.pi / 2 + i * math.pi])
            #grasps += [multiply(tool_pose, translate_z, rotate_z,
                                #reflect_z, translate_center, body_pose)]
    #if l <= max_width:
        #for i in range(1 + under):
            #rotate_z = Pose(euler=[0, 0, i * math.pi])
            #grasps += [multiply(tool_pose, translate_z, rotate_z,
                                #reflect_z, translate_center, body_pose)]
    #return grasps

##################################################

class BodyPose(object):
    num = count()
    def __init__(self, body, pose=None):
        if pose is None:
            pose = get_pose(body)
        self.body = body
        self.pose = pose
        self.index = next(self.num)
    @property
    def value(self):
        return self.pose
    def assign(self):
        set_pose(self.body, self.pose)
        return self.pose
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'p{}'.format(index)


class BodyGrasp(object):
    num = count()
    def __init__(self, body, grasp_pose, approach_pose, robot, link):
        self.body = body
        self.grasp_pose = grasp_pose
        self.approach_pose = approach_pose
        self.robot = robot
        self.link = link
        self.index = next(self.num)
    @property
    def value(self):
        return self.grasp_pose
    @property
    def approach(self):
        return self.approach_pose
    #def constraint(self):
    #    grasp_constraint()
    def attachment(self):
        return Attachment(self.robot, self.link, self.grasp_pose, self.body)
    def assign(self):
        return self.attachment().assign()
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'g{}'.format(index)
    def serialize(self):
        result = {}

        result["class"] = "BodyGrasp"
        result["args"] = ["body", "grasp_pose", "approach_pose", "robot", "link"]
        result["body"] = self.body
        result["grasp_pose"] = tuple((tuple(self.grasp_pose[0]), tuple(self.grasp_pose[1])))
        result["approach_pose"] = tuple((tuple(self.approach_pose[0]), tuple(self.approach_pose[1])))
        result["robot"] = self.robot
        result["link"] = self.link
        
        return result

class BodyConf(object):
    num = count()
    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = get_movable_joints(body)
        if configuration is None:
            configuration = get_joint_positions(body, joints)
        self.body = body
        self.joints = joints
        self.configuration = configuration
        self.index = next(self.num)
    @property
    def values(self):
        return self.configuration
    def assign(self):
        set_joint_positions(self.body, self.joints, self.configuration)
        return self.configuration
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'q{}'.format(index)

class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = get_movable_joints(body)
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments
    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])
    def iterator(self):
        # TODO: compute and cache these
        # TODO: compute bounding boxes as well
        for i, configuration in enumerate(self.path):
            set_joint_positions(self.body, self.joints, configuration)
            for grasp in self.attachments:
                grasp.assign()
            yield i
    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            enable_real_time()
        else:
            disable_real_time()
        for values in self.path:
            for _ in joint_controller(self.body, self.joints, values):
                enable_gravity()
                if not real_time:
                    step_simulation()
                time.sleep(dt)
    # def full_path(self, q0=None):
    #     # TODO: could produce sequence of savers
    def refine(self, num_steps=0):
        return self.__class__(self.body, refine_path(self.body, self.joints, self.path, num_steps), self.joints, self.attachments)
    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)
    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.body, len(self.joints), len(self.path), len(self.attachments))
    def serialize(self):
        result = {}       
        
        result["class"] = "BodyPath"
        result["args"] = ["body", "path", "joints", "attachments"]
        result["body"] = self.body
        result["path"] = self.path
        result["joints"] = self.joints
        result["attachments"] = {}
        for i in range(len(self.attachments)):
            result["attachments"][i] = self.attachments[i].serialize()
        
        return result
        

##################################################

class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link
    def bodies(self):
        return {self.body, self.robot}
    def iterator(self, **kwargs):
        return []
    def refine(self, **kwargs):
        return self
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)

class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        add_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Detach(self.body, self.robot, self.link)
    def serialize(self):
        result = {}
        
        result["class"] = "Attach"
        result["args"] = ["body", "robot", "link"]
        result["body"] = self.body
        result["robot"] = self.robot
        result["link"] = self.link
        
        return result
        
class Detach(ApplyForce):
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Attach(self.body, self.robot, self.link)
    def serialize(self):
        result = {}
        
        result["class"] = "Detach"
        result["args"] = ["body", "robot", "link"]
        result["body"] = self.body
        result["robot"] = self.robot
        result["link"] = self.link
        
        return result    

class Command(object):
    num = count()
    def __init__(self, body_paths):
        self.body_paths = body_paths
        self.index = next(self.num)
    def bodies(self):
        return set(flatten(path.bodies() for path in self.body_paths))
    # def full_path(self, q0=None):
    #     if q0 is None:
    #         q0 = Conf(self.tree)
    #     new_path = [q0]
    #     for partial_path in self.body_paths:
    #         new_path += partial_path.full_path(new_path[-1])[1:]
    #     return new_path
    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step?'.format(i, j)
                wait_if_gui(msg)
                #print(msg)
    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                #time.sleep(time_step)
                wait_for_duration(time_step)
    def control(self, real_time=False, dt=0): # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)
    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs) for body_path in self.body_paths])
    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])
    def __repr__(self):
        index = self.index
        #index = id(self) % 1000
        return 'c{}'.format(index)
    def serialize(self):
        result = {}
        for i in range(len(self.body_paths)):
            result[i] = self.body_paths[i].serialize()
        return result

#######################################################

def get_tool_link(robot):
    return link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])


def get_grasp_gen(robot, grasp_name='top', tool_grasp_pose={}):
    grasp_info = GRASP_INFO[grasp_name]
    tool_link = get_tool_link(robot)
    def gen(body):
        print "[Meiying::get_grasp_gen] sampling grasps of body {}!!!".format(body)
        #grasp_poses = grasp_info.get_grasps(body)
        # TODO: continuous set of grasps
        for grasp_pose in grasp_info.get_grasps(body):      
            Tbodycenter_bodygrasp = ((0., 0., 0.), (0., 0., 0., 1.))
            Tee_bodygrasp = grasp_pose
            if body in tool_grasp_pose.keys():
                Tbodycenter_bodygrasp = tool_grasp_pose[body]
            Tee_bodycenter = multiply(Tee_bodygrasp, invert(Tbodycenter_bodygrasp))
            grasp_pose = Tee_bodycenter
            #if body == pull_tool:
                #Tee_bodygrasp = grasp_pose
                #Tbodycenter_bodygrasp = ((0.1, 0.1, 0.), (0., 0., 0., 1.))
                #Tee_bodycenter = multiply(Tee_bodygrasp, invert(Tbodycenter_bodygrasp))
                #grasp_pose = Tee_bodycenter
            #elif body == push_tool:
                #Tee_bodygrasp = grasp_pose
                #Tbodycenter_bodygrasp = ((-0.1, 0., 0.), (0., 0., 0., 1.))
                #Tee_bodycenter = multiply(Tee_bodygrasp, invert(Tbodycenter_bodygrasp))
                #grasp_pose = Tee_bodycenter
            body_grasp = BodyGrasp(body, grasp_pose, grasp_info.approach_pose, robot, tool_link)
            yield (body_grasp,)
    return gen

def _get_movable_obstacles_from_args(*args):
    movable_bodies = []
    
    if len(args) > 0:
        if len(args) % 2 != 0:
            raise Exception("Must be body, pose pairs, cannot pass body or pose alone")
        movable_bodies = [args[i * 2] for i in range(len(args) / 2)]
        movable_bodies_poses = [args[i * 2 + 1] for i in range(len(args) / 2)]
        for movable_body_pose in movable_bodies_poses:
            movable_body_pose.assign()
    
    return movable_bodies

def get_stable_gen(fixed=[], movables=[]):
    def gen(body, surface, at_pose_fluents=[], *args):
        #current_movables = _get_movable_obstacles_from_args(*args)
        #if body in current_movables:
            #return
        #if len(set(current_movables)) != len(movables) - 1:
            #return        
        #print "[Meiying::kuka_primitives::get_stable_gen] at_pose_fluents:", at_pose_fluents
        print "Meiying::get_stable_gen::body = {}, surface = {}".format(body, surface)
        while True:
            pose = sample_placement(body, surface)
            if (pose is None) or any(pairwise_collision(body, b) for b in fixed):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)
    return gen

def get_ik_fn(robot, fixed=[], teleport=False, num_attempts=10, body_path={}):
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)
    def fn(body, pose, grasp):
        print "Meiying::get_ik_fn"
        obstacles = [body] + fixed
        gripper_pose = end_effector_from_body(pose.pose, grasp.grasp_pose)
        approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)
        for _ in range(num_attempts):
            set_joint_positions(robot, movable_joints, sample_fn()) # Random seed
            # TODO: multiple attempts?
            q_approach = inverse_kinematics(robot, grasp.link, approach_pose)
            if (q_approach is None) or any(pairwise_collision(robot, b, body_path=body_path) for b in obstacles):
                continue
            conf = BodyConf(robot, q_approach)
            q_grasp = inverse_kinematics(robot, grasp.link, gripper_pose)
            if (q_grasp is None) or any(pairwise_collision(robot, b, body_path=body_path) for b in obstacles):
                continue
            if teleport:
                path = [q_approach, q_grasp]
            else:
                conf.assign()
                #direction, _ = grasp.approach_pose
                #path = workspace_trajectory(robot, grasp.link, point_from_pose(approach_pose), -direction,
                #                                   quat_from_pose(approach_pose))
                path = plan_direct_joint_motion(robot, conf.joints, q_grasp, obstacles=obstacles)

                current_body_pose = pose.pose
                positions = [np.array(current_body_pose[0]) + i * np.array([0., 0., 0.01]) for i in range(1, 11)]
                quaternion = current_body_pose[1]
                for position in positions:
                    intermediate_body_pose = BodyPose(body, (position, quaternion))
                    intermediate_body_pose.assign()
                    #print "body_path"
                    #print body_path
                    #raise Exception("Stop!!")                    
                    if pairwise_collisions(body, fixed[1:], body_path=body_path):
                        path = None
                        break                
                if path is None:
                    if DEBUG_FAILURE: wait_if_gui('Approach motion failed')
                    continue
            command = Command([BodyPath(robot, path),
                               Attach(body, robot, grasp.link),
                               BodyPath(robot, path[::-1], attachments=[grasp])])
            return (conf, command)
            # TODO: holding collisions
        return None
    return fn

##################################################

def assign_fluent_state(fluents):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            obstacles.append(o)
            p.assign()
        else:
            raise ValueError(name)
    return obstacles

def get_free_motion_gen(robot, fixed=[], teleport=False, self_collisions=True):
    def fn(conf1, conf2, fluents=[]):
        #import inspect
        #print("in get_free_motion_gen, fluents:", fluents)
        #print("the stack is:")
        #for stack in inspect.stack():
            #print("\t", stack) 
        print "Meiying::get_free_motion_gen"
        assert ((conf1.body == conf2.body) and (conf1.joints == conf2.joints))
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            obstacles = fixed + assign_fluent_state(fluents)
            path = plan_joint_motion(robot, conf2.joints, conf2.configuration, obstacles=obstacles, self_collisions=self_collisions)
            if path is None:
                if DEBUG_FAILURE: wait_if_gui('Free motion failed')
                return None
        command = Command([BodyPath(robot, path, joints=conf2.joints)])
        return (command,)
    return fn


def get_holding_motion_gen(robot, fixed=[], teleport=False, self_collisions=True):
    def fn(conf1, conf2, body, grasp, fluents=[]):
        print "Meiying::get_holding_motion_gen"
        print "conf2:", conf2.configuration
        assert ((conf1.body == conf2.body) and (conf1.joints == conf2.joints))
        if teleport:
            path = [conf1.configuration, conf2.configuration]
        else:
            conf1.assign()
            obstacles = fixed + assign_fluent_state(fluents)
            path = plan_joint_motion(robot, conf2.joints, conf2.configuration,
                                     obstacles=obstacles, attachments=[grasp.attachment()], self_collisions=self_collisions)
            if path is None:
                if DEBUG_FAILURE: wait_if_gui('Holding motion failed')
                return None
        command = Command([BodyPath(robot, path, joints=conf2.joints, attachments=[grasp])])
        return (command,)
    return fn

##################################################

def get_movable_collision_test(body_path={}):
    def test(command, body, pose):
        print "Meiying::get_movable_collision_test"
        if body in command.bodies():
            return False
        pose.assign()
        for path in command.body_paths:
            moving = path.bodies()
            current_body_path = body_path
            if body in moving:
                # TODO: cannot collide with itself
                continue
            #print "current body:", body           
            for _ in path.iterator():
                # TODO: could shuffle this
                collisions = []
                for mov in moving:
                    current_body_path = body_path
                    if body not in body_path.keys():
                        current_body_path = {}                       
                    if mov not in body_path.keys():
                        current_body_path = {}
                    #print "body:", body
                    #print "mov:", mov
                    #print "body_path.keys():", body_path.keys()
                    #print "current_body_path.keys():", current_body_path.keys()                    
                    collisions.append(pairwise_collision(mov, body, body_path=current_body_path))
                if any(collisions):
                    if DEBUG_FAILURE: wait_if_gui('Movable collision')
                    return True
                    
        return False
    return test

######################Tool use control condition############################
def _get_update_manipulandum_pose(start_pose, end_pose):
    x = np.array(end_pose[0]) - np.array(start_pose[0])
    if np.linalg.norm(x) == 0:
        x = np.array([1., 0., 0.])
    else:
        x = x / np.linalg.norm(x)
    z = np.array([0., 0., 1.])
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    
    rotation = np.array([x, y, z]).T
    rotation_matrix = np.identity(4)
    rotation_matrix[:3, :3] = rotation
    
    quaternion = tuple(tfs.quaternion_from_matrix(rotation_matrix))
    
    return (start_pose[0], quaternion), (end_pose[0], quaternion)

def _get_ee_approach_pose(ee_touch_pose, body_start_pose, body_end_pose):
    if body_start_pose == body_end_pose:
        return ee_touch_pose
    
    mov_direction = np.array(body_end_pose[0]) - np.array(body_start_pose[0])
    mov_direction = mov_direction / np.linalg.norm(mov_direction)
    approach_direction = -mov_direction
    approach_offset = -mov_direction * 0.05 # 5 cm
    approach_position = np.array(ee_touch_pose[0]) + approach_offset
    
    return (tuple(approach_position), ee_touch_pose[1])

def get_pull_tool_goal_gen(fixed=[], movable=[]):
    def gen(body, surface, at_pose_fluents=[], *args):
        #current_movables = _get_movable_obstacles_from_args(*args)
        #if body in current_movables:
            #return
        #if len(set(current_movables)) != len(movables) - 1:
            #return        
        #print "[Meiying::kuka_primitives::get_stable_gen] at_pose_fluents:", at_pose_fluents
        print "Meiying::get_tool_goal_gen::body = {}, surface = {}".format(body, surface)
        while True:
            pose = sample_placement(body, surface)
            if (pose is None):
                continue
            position = list(pose[0])
            position[0] = -abs(position[0])
            position[1] = abs(position[1])
            position[2] = stable_z(body, surface)
            r = np.linalg.norm(np.array([position[0], position[1]]))
            if r > 0.707:
                direction = np.array([position[0], position[1]]) / r * 0.707
                position = (direction[0], direction[1], position[2])
            if r < 0.5:
                direction = np.array([position[0], position[1]]) / r * 0.5
                position = (direction[0], direction[1], position[2])                
            print "position sampled: ", position
            pose = (tuple(position), pose[1])
            set_pose(body, pose)
            if any(pairwise_collision(body, b) for b in fixed):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)
            #pose = ((-0.5, 0.5, stable_z(body, surface)), (0., 0., 0., 1.))
            #body_pose = BodyPose(body, pose)
            #yield (body_pose,)
    return gen

def get_push_tool_goal_gen(fixed=[], movable=[], manipulanda_tool_pose={}):
    def gen(body, surface, at_pose_fluents=[], *args):
        #current_movables = _get_movable_obstacles_from_args(*args)
        #if body in current_movables:
            #return
        #if len(set(current_movables)) != len(movables) - 1:
            #return        
        #print "[Meiying::kuka_primitives::get_stable_gen] at_pose_fluents:", at_pose_fluents
        print "Meiying::get_tool_goal_gen::body = {}, surface = {}".format(body, surface)
        while True:
            pose = sample_placement(body, surface)
            if (pose is None):
                continue
            position = list(pose[0])
            #position[0] = -abs(position[0])
            position[1] = abs(position[1])
            position[2] = stable_z(body, surface)
            r = np.linalg.norm(np.array([position[0], position[1]]))
            if r > 0.707:
                direction = np.array([position[0], position[1]]) / r * 0.707
                position = (direction[0], direction[1], position[2])
            if r < 0.5:
                direction = np.array([position[0], position[1]]) / r * 0.5
                position = (direction[0], direction[1], position[2])                
            print "position sampled: ", position
            pose = (tuple(position), pose[1])
            set_pose(body, pose)
            if any(pairwise_collision(body, b) for b in fixed):
                continue
            body_pose = BodyPose(body, pose)
            yield (body_pose,)
            #pose = ((-0.5, 0.5, stable_z(body, surface)), (0., 0., 0., 1.))
            #body_pose = BodyPose(body, pose)
            #yield (body_pose,)
    return gen    

# parameter body_pose1, body_pose2 are just poses tuples
def _get_tool_use_poses(body, tool, body_pose1, body_pose2, contact_pose=np.identity(4), manipulanda_tool_pose={}):
    if tool in manipulanda_tool_pose.keys():
        if body in manipulanda_tool_pose[tool].keys():
            Tbody_updatedpoint = manipulanda_tool_pose[tool][body][0]
            body_pose1 = multiply(body_pose1, Tbody_updatedpoint)
            body_pose2 = multiply(body_pose2, Tbody_updatedpoint)
    
    updated_body_start_pose, updated_body_end_pose = _get_update_manipulandum_pose(body_pose1, body_pose2)
    contact_pose_position, contact_pose_quaternion = _decompose_homogeneous_transformation_matrix(contact_pose)
    contact_pose = (tuple(contact_pose_position), tuple(contact_pose_quaternion))
    tool_pose1 = multiply(updated_body_start_pose, contact_pose)
    tool_pose2 = multiply(updated_body_end_pose, contact_pose)
    
    return tool_pose1, tool_pose2

# parameter body_pose1, body_pose2 are just poses tuples
def _get_tool_use_config(body, tool, body_pose1, body_pose2, tool_grasp, contact_pose=np.identity(4), manipulanda_tool_pose={}):
    tool_pose1, tool_pose2 = _get_tool_use_poses(body, tool, body_pose1, body_pose2, contact_pose=contact_pose, manipulanda_tool_pose=manipulanda_tool_pose)
    
    #print "[Meiying::_get_tool_use_config] tool_pose1:", tool_pose1
    #print "[Meiying::_get_tool_use_config] tool_pose2:", tool_pose2
    
    tool_grasp_pose = tool_grasp.grasp_pose
    ee_pose = end_effector_from_body(tool_pose1, tool_grasp_pose)
    #print "[Meiying::_get_tool_use_config] ee_pose:", ee_pose
    ee_approach_pose = _get_ee_approach_pose(ee_pose, body_pose1, (body_pose2[0], body_pose1[1]))
    #print "[Meiying::_get_tool_use_config] ee_approach_pose:", ee_approach_pose    
    body_grasp_pose = grasp_from_body_ee(body_pose1, ee_pose)
    #print "[Meiying::_get_tool_use_config] body_grasp_pose:", body_grasp_pose
    body_approach_grasp_pose = grasp_from_body_ee(body_pose1, ee_approach_pose)
    #print "[Meiying::_get_tool_use_config] body_approach_grasp_pose:", body_approach_grasp_pose
    contact_pose_grasp = BodyGrasp(body, body_grasp_pose, body_approach_grasp_pose, tool_grasp.robot, tool_grasp.link)
    
    #return tool_body_pose1, tool_body_pose2, contact_pose_grasp
    return tool_pose1, tool_pose2, contact_pose_grasp

def _get_axis_range(extent, n, is_fixed=False):
    if is_fixed:
        return np.ones(n) * extent
    return np.random.uniform(-extent, extent, size=(n, ))

# for grasping purposes
def generate_cube(center, extent, orientation=np.identity(3), n=10000):
    x = abs(extent[0] / 2.)
    y = abs(extent[1] / 2.)
    z = abs(extent[2] / 2.)
    
    front_n = int(4 * x * y * n)
    back_n = int(4 * x * y * n)
    left_n = int(4 * y * z * n)
    right_n = int(4 * y * z * n)
    top_n = int(4 * x * z * n)
    bottom_n = int(4 * x * z * n)
    
    front = np.array([_get_axis_range(x, front_n), _get_axis_range(y, front_n), -_get_axis_range(z, front_n, is_fixed=True)]).T
    back  = np.array([ _get_axis_range(x, back_n),  _get_axis_range(y, back_n),  _get_axis_range(z, back_n, is_fixed=True)]).T
    
    left  = np.array([-_get_axis_range(x, left_n, is_fixed=True),  _get_axis_range(y, left_n),  _get_axis_range(z, left_n)]).T
    right = np.array([ _get_axis_range(x, right_n, is_fixed=True), _get_axis_range(y, right_n), _get_axis_range(z, right_n)]).T
    
    top    = np.array([_get_axis_range(x, top_n),     _get_axis_range(y, top_n, is_fixed=True),    _get_axis_range(z, top_n)]).T
    bottom = np.array([_get_axis_range(x, bottom_n), -_get_axis_range(y, bottom_n, is_fixed=True), _get_axis_range(z, bottom_n)]).T
    
    cube = np.vstack([front, back, left, right, top, bottom])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cube)
    pcd.rotate(orientation)
    pcd.translate(center)
    
    return pcd

def _create_ee_box_pcd(ee_pose):
    center = list(ee_pose[0])
    center[2] += 0.1
    extent = [0.05, 0.05, 0.2]
    
    return generate_cube(center, extent)

def _merge_pcds(pcds):
    points = []
    
    for pcd in pcds:
        points.extend(list(np.asarray(pcd.points)))
    
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.array(points))
    
    return merged_pcd

# contact pose is tool pose in the object frame
def get_tool_motion_gen(robot, fixed=[], teleport=False, num_attempts=10, body_path={}, contact_poses={}, manipulanda_tool_pose={}, tool_put_down_z={}):
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)   
    def fn(body, tool, body_pose1, body_pose2, tool_grasp):
    #def fn(*args):
        #return 1, 2, 3, 4, 5
        #print "~~~~~~~~~~~~~~~~~[Meiying::get_tool_motion_gen] body={}, tool={}, body_pose1={}, body_pose2={}~~~~~~~~~~~~~~~~~".format(body, tool, body_pose1.pose, body_pose2.pose)
        
        if not (tool in contact_poses.keys()):
            return
        if not (body in contact_poses[tool].keys()):
            return
        
        check_tool_body_collision = False
        if tool in manipulanda_tool_pose.keys():
            if body in manipulanda_tool_pose[tool].keys():
                check_tool_body_collision = True        
        
        contact_pose = contact_poses[tool][body]        
        
        body_pose3 = BodyPose(body, (body_pose2.pose[0], body_pose1.pose[1]))
        tool_pose1, tool_pose2, contact_grasp_pose = _get_tool_use_config(body, tool, body_pose1.pose, body_pose3.pose, tool_grasp, contact_pose=contact_pose)

        if check_tool_body_collision:
            body_pcd = _get_pcd(body, body_path, _get_homogeneous_transformation_matrix_from_quaternion(np.array(body_pose1.pose[1]), np.array(body_pose1.pose[0])))
            tool_pcd = _get_pcd(tool, body_path, _get_homogeneous_transformation_matrix_from_quaternion(np.array(tool_pose1[1]), np.array(tool_pose1[0])))
            if is_pcd_collide(body_pcd, tool_pcd):
                return None        

        obstacles = fixed
        obstacles_pcds = [_get_pcd(i, body_path) for i in fixed]
        obstacles_pcd = _merge_pcds(obstacles_pcds)
        
        # approach object
        gripper_pose = end_effector_from_body(body_pose1.pose, contact_grasp_pose.grasp_pose)
        #approach_pose = approach_from_grasp(contact_grasp_pose.approach_pose, gripper_pose) 
        approach_pose = end_effector_from_body(body_pose1.pose, contact_grasp_pose.approach_pose)
        #print "[Meiying::get_tool_motion_gen] gripper_pose: ", gripper_pose
        #print "[Meiying::get_tool_motion_gen] approach_pose: ", approach_pose
        approach_result = _get_straightline_path(robot, movable_joints, 
                                                 sample_fn, 
                                                 contact_grasp_pose.link, 
                                                 approach_pose, 
                                                 gripper_pose, 
                                                 num_attempts=num_attempts, 
                                                 obstacles=obstacles, 
                                                 body_path=body_path,
                                                 teleport=teleport)
        #print "[Meiying::get_tool_motion_gen] approach_result:", approach_result
        if approach_result is None:
            return None
        
        # check collision
        tool_approach_start_pose = body_from_end_effector(approach_pose, tool_grasp.grasp_pose)
        tool_approach_end_pose = body_from_end_effector(gripper_pose, tool_grasp.grasp_pose)
        
        set_pose(tool, tool_approach_start_pose)
        tool_start_collision = any(pairwise_collision(tool, b, body_path=body_path) for b in obstacles)
        if tool_start_collision:
            return None
        set_pose(tool, tool_approach_end_pose)
        tool_end_collision = any(pairwise_collision(tool, b, body_path=body_path) for b in obstacles)
        if tool_end_collision:
            return None
        
        if is_pcd_collide(obstacles_pcd, _create_ee_box_pcd(gripper_pose)):
            return None
        if is_pcd_collide(obstacles_pcd, _create_ee_box_pcd(approach_pose)):
            return None        
        
        approach_q_start, approach_path, approach_q_end = approach_result
        
        # tool_use, pull or push
        tool_start_pose = gripper_pose
        tool_end_pose = end_effector_from_body(body_pose3.pose, contact_grasp_pose.grasp_pose)
        tool_result = _get_straightline_path(robot, movable_joints,
                                             sample_fn, 
                                             contact_grasp_pose.link, 
                                             tool_start_pose, 
                                             tool_end_pose, 
                                             q_start=approach_q_end,
                                             num_attempts=num_attempts, 
                                             obstacles=obstacles, 
                                             body_path=body_path,
                                             teleport=teleport)
        #print "[Meiying::get_tool_motion_gen] tool_result:", tool_result
        if tool_result is None:
            return None
        
        if is_pcd_collide(obstacles_pcd, _create_ee_box_pcd(tool_start_pose)):
            return None
        if is_pcd_collide(obstacles_pcd, _create_ee_box_pcd(tool_end_pose)):
            return None 
        
        for ee_pose in interpolate_poses(tool_start_pose, tool_end_pose):
            if is_pcd_collide(obstacles_pcd, _create_ee_box_pcd(ee_pose)):
                return None          
        
        tool_q_start, tool_path, tool_q_end = tool_result
        
        # put tool down
        put_start_pose = tool_end_pose
        # stable z gives a weird result, so just hard code the z for now
        tool_z = -0.025000000366475428
        if tool in tool_put_down_z.keys():
            tool_z = tool_put_down_z[tool]        
        tool_put_down_pose = ((tool_pose2[0][0], tool_pose2[0][1], tool_z), tool_pose2[1])
        put_end_pose = end_effector_from_body(tool_put_down_pose, tool_grasp.grasp_pose)
        put_result = _get_straightline_path(robot, movable_joints,
                                            sample_fn, 
                                            contact_grasp_pose.link, 
                                            put_start_pose, 
                                            put_end_pose, 
                                            q_start=tool_q_end,
                                            num_attempts=num_attempts, 
                                            obstacles=obstacles, 
                                            body_path=body_path,
                                            teleport=teleport)
        
        if put_result is None:
            return None
        
        put_q_start, put_path, put_q_end = put_result
        
        lift_start_pose = put_end_pose
        lift_end_pose = ((put_end_pose[0][0], put_end_pose[0][1], put_end_pose[0][2] + 0.1), put_end_pose[1])
        lift_result = _get_straightline_path(robot, movable_joints,
                                             sample_fn, 
                                             contact_grasp_pose.link, 
                                             lift_start_pose, 
                                             lift_end_pose, 
                                             q_start=put_q_end,
                                             num_attempts=num_attempts, 
                                             obstacles=obstacles, 
                                             body_path=body_path,
                                             teleport=teleport)
        
        if lift_result is None:
            return None
        
        lift_q_start, lift_path, lift_q_end = lift_result
        
        # get return parameters
        command = Command([BodyPath(robot, approach_path, attachments=[tool_grasp]),
                           Attach(body, robot, contact_grasp_pose.link),
                           BodyPath(robot, tool_path, attachments=[contact_grasp_pose, tool_grasp]),
                           Detach(body, robot, contact_grasp_pose.link),
                           BodyPath(robot, put_path, attachments=[tool_grasp]),
                           Detach(tool, robot, tool_grasp.link),
                           BodyPath(robot, lift_path)])
        start_conf = BodyConf(robot, approach_q_start)
        end_conf = BodyConf(robot, lift_q_end)
        tool_final_body_pose = BodyPose(tool, tool_put_down_pose)
        
        #print "start_conf:"
        #print approach_q_start

        return body_pose3, tool_final_body_pose, start_conf, end_conf, command
        
    return fn

#def _get_straightline_interpolate_points(start_pose, end_pose, interval = 0.01): # interval: 1 cm
    #start_position = np.array(start_pose[0])
    #end_position = np.array(end_pose[0])
    #difference = end_position - start_position
    #distance = np.linalg.norm(difference)
    
    #step_size = interval
    #num_steps = int(distance / step_size)
    #step = difference / float(num_steps)
    
    #positions = [np.array(start_position) + i * step for i in range(num_steps)]
    #positions.append(np.array(end_position))
    
    ##print "start:", start_pose[0]
    ##print "end:",end_pose[0]
    ##print "positions:"
    ##for position in positions:
        ##print position
    
    ##raise Exception("Stop!!")
    
    #poses = []
    #quaternion = start_pose[1]
    #for position in positions:
        #poses.append((tuple(position), tuple(quaternion)))
    
    #return poses
    
#def _get_straightline_mini_step_ik(robot, movable_joints, sample_fn, ee_link, ee_start, ee_end, q_start=None, num_attempts=10, obstacles=[], body_path={}):
    #for _ in range(num_attempts):
        #if q_start is None:
            #set_joint_positions(robot, movable_joints, sample_fn()) # Random seed
            #q_start = inverse_kinematics(robot, ee_link, ee_start)
        
        #if (q_start is None) or any(pairwise_collision(robot, b, body_path=body_path) for b in obstacles):
            ##if q_start is None:
                ##print "[Meiying::_get_straightline_path] q_start is None" 
            #continue        
        #conf = BodyConf(robot, q_start)
        #set_joint_positions(robot, movable_joints, q_start) # set current seed
        #q_end = inverse_kinematics(robot, ee_link, ee_end)
        #if (q_end is None) or any(pairwise_collision(robot, b, body_path=body_path) for b in obstacles):
            ##if q_start is None:
                ##print "[Meiying::_get_straightline_path] q_end is None"             
            #continue

        #conf.assign()
        #path = plan_direct_joint_motion(robot, conf.joints, q_end, obstacles=obstacles)
        #if path is None:
            #if DEBUG_FAILURE: wait_if_gui('Approach motion failed')
            #continue
        #return q_start, path, q_end
    #return None

#def _get_straightline_path(robot, movable_joints, sample_fn, ee_link, ee_start, ee_end, q_start=None, num_attempts=10, obstacles=[], body_path={}, teleport=False):
    #return _get_straightline_mini_step_ik(robot,
                                          #movable_joints, 
                                          #sample_fn, 
                                          #ee_link, 
                                          #ee_start, 
                                          #ee_end,
                                          #q_start=None, 
                                          #num_attempts=num_attempts,
                                          #obstacles=obstacles,
                                          #body_path=body_path)    
    
    
    ##if teleport:
        ##return _get_straightline_mini_step_ik(robot,
                                              ##movable_joints, 
                                              ##sample_fn, 
                                              ##ee_link, 
                                              ##ee_start, 
                                              ##ee_end,
                                              ##q_start=None, 
                                              ##num_attempts=num_attempts,
                                              ##obstacles=obstacles,
                                              ##body_path=body_path)
    
    ##q_start = None
    ##q_current = None
    ##q_end = None
    ##path = []
    ##previous_pose = ee_start
    ###waypoints = _get_straightline_interpolate_points(ee_start, ee_end, interval=2.0)
    
    ##for point in waypoints[1:]:
        ##result = _get_straightline_mini_step_ik(robot,
                                                ##movable_joints, 
                                                ##sample_fn, 
                                                ##ee_link, 
                                                ##previous_pose, 
                                                ##point,
                                                ##q_start=q_current, 
                                                ##num_attempts=num_attempts,
                                                ##obstacles=obstacles,
                                                ##body_path=body_path)
        
        ##if result is None:
            ##return None
        
        ##q_current_start, q_current_path, q_current_end = result
        
        ##if q_start is None:
            ##q_start = q_current_start
        ##q_current = q_current_start
        ##q_end = q_current_end
        ##path.extend(q_current_path)
    
    ##return q_start, path, q_end

def _get_straightline_path(robot, movable_joints, sample_fn, ee_link, ee_start, ee_end, q_start=None, num_attempts=10, obstacles=[], body_path={}, teleport=False):
    for _ in range(num_attempts):
        # TODO: multiple attempts?
        if q_start is None:
            set_joint_positions(robot, movable_joints, sample_fn()) # Random seed
            q_start = inverse_kinematics(robot, ee_link, ee_start)
        if (q_start is None) or any(pairwise_collision(robot, b, body_path=body_path) for b in obstacles):
            #if q_start is None:
                #print "[Meiying::_get_straightline_path] q_start is None" 
            continue
        conf = BodyConf(robot, q_start)
        set_joint_positions(robot, movable_joints, q_start) # set current seed
        q_end = inverse_kinematics(robot, ee_link, ee_end)
        if (q_end is None) or any(pairwise_collision(robot, b, body_path=body_path) for b in obstacles):
            #if q_start is None:
                #print "[Meiying::_get_straightline_path] q_end is None"           
            continue
        if teleport:
            path = [q_start, q_end]
        else:
            conf.assign()
            path = plan_direct_joint_motion(robot, conf.joints, q_end, obstacles=obstacles)
            if path is None:
                if DEBUG_FAILURE: wait_if_gui('Approach motion failed')
                continue
        return q_start, path, q_end
    return None

def get_cfree_tool_pose_test(collisions=True, body_path={}):
    def test(body, body_pose1, body_pose3, tool, obstacle, obstacle_pose, grasp_pose):
        print "[Meiying::get_cfree_tool_pose_test]"
        if not collisions or obstacle == body or obstacle == tool:
            return True
        obstacle_pose.assign()
        
        for body_pose in interpolate_poses(body_pose1.pose, body_pose3.pose):
            set_pose(body, body_pose)
            if pairwise_collision(body, obstacle, body_path=body_path):
                return False 
        
        tool_pose1, tool_pose2, _ = _get_tool_use_config(body, tool, body_pose1.pose, body_pose3.pose, grasp_pose)
        for tool_pose in interpolate_poses(tool_pose1, tool_pose2):
            set_pose(tool, tool_pose)
            if pairwise_collision(tool, obstacle, body_path=body_path):
                return False 
        
        return True
    return test

def get_cfree_tool_traj_test(body_path={}):
    def test(body, tool, command, obstacle, obstacle_pose2):
        print "[Meiying::get_cfree_tool_traj_test]"
        if obstacle in command.bodies() or obstacle == body or obstacle == tool:
            return True
        return not get_movable_collision_test(body_path=body_path)(command, obstacle, obstacle_pose2)
    return test

def retrieve_command(current_directory, file_name_index):
    command = None
    file_path = os.path.join(current_directory, "{}.json".format(file_name_index))
    print "file_path:", file_path
    
    data = {}
    try:
        with open(file_path) as f:
            data = json.load(f)
    except:
        return command

    paths = []
    steps = [int(i) for i in data.keys()]
    steps.sort()
    steps = [str(i) for i in steps]
    
    for i in steps:
        current_command = None
        
        current_path = data[i]
        current_path_class = current_path["class"]
        if current_path_class == "BodyPath":
            body = current_path["body"]
            path = current_path["path"]
            joints = current_path["joints"]
            attachments = []
            for key, value in current_path["attachments"].items():
                attachment = BodyGrasp(value["body"], value["grasp_pose"], value["approach_pose"], value["robot"], value["link"])
                attachments.append(attachment)
            current_command = BodyPath(body, path, joints, attachments)
        elif current_path_class == "Attach":
            current_command = Attach(current_path["args"][0], current_path["args"][1], current_path["args"][2])
        elif current_path_class == "Detach":
            current_command = Detach(current_path["args"][0], current_path["args"][1], current_path["args"][2])
        
        if not current_command is None:
            paths.append(current_command)
    
    return Command(paths)