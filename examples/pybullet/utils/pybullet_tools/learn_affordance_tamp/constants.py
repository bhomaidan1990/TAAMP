#!/usr/bin/env python

import os
import glob
import numpy as np

from learn_affordance_tamp.msg import TargetPosition 
#from learn_affordance_tamp.file_util import get_filenames_in_dir

PERCEPTION_METHOD_ARUCO = "aruco"
PERCEPTION_METHOD_POINTCLOUD = "pointcloud"

CONDITION_SIMULATOR = "simulator"
CONDITION_REAL = "real_robot"

ROBOT_PLATFORM_KUKA = "kuka"
ROBOT_PLATFORM_UR5E = "ur"
ROBOT_PLATFORM_BAXTER = "baxter"

TOOL_TYPE_SOURCE     = "tool_source"
TOOL_TYPE_SUBSTITUTE = "tool_substitute"
GOAL_TYPE_SOURCE     = "goal_source"
GOAL_TYPE_SUBSTITUTE = "goal_substitute"

# the pose changes are in the manipulandum frame if not specified
TASK_TYPE_POSE_CHANGE = "pose_change"
TASK_SUBTYPE_POSE_CHANGE_GENERAL = "pose_change_type_general" # e.g., push
TASK_SUBSUBTYPE_POSE_CHANGE_WORLD_FRAME_SPECIFIC = "pose_change_world_frame_specific" # e.g., scoop
TASK_SUBSUBTYPE_POSE_CHANGE_WORLD_FRAME_GENERAL = "pose_change_world_frame_general" # e.g., push. In this case, the goal pose is relatively free in both manipulandum and the world frame
TASK_SUBTYPE_POSE_CHANGE_SPECIFIC = "pose_change_type_specific" # manipulandum frame specific
TASK_TYPE_OTHER = "other"
TASK_GOAL_LENGTH_SPECIFIC = "goal_length_specific"
TASK_GOAL_LENGTH_GENERAL = "goal_length_general"
TASK_GOAL_FRAME_GOAL  = "goal_frame"
TASK_GOAL_FRAME_WORLD = "world_frame"

DIRNAME_ARCHIVE = "archive_{}"

SYMMETRY_TOOL_FILE_NAME = "tools.json"
SYMMETRY_GOAL_FILE_NAME = "goals.json"

OBJECT_TYPE_GOAL = TargetPosition.NAME_GOAL
OBJECT_TYPE_TOOL = TargetPosition.NAME_TOOL

PC_MASTER_INDEX = 0
PC_SUB_INDEX = 1

STAR_1_TEE_TOOL = {}
STAR_1_TEE_TOOL["plunger"] = np.array([[-1., 0., 0., 0.],
                                       [0., -1., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]])
STAR_1_TEE_TOOL["xylo_stick"] = np.array([[0., 0., 1., 0.],
                                          [0., 1., 0., 0.],
                                          [-1., 0., 0., 0.05],
                                          [0., 0., 0., 1.]])
STAR_1_TEE_TOOL["butcher_knife"] = np.array([[-1., 0., 0., 0.05],
                                             [0., 0., -1., 0.],
                                             [0., -1., 0., 0.],
                                             [0., 0., 0., 1.]])
STAR_1_TEE_TOOL["blue_scooper"] = np.array([[-1., 0., 0., 0.1],
                                            [0., 1., 0., 0.],
                                            [0., 0., -1., 0.02],
                                            [0., 0., 0., 1.]])
STAR_1_TEE_TOOL["small_blue_spatula"] = np.array([[0., 0., -1., 0.],
                                                  [0., 1., 0., 0.],
                                                  [1., 0., 0., 0.05],
                                                  [0., 0., 0., 1]])
STAR_1_TEE_TOOL["writing_brush"] = np.array([[0., 0., -1., 0.],
                                             [0., 1., 0., 0.],
                                             [1., 0., 0., 0.05],
                                             [0., 0., 0., 1]])
STAR_1_TEE_TOOL["gavel"] = np.array([[-1., 0., 0., 0.],
                                     [0., 0., -1., 0.],
                                     [0., -1., 0., 0.],
                                     [0., 0., 0., 1.]])  

VISUALIZATION_MASTER = "master"
VISUALIZATION_SUB = "sub"

# temp solution, copied directly from file_util
def get_filenames_in_dir(dir_path, ext=None):
    if ext:
        dir_path += "/*." + ext
    else:
        if not dir_path.endswith("/"):
            dir_path += "/*"
    return [os.path.basename(i) for i in glob.glob(dir_path)]

def get_perception_method():
    return "pointcloud"

def get_package_dir():
    #return "/home/meiying/ros_devel_ws/src/learn_affordance_tamp"
    return "/home/meiying/ros_devel_ws/src/learn_affordance_tamp/tamp"

def get_T_subs_dir():
    return "tamp/examples/pybullet/utils/models"

def get_T_tool_sub_path():
    return os.path.join(get_T_subs_dir(), "tool.json")

def get_T_goal_sub_path():
    return os.path.join(get_T_subs_dir(), "goal.json")

#def get_self_T_tool_sub_path():
    #return os.path.join(get_self_T_subs_dir(), "tool.json")

def get_T_tool_sub_control_path():
    return os.path.join(get_T_subs_dir(), "tool_control.json")

def get_T_goal_sub_control_path():
    return os.path.join(get_T_subs_dir(), "goal_control.json")

def get_learned_data_dir(platform=None):
    if platform is None:
        platform = get_robot_platform()
    
    condition_name = ""
    if is_simulator():
        condition_name = CONDITION_SIMULATOR
    else:
        condition_name = CONDITION_REAL
    
    return os.path.join(get_package_dir(), "learned_samples", platform, condition_name, "pointcloud")

def get_tool_mesh_path(tool_name):
    tool_mesh_path = os.path.join(tool_mesh_dir(), tool_name + ".ply")
    return tool_mesh_path

def get_goal_mesh_path(goal_name):
    goal_mesh_path = os.path.join(goal_mesh_dir(), goal_name + ".ply")
    return goal_mesh_path    

def tool_mesh_dir():
    return os.path.join(get_package_dir(), "pointcloud", "tools")

def goal_mesh_dir():
    return os.path.join(get_package_dir(), "pointcloud", "goals")

def symmetry_dir():
    return os.path.join(get_package_dir(), "pointcloud", "symmetry")

def pointcloud_dir():
    return os.path.join(get_package_dir(), "pointcloud", "raws")

def get_candidate_tools():
    tools = get_filenames_in_dir(tool_mesh_dir(), ext="ply")
    
    return [os.path.splitext(i)[0] for i in tools]

def get_candidate_goals():
    goals = get_filenames_in_dir(goal_mesh_dir(), ext="ply")
    
    return [os.path.splitext(i)[0] for i in goals]

def get_candidate_tasks():
    return ["push", "knock", "stir", "cut", "scoop", "draw", "screw"]

def is_simulator():
    return True

def is_testing():
    return True 

def get_sampling_repeatition():
    return 5

def get_demo_robot_platform():
    return "ur"

def get_robot_platform():
    return "ur"

def _get_symmetry(path_name, name):
    group_symmetry = {}
    with open(path_name, "r") as read_file:
        group_symmetry = json.load(read_file)
    object_symmetry = group_symmetry[name]
    #group_symmetry = file_util.read_variable(path_name, name) # change to this after merge with new_algorithm branch

    return object_symmetry    

def get_tool_symmetry(tool_name):
    path_name = os.path.join(symmetry_dir(), SYMMETRY_TOOL_FILE_NAME)
    
    return _get_symmetry(path_name, tool_name)

def get_goal_symmetry(goal_name):
    path_name = os.path.join(symmetry_dir(), SYMMETRY_GOAL_FILE_NAME)
    
    return _get_symmetry(path_name, goal_name)   