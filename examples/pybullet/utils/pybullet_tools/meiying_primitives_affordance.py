#!/usr/bin/env python

import os
import socket
from timeit import default_timer as timer
import copy
import json
import time
import transformations as tfs
import random

import open3d as o3d
import pybullet as p
import numpy as np

#from .utils import create_box, pairwise_collision

from .utils import get_pose, set_pose, get_movable_joints, \
    set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
    enable_gravity, get_refine_fn, wait_for_duration, link_from_name, get_body_name, sample_placement, \
    end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
    inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
    step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, wait_if_gui, flatten, \
    body_from_end_effector, get_model_path, BB_INFO, multiply, invert

from .kuka_primitives import Command, BodyPose, BodyConf, BodyPath, BodyGrasp, Attach, assign_fluent_state, DEBUG_FAILURE, get_stable_gen, \
     _get_straightline_path, Detach, stable_z, GRASP_INFO, get_tool_link, _get_ee_approach_pose, grasp_from_body_ee

from .bounding_box import BoundingBox3D
from .meiying_collision_checking import _is_point_in_bb, _is_obj_collide, _is_pick_obj_env_collide, is_pcd_collide

from .learn_affordance_tamp import constants, tool_usage

from .learn_affordance_tamp import transformation_util

FLOOR = "floor"
CEILING = "ceiling"
WALL_FRONT = "wall_front"
WALL_BACK = "wall_back"
WALL_LEFT = "wall_left"
WALL_RIGHT = "wall_right"

POINTCLOUD = "pcd"
BOUNDINGBOX = "bb"
POSE = "pose"
WALL_THICKNESS = 0.015
WORKSPACE_HEIGHT = 1.
WORKSPACE_WIDTH = 2.
WORKSPACE_LENGTH = 2.

# data related
PATHS = "paths"
ENCLOSED = "is_enclosed"

CENTER = "center"
R = "r"
EXTENT = "extent"

# for place objects
BODY = "body"
SURFACE = "surface"
MOVABLES = "movables" # this name is misleading. it should be any objects, including fixed objects on the surface
ALL_AREA_FEASIBLE = "all_area_feasible"
FEASIBLE_POSITIONS = "feasible_positions"
PLACEABLE = "placeable"

# for pull objects
# BODY = "body" # this is the same as place objects
BODY_POSE = "pose"
TOOL = "tool"
OBSTACLES = "obstacles" # this name is misleading. it should be any objects, including fixed objects on the surface
ALL_DIRECTION_FEASIBLE = "all_direction_feasible"
#FEASIBLE_END_POSES = "feasible_positions"
PULLABLE = "pullable"

ROBOT_RANGE = [0.45, 0.55 * np.sqrt(2)]

WORLD_FRAME = "world"
BODY_FRAME = "body"

def _get_custom_bb(pcd, o3d_bb, o3d_bb_extent, custom_bb_axis=np.identity(3)):
    sorted_extent = copy.deepcopy(list(o3d_bb_extent))
    sorted_extent.sort()
    min_extent, mid_extent, max_extent = sorted_extent[0], sorted_extent[1], sorted_extent[2]
    chosen_extent = [min_extent]
    if min_extent / mid_extent > 0.9:
        chosen_extent.append(mid_extent)
    if mid_extent / max_extent > 0.9:
        chosen_extent.append(max_extent)
    
    norm_axis_indices = [list(o3d_bb_extent).index(i) for i in chosen_extent]
    projection_axis_indices = []
    all_indices = [i for i in range(len(o3d_bb_extent))]
    for norm_axis_index in norm_axis_indices:
        projection_axis_indices.append([axis for axis in all_indices if axis != norm_axis_index])
    
    bbs = []
    bbs_vol = []
    for i in range(len(norm_axis_indices)):
        norm_axis_index = norm_axis_indices[i]
        projection_axis_index = projection_axis_indices[i]
        custom_bb = BoundingBox3D(np.asarray(pcd.points))
        custom_bb.set_axis(custom_bb_axis)
        custom_bb.set_projection_axis(projection_axis_index, norm_axis_index)
        custom_bb_vol = custom_bb.volumn()
        # convert to open3d bb
        center = custom_bb.get_center()
        rotation = custom_bb.normalized_axis
        extent = custom_bb.norms
        bb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        # get results
        bbs.append(bb)
        bbs_vol.append(custom_bb_vol)
    
    return bbs, bbs_vol

def _get_bb(pcd):
    o3d_obb = pcd.get_oriented_bounding_box()
    o3d_obb_vol = o3d_obb.volume()
    o3d_obb_extent = o3d_obb.extent
    
    o3d_aabb = pcd.get_axis_aligned_bounding_box()
    o3d_aabb_vol = o3d_aabb.volume()
    o3d_aabb_extent = o3d_aabb.get_extent()
    o3d_aabb_center = o3d_aabb.get_center()
    o3d_aabb_R = np.identity(3)
    o3d_aabb = o3d.geometry.OrientedBoundingBox(o3d_aabb_center, o3d_aabb_R, o3d_aabb_extent)    
    
    bbs = [o3d_obb, o3d_aabb]
    bbs_vol = [o3d_obb_vol, o3d_aabb_vol]
    
    custom_obbs, custom_obbs_vol = _get_custom_bb(pcd, o3d_obb, o3d_obb_extent, custom_bb_axis=o3d_obb.R)
    custom_aabbs, custom_aabbs_vol = _get_custom_bb(pcd, o3d_aabb, o3d_aabb_extent)
    
    bbs.extend(custom_obbs)
    bbs.extend(custom_aabbs)
    
    bbs_vol.extend(custom_obbs_vol)
    bbs_vol.extend(custom_aabbs_vol)
    
    min_vol = min(bbs_vol)
    min_vol_index = bbs_vol.index(min_vol)
    
    return bbs[min_vol_index]

def _get_bb_along_axis(pcd, norm_axis_index, projection_axis_index):
    custom_bb = BoundingBox3D(np.asarray(pcd.points))
    custom_bb.set_axis()
    custom_bb.set_projection_axis(projection_axis_index, norm_axis_index[0])
    
    # convert to open3d bb
    center = custom_bb.get_center()
    rotation = custom_bb.normalized_axis
    extent = custom_bb.norms
    bb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
    
    return bb

# TODO: segment it here, or read in the segmentation from pre-processed files
def _get_bbs(pcd):
    return [_get_bb(pcd)]

def _get_preprocessed_bbs():
    bb_path = get_model_path(BB_INFO)
    if os.path.exists(bb_path):
        with open(bb_path) as f:
            return json.load(f) 
    return None

def _get_body_bbs(body_id, body_path, transformation=None):
    bbs_info = _get_preprocessed_bbs()
    pcd_abs_path = _get_pcd_path(body_id, body_path)
    pcd_bbs_info = bbs_info[pcd_abs_path]

    if transformation is None:
        pose = _get_pose_from_id(body_id)
        position = list(pose[0])
        quaternion = list(pose[1])
        transformation = _get_homogeneous_transformation_matrix_from_quaternion(quaternion, position)        
    
    bbs = []
    #temp = [] # TODO: delete
    
    for pcd_bb_info in pcd_bbs_info:
        center = np.array(pcd_bb_info[CENTER])
        rotation = np.array(pcd_bb_info[R])
        extent = np.array(pcd_bb_info[EXTENT])
        bb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        bb = _transform_bb(bb, transformation)     
        bbs.append(bb)
        
        #temp.append(bb) # TODO:delete
    
    #if body_id == 6: # TODO: delete
        #print "current transformation:"
        #print transformation
        #temp.append(_get_pcd(body_id, body_path, transformation=transformation))
        #import inspect
        #for line in inspect.stack():
            #print "\t", line
        #o3d.visualization.draw_geometries(temp)
    
    return bbs

def _rotate_point(rotation, point):
    rotated_point = np.matmul(rotation, np.array([point]).T).T[0]
    
    return rotated_point

#def _is_point_in_bb(bb, point):
    #min_bound = bb.get_min_bound()
    #max_bound = bb.get_max_bound()
    #rotation = bb.R
    
    #rotation_inv = np.linalg.inv(rotation)
    
    #min_bound_normalized = _rotate_point(rotation_inv, min_bound)
    #max_bound_normalized = _rotate_point(rotation_inv, max_bound)
    #point_normalized = _rotate_point(rotation_inv, point)
    
    #for i in range(len(min_bound_normalized)):
        #if point_normalized[i] < min(min_bound_normalized[i], max_bound_normalized[i]) or point_normalized[i] > max(min_bound_normalized[i], max_bound_normalized[i]):
            #return False
    
    #return True

## this one includes contain
#def _is_bb_collide(bb1, bb2):
    #bb1_half_diagonal = np.linalg.norm(bb1.extent) / 2.
    #bb2_half_diagonal = np.linalg.norm(bb2.extent) / 2.
    
    #center_distance = np.linalg.norm(bb1.get_center() - bb2.get_center())
    
    #if center_distance >= bb1_half_diagonal + bb2_half_diagonal:
        #return False
    
    #if _is_point_in_bb(bb2, bb1.get_center()) or _is_point_in_bb(bb1, bb2.get_center()):
        #return True
    
    #for point in np.asarray(bb1.get_box_points()):
        #if _is_point_in_bb(bb2, point):
            #return True
    
    #for point in np.asarray(bb2.get_box_points()):
        #if _is_point_in_bb(bb1, point):
            #return True
    
    #return False

# check whether bb1 is contained in bb2
def _is_bb_contain(bb1s, bb2):
    for bb1 in bb1s:
        for point in np.asarray(bb1.get_box_points()):
            if not _is_point_in_bb(bb2, point):
                return False
    
    return True

def _is_bb_separate(bb1s, bb2):
    for bb1 in bb1s:
        for point in np.asarray(bb1.get_box_points()):
            if _is_point_in_bb(bb2, point):
                return False        
    
    return True

#def _is_obj_collide(body1_pcd, body2_bbs):
    #body_1_pcd_bb_vol = body1_pcd.get_oriented_bounding_box().volume()
    
    #for bb in body2_bbs:
        #cropped_pcd = body1_pcd.crop(bb)
        ## TODO
        
        #if not cropped_pcd.is_empty():
            #cropped_pcd_bb = cropped_pcd.get_oriented_bounding_box()
            #cropped_pcd_bb_vol = cropped_pcd_bb.volume()
            #overlap_percentage = cropped_pcd_bb_vol / body_1_pcd_bb_vol
            ##print "[Meiying::meiying_primitives::_is_obj_collide] vol:", cropped_pcd_bb_vol
            ##print "[Meiying::meiying_primitives::_is_obj_collide] percentage vol:", overlap_percentage
            ##print "[Meiying::meiying_primitives::_is_obj_collide] number of points:", len(np.asarray(cropped_pcd.points))
            ##body1_pcd.paint_uniform_color(np.array([0., 1., 0.]))
            ##cropped_pcd.paint_uniform_color(np.array([1., 0., 0.]))
            ##o3d.visualization.draw_geometries([body1_pcd, bb, cropped_pcd])
            ##o3d.visualization.draw_geometries([bb, cropped_pcd])
            #if overlap_percentage > 0.95:
                #return True
            ##return True
    
    #return False

def _get_ply_path(urdf_path):
    basepath, urdf_filename = os.path.split(urdf_path)
    filename = urdf_filename.split(".")
    base_filename = filename[0]
    ply_filename = "{}.ply".format(base_filename)
    ply_path = os.path.join(basepath, ply_filename)
    
    return ply_path

def _get_homogeneous_transformation_matrix_from_quaternion(rotation_quaternion, translation_vector):
    # translation_vector: np.array([1, 2, 3])
    alpha, beta, gamma = tfs.euler_from_quaternion(rotation_quaternion)
    rotation_matrix = tfs.euler_matrix(alpha, beta, gamma)

    result = rotation_matrix
    result[:3, 3] = translation_vector

    return result

def _get_transformation_matrix_with_rotation_matrix(rotation_matrix, translation_vector):
    # translation_vector: np.array([1, 2, 3])
    result = np.identity(4)
    result[:3, 3] = translation_vector
    result[:3, :3] = rotation_matrix
    return result

def _decompose_homogeneous_transformation_matrix_to_rotation_matrix(matrix):
    translation = np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
    rotation_matrix = matrix[:3, :3]
    return translation, rotation_matrix

def _decompose_homogeneous_transformation_matrix(matrix):
    translation = np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
    # translation.x = matrix[0, 3]
    # translation.y = matrix[1, 3]
    # translation.z = matrix[2, 3]

    quaternion_matrix = matrix.copy()
    quaternion_matrix[:3, 3] = 0
    quaternion = tfs.quaternion_from_matrix(quaternion_matrix)

    rotation = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    # rotation.x = quaternion[0]
    # rotation.y = quaternion[1]
    # rotation.z = quaternion[2]
    # rotation.w = quaternion[3]

    return translation, rotation

def _get_pose_from_id(pybullet_uniqueid):
    return p.getBasePositionAndOrientation(pybullet_uniqueid)

#def _is_same_pose(pose1, pose2):
    #return pose1 == pose2

def _get_pcd_name(pybullet_uniqueid, body_path): 
    return os.path.basename(_get_pcd_path(pybullet_uniqueid, body_path)).split(".")[0]

def _get_pcd_path(pybullet_uniqueid, body_path):
    #print "id: ", pybullet_uniqueid
    #print "body_path:"
    #print body_path
    ply_path = body_path[pybullet_uniqueid]
    if os.path.splitext(ply_path)[1][1:] == "urdf":
        urdf_path = ply_path
        ply_path = get_model_path(_get_ply_path(urdf_path))

    return os.path.abspath(ply_path)

def _get_pcd(pybullet_uniqueid, body_path, transformation=None):
    #print "pybullet_uniqueid:", pybullet_uniqueid
    #print "body_path:", body_path
    ply_path = _get_pcd_path(pybullet_uniqueid, body_path)
    #print "[Meiying::meiying_primitives::_get_pcd] ply_path:", ply_path
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if transformation is None:
        pose = _get_pose_from_id(pybullet_uniqueid)
        position = list(pose[0])
        quaternion = list(pose[1])
        transformation = _get_homogeneous_transformation_matrix_from_quaternion(quaternion, position)
    
    pcd.transform(transformation)

    return pcd

# bb: o3d OrientedBoundingBox. The transform function is not implemented.
def _transform_bb(bb, transformation):
    copied_bb = copy.deepcopy(bb)
    
    position, rotation = _decompose_homogeneous_transformation_matrix_to_rotation_matrix(transformation)
    
    current_center = copied_bb.get_center()
    reformed_center = np.array([[current_center[0], current_center[1], current_center[2], 1.]]).T
    
    reformed_transformed_center = np.matmul(transformation, reformed_center).T[0]
    transformed_center = np.array([reformed_transformed_center[0], reformed_transformed_center[1], reformed_transformed_center[2]])
    
    transformed_rotation = np.matmul(rotation, copied_bb.R)
    
    new_bb = o3d.geometry.OrientedBoundingBox(center=transformed_center, R=transformed_rotation, extent=copied_bb.extent)
    
    return new_bb

# body_bbs are the original ones
def _is_obj_env_collide(body, obstacles, body_bbs, body_pose, obstacles_bbs, body_path, visualize=False):
    #print "[meiying::_is_obj_env_collide] body_pose:", body
    #print "[meiying::_is_obj_env_collide] obstacles:", obstacles
    #print "[meiying::_is_obj_env_collide] body_pose:", body_pose 
    transformed_body_bbs = []
    
    position = list(body_pose[0])
    quaternion = list(body_pose[1])
    body_transformation = _get_homogeneous_transformation_matrix_from_quaternion(quaternion, position)    
    
    for body_bb in body_bbs:
        copied_body_bb = _transform_bb(body_bb, body_transformation)
        transformed_body_bbs.append(copied_body_bb)

    if visualize:
        to_visualize = []
        body_pcd = _get_pcd(body, body_path)
        obstacle_pcd = []
        for i in obstacles:
            if i not in [FLOOR]:
                obstacle_pcd.append(_get_pcd(i, body_path))
        to_visualize = [body_pcd]
        to_visualize.extend(obstacle_pcd)
        for bbs in obstacles_bbs:
            to_visualize.extend(bbs)
        to_visualize.extend(transformed_body_bbs)
        o3d.visualization.draw_geometries(to_visualize, "body {} and obstacles {}".format(body, obstacles))        

    for obstacle_index in range(len(obstacles_bbs)):
        if _is_obj_collide(body, transformed_body_bbs, obstacles[obstacle_index], obstacles_bbs[obstacle_index], body_path, visualize=visualize):
            #print "body {} and obstacle {} collide".format(body, obstacles[obstacle_index])
            #body_pcd = _get_pcd(body, body_path)
            #obstacle_pcd = _get_pcd(obstacles[obstacle_index], body_path)
            #o3d.visualization.draw_geometries([body_pcd, obstacle_pcd])
            return True
    
    return False

# obstacles are a list of uniqueid in pybullet
def _get_obstacles_bbs(obstacles, body_path):
    bbs = []

    for obstacle in obstacles:
        #pcd = _get_pcd(obstacle, body_path)
        #bb = _get_bbs(pcd)
        #bbs.append(bb)
        try:
            bbs.append(_get_body_bbs(obstacle, body_path))
        except KeyError:
            pass
    
    return bbs

# pose: BodyPose
# end_position: (x, y, z)
# this function is for pick feasible test
def _is_obj_straightline_traj_feasible(body, pose, end_position, obstacles, body_path, exclude=[]):
    #start = timer()
    start_position = pose.pose[0]
    quaternion = pose.pose[1]
    
    step_size = 0.01
    difference = np.array(end_position) - np.array(start_position)
    distance = np.linalg.norm(difference)
    num_steps = int(distance / 0.01)
    step = difference / float(num_steps)
    
    positions = [np.array(start_position) + i * step for i in range(num_steps)]
    positions.append(np.array(end_position))
    
    pcds = [_get_pcd(i, body_path) for i in obstacles]
    obstacle_pcd = _merge_pcds(pcds)
    
    is_feasible = True
    previous_min_distance = None
    for position in positions:
        #print "position:", position
        set_pose(body, (position, quaternion))
        body_pcd = _get_pcd(body, body_path)
        is_collide, min_distance = _is_pick_obj_env_collide(body_pcd, obstacle_pcd)
        #o3d.visualization.draw_geometries([body_pcd, obstacle_pcd], "is_collide:{}".format(is_collide))
        if is_collide:
            is_feasible = False
            break
        if previous_min_distance is None:
            previous_min_distance = min_distance
        elif previous_min_distance < min_distance: # this means the body is leaving the group of obstacles
            print "moving away from the group"
            print "previous_min_distance:", previous_min_distance
            print "min_distance:", min_distance
            is_feasible = True
            break
        previous_min_distance = min_distance
    
    return is_feasible 

def _is_obj_traj_feasible(body, pose, S, theta, obstacles, body_path, frame=WORLD_FRAME, exclude=[]):
    #start = timer()
    S = S[0]
    start_position = pose.pose[0]
    quaternion = pose.pose[1]
    start_pose = transformation_util.get_homogeneous_transformation_matrix_from_quaternion(np.array(quaternion), np.array(start_position))
    
    step_size = 0.01
    num_steps = int(theta / 0.01)
    step = theta / float(num_steps)
    
    poses = []
    if frame == WORLD_FRAME:
        poses = [np.matmul(transformation_util.get_transformation_matrix_from_exponential(S, i * step), start_pose) for i in range(1, num_steps)]
        poses.append(np.matmul(transformation_util.get_transformation_matrix_from_exponential(S, theta), start_pose))
    else:
        poses = [np.matmul(start_pose, transformation_util.get_transformation_matrix_from_exponential(S, i * step)) for i in range(1, num_steps)]
        poses.append(np.matmul(start_pose, transformation_util.get_transformation_matrix_from_exponential(S, theta)))        
    
    pcds = [_get_pcd(i, body_path) for i in obstacles]
    obstacle_pcd = _merge_pcds(pcds)
    
    is_feasible = True
    previous_min_distance = None
    for pose in poses:
        position, quaternion = transformation_util.decompose_homogeneous_transformation_matrix(pose)
        #print "position:", position
        position = tuple(position)
        quaternion = tuple(quaternion)
        set_pose(body, (position, quaternion))
        body_pcd = _get_pcd(body, body_path)
        is_collide, min_distance = _is_pick_obj_env_collide(body_pcd, obstacle_pcd)
        if is_collide:
            #if body == 6:
                #o3d.visualization.draw_geometries([body_pcd, obstacle_pcd], "is_collide:{}".format(is_collide))            
            is_feasible = False
            break
        if previous_min_distance is None:
            previous_min_distance = min_distance
        elif previous_min_distance < min_distance: # this means the body is leaving the group of obstacles
            #print "moving away from the group"
            #print "previous_min_distance:", previous_min_distance
            #print "min_distance:", min_distance
            is_feasible = True
            break
        previous_min_distance = min_distance
    
    return is_feasible 

# existing objects, so it is impossible to collide
def _is_aabb_touch(body1_aabb, body2_aabb):
    body1_min_bound = body1_aabb.get_min_bound()
    body1_max_bound = body1_aabb.get_max_bound()
    body1_extent = abs(body1_max_bound - body1_min_bound)
    
    body2_min_bound = body2_aabb.get_min_bound()
    body2_max_bound = body2_aabb.get_max_bound()
    body2_extent = abs(body2_max_bound - body2_min_bound)
    
    center_distance = abs(body1_aabb.get_center() - body2_aabb.get_center())
    
    for i in range(len(center_distance)):
        if (body1_extent[i] / 2. + body2_extent[i] / 2.) * 1.05 < center_distance[i]:
            return False
    
    return True

def _is_pcd_touch(body1_pcd, body2_pcd):
    body1_aabb = body1_pcd.get_axis_aligned_bounding_box()
    body2_aabb = body2_pcd.get_axis_aligned_bounding_box()
    
    #if _is_aabb_touch(body1_aabb, body2_aabb):
        #o3d.visualization.draw_geometries([body1_pcd, body2_pcd, body1_aabb, body2_aabb], "touches!!!!")
    
    return _is_aabb_touch(body1_aabb, body2_aabb)

# extent: half of the actual extent
def _get_axis_range(extent, n, is_fixed=False):
    if is_fixed:
        return np.ones(n) * extent
    return np.random.uniform(-extent, extent, size=(n, ))

# 10000/100000
# test: if this doesn't work, set n=100000
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

def _get_quaternion_from_rotation_matrix(matrix):
    result = np.identity(4)
    result[:3, :3] = matrix
    return tfs.quaternion_from_matrix(result)

# Bodies is a list of body_id:{"pcd":o3d_pcd, "bb":[o3d_obb]}
# top is 0.
def _get_floor(bodies):
    bodies_pcds = []
    
    for body, body_info in bodies.items():
        bodies_pcds.append(body_info[POINTCLOUD])
        
    bodies_pcd = _merge_pcds(bodies_pcds)
    body_o3d_obb = _get_bb_along_axis(bodies_pcd, [2], [0, 1])
    body_o3d_obb_center = body_o3d_obb.get_center()
    body_o3d_obb_extent = body_o3d_obb.extent
    
    floor_height = WALL_THICKNESS
    
    center = np.array([body_o3d_obb_center[0], body_o3d_obb_center[1], - floor_height / 2.])
    extent = np.array([body_o3d_obb_extent[0], body_o3d_obb_extent[1], floor_height])
    rotation = body_o3d_obb.R
    
    return _get_box_info(FLOOR, center, extent, orientation=rotation)

def _get_pcds_aabb(pcds):
    bodies_points = []
    
    for pcd in pcds:
        bodies_points.extent(list((pcds.points)))
        
    bodies_pcd = o3d.geometry.PointCloud()
    bodies_pcd.points = o3d.Vector3dVector(np.array(bodies_points)) 
    
    return bodies_pcd.get_axis_aligned_bounding_box()

def _get_box_info(name, center, extent, orientation=np.identity(3)):
    info = {}
    pcd = generate_cube(center, extent, orientation=orientation)
    pose = (tuple(center), tuple(_get_quaternion_from_rotation_matrix(orientation)))
    bb = o3d.geometry.OrientedBoundingBox(center = center, R = orientation, extent = extent)
    
    return _group_info(name, pcd, pose, [bb])

def _group_info(name, pcd, pose, bb):
    info = {}
    info[POINTCLOUD] = pcd
    info[POSE] = pose
    info[BOUNDINGBOX] = bb    
    
    return info

# initial_index_grouping = [[1, 2, 3], [4, 5], [6]]
# grouping: [{body_id:{"pcd":o3d_pcd, "bb":[o3d_obb, ...]}, ...}, {}, {}, ...]
# body_id can be index if it is ceilings, walls, or floors
def group_initial_bodies(body_path, initial_index_grouping, fixed=[]):
    grouping = []
    floor_id = fixed[0] # the first one is always the floor
    to_add_boundary = {}
    
    # grouping initially is
    # [[{body_id:{"pcd":o3d_pcd, "bb":[o3d_obb, ...]}, {}...], [{}], [{}], ...]
    for each_group in initial_index_grouping:
        grouping.append([{}])
        for body_index in each_group:
            body_pcd = _get_pcd(body_index, body_path)
            body_pose = _get_pose_from_id(body_index)
            #body_bb = _get_bbs(body_pcd)
            body_bb = _get_body_bbs(body_index, body_path)
            grouping[-1][0][body_index] = _group_info(body_index, body_pcd, body_pose, body_bb)
        
        for body_index in each_group:
            body_aabb = grouping[-1][0][body_index][POINTCLOUD].get_axis_aligned_bounding_box()
            body_aabb_extent = body_aabb.get_extent()
            if body_aabb_extent[2] >= WORKSPACE_HEIGHT:
                to_add_boundary[body_index] = grouping.pop()
                continue

        grouping[-1][0][FLOOR] = _get_floor(grouping[-1][0])
    
    for body_index, bodies_info in to_add_boundary.items():
        body_aabb_extent = body_info[body_index][POINTCLOUD].get_axis_aligned_bounding_box()
        body_aabb_extent = body_aabb.get_extent()
        if body_aabb_extent[0] >= WORKSPACE_LENGTH:
            # this is the back, get the front boundary
            grouping.append({})
            grouping[-1][0][body_index] = bodies_info[body_index]
            for other_body in bodies_info.keys():
                body_pose = bodies_info[body_index][POSE]
                if other_body != body_index:
                    other_body_pose = bodies_info[other_body][POSE]
                    if other_body_pose[0][1] < body_pose[0][1]:
                        grouping[-1][0][other_body] = bodies_info[other_body]
            
            y = max(body_aabb.get_min_bound([1]), body_aabb.get_max_bound([1]))
            
            front_center = np.array([0., -WORKSPACE_WIDTH / 2., WORKSPACE_HEIGHT / 2.])
            front_extent = np.array([WORKSPACE_LENGTH, WALL_THICKNESS, WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_FRONT] = _get_box_info(WALL_FRONT, front_center, front_extent)
            
            left_center = np.array([-WORKSPACE_LENGTH / 2., (-WORKSPACE_WIDTH / 2. + y) / 2., WORKSPACE_HEIGHT / 2.])
            left_extent = np.array([WALL_THICKNESS, abs(-WORKSPACE_WIDTH / 2. - y), WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_LEFT] = _get_box_info(WALL_LEFT, left_center, left_extent)
            
            right_center = np.array([WORKSPACE_LENGTH / 2., (-WORKSPACE_WIDTH / 2. + y) / 2., WORKSPACE_HEIGHT / 2.])
            right_extent = np.array([WALL_THICKNESS, abs(-WORKSPACE_WIDTH / 2. - y), WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_RIGHT] = _get_box_info(WALL_RIGHT, right_center, right_extent)
            
            top_center = np.array([0., (-WORKSPACE_WIDTH / 2. + y) / 2., WORKSPACE_HEIGHT])
            top_extent = np.array([WORKSPACE_LENGTH, abs(-WORKSPACE_WIDTH / 2. - y), WALL_THICKNESS])
            grouping[-1][0][CEILING] = _get_box_info(CEILING, top_center, top_extent)
            
            grouping[-1][0][FLOOR] = _get_floor(grouping[-1][0])
            
            # this is the front, get the back boundary
            grouping.append({})
            grouping[-1][0][body_index] = bodies_info[body_index]
            for other_body in bodies_info.keys():
                body_pose = bodies_info[body_index][POSE]
                if other_body != body_index:
                    other_body_pose = bodies_info[other_body][POSE]
                    if other_body_pose[0][1] > body_pose[0][1]:
                        grouping[-1][0][other_body] = bodies_info[other_body]            
            
            y = min(body_aabb.get_min_bound([1]), body_aabb.get_max_bound([1]))
            
            back_center = np.array([0., WORKSPACE_WIDTH / 2., WORKSPACE_HEIGHT / 2.])
            back_extent = np.array([WORKSPACE_LENGTH, WALL_THICKNESS, WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_BACK] = _get_box_info(WALL_BACK, back_center, back_extent)            
            
            left_center = np.array([-WORKSPACE_LENGTH / 2., (WORKSPACE_WIDTH / 2. + y) / 2., WORKSPACE_HEIGHT / 2.])
            left_extent = np.array([WALL_THICKNESS, abs(WORKSPACE_WIDTH / 2. - y), WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_LEFT] = _get_box_info(WALL_LEFT, left_center, left_extent)
            
            right_center = np.array([WORKSPACE_LENGTH / 2., (WORKSPACE_WIDTH / 2. + y) / 2., WORKSPACE_HEIGHT / 2.])
            right_extent = np.array([WALL_THICKNESS, abs(WORKSPACE_WIDTH / 2. - y), WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_RIGHT] = _get_box_info(WALL_RIGHT, right_center, right_extent)
            
            top_center = np.array([0., (WORKSPACE_WIDTH / 2. + y) / 2., WORKSPACE_HEIGHT])
            top_extent = np.array([WORKSPACE_LENGTH, abs(WORKSPACE_WIDTH / 2. - y), WALL_THICKNESS])
            grouping[-1][0][CEILING] = _get_box_info(CEILING, top_center, top_extent)
            
            grouping[-1][0][FLOOR] = _get_floor(grouping[-1])
            
        if body_aabb_extent[1] >= WORKSPACE_WIDTH:
            # this is the left, get the right boundary
            goruping.append({})
            grouping[-1][0][body_index] = bodies_info[body_index]
            for other_body in bodies_info.keys():
                body_pose = bodies_info[body_index][POSE]
                if other_body != body_index:
                    other_body_pose = bodies_info[other_body][POSE]
                    if other_body_pose[0][0] > body_pose[0][0]:
                        grouping[-1][0][other_body] = bodies_info[other_body]            
            
            x = min(body_aabb.get_min_bound([0]), body_aabb.get_max_bound([0]))
            
            right_center = np.array([WORKSPACE_LENGTH / 2., 0., WORKSPACE_HEIGHT / 2.])
            right_extent = np.array([WALL_THICKNESS, WORKSPACE_WIDTH, WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_RIGHT] = _get_box_info(WALL_RIGHT, right_center, right_extent)
            
            front_center = np.array([(WORKSPACE_LENGTH / 2. + x) / 2., -WORKSPACE_WIDTH / 2., WORKSPACE_HEIGHT / 2.])
            front_extent = np.array([WORKSPACE_LENGTH / 2. - x, WALL_THICKNESS, WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_FRONT] = _get_box_info(WALL_FRONT, front_center, front_extent)
            
            back_center = np.array([(WORKSPACE_LENGTH / 2. + x) / 2., WORKSPACE_WIDTH / 2., WORKSPACE_HEIGHT / 2.])
            back_extent = np.array([WORKSPACE_LENGTH / 2. - x, WALL_THICKNESS, WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_BACK] = _get_box_info(WALL_BACK, back_center, back_extent)
            
            top_center = np.array([(WORKSPACE_LENGTH / 2. + x) / 2., 0., WORKSPACE_HEIGHT])
            top_extent = np.array([WORKSPACE_LENGTH / 2. - x, WORKSPACE_WIDTH, WALL_THICKNESS])
            grouping[-1][0][CEILING] = _get_box_info(CEILING, top_center, top_extent)
            
            grouping[-1][0][FLOOR] = _get_floor(grouping[-1][0])
            
            # this is the right, get the left boundary
            goruping.append({})
            grouping[-1][0][body_index] = bodies_info[body_index]
            for other_body in bodies_info.keys():
                body_pose = bodies_info[body_index][POSE]
                if other_body != body_index:
                    other_body_pose = bodies_info[other_body][POSE]
                    if other_body_pose[0][0] < body_pose[0][0]:
                        grouping[-1][0][other_body] = bodies_info[other_body]              
            
            x = max(body_aabb.get_min_bound([0]), body_aabb.get_max_bound([0]))
            
            left_center = np.array([-WORKSPACE_LENGTH / 2., 0., WORKSPACE_HEIGHT / 2.])
            left_extent = np.array([WALL_THICKNESS, WORKSPACE_WIDTH, WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_RIGHT] = _get_box_info(WALL_LEFT, left_center, left_extent)
            
            front_center = np.array([(-WORKSPACE_LENGTH / 2. + x) / 2., -WORKSPACE_WIDTH / 2., WORKSPACE_HEIGHT / 2.])
            front_extent = np.array([abs(-WORKSPACE_LENGTH / 2. - x), WALL_THICKNESS, WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_FRONT] = _get_box_info(WALL_FRONT, front_center, front_extent)
            
            back_center = np.array([(-WORKSPACE_LENGTH / 2. + x) / 2., WORKSPACE_WIDTH / 2., WORKSPACE_HEIGHT / 2.])
            back_extent = np.array([abs(-WORKSPACE_LENGTH / 2. - x), WALL_THICKNESS, WORKSPACE_HEIGHT])
            grouping[-1][0][WALL_BACK] = _get_box_info(WALL_BACK, back_center, back_extent)
            
            top_center = np.array([(-WORKSPACE_LENGTH / 2. + x) / 2., 0., WORKSPACE_HEIGHT])
            top_extent = np.array([abs(-WORKSPACE_LENGTH / 2. - x), WORKSPACE_WIDTH, WALL_THICKNESS])
            grouping[-1][0][CEILING] = _get_box_info(CEILING, top_center, top_extent)
            
            grouping[-1][0][FLOOR] = _get_floor(grouping[-1][0])            
    
    # reformat grouping
    grouping_to_return = []
    for group in grouping:
        grouping_to_return.append(group[0])
    
    return grouping_to_return

# the first one is always the workspace group
# can pass in existing groupings
def _group_bodies(initial_grouping, body_path, movables=[]):
    current_grouping = []
    bodies_to_be_sorted = []
    
    for group in initial_grouping:
        current_grouping.append([{}])
        for current_body in group.keys():
            if current_body in movables and not _pose_equal(group[current_body][POSE], _get_pose_from_id(current_body)):
                bodies_to_be_sorted.append(current_body)
            else:
                current_grouping[-1][0][current_body] = copy.deepcopy(group[current_body])
        if len(current_grouping[-1][0].keys()) == 1:
            current_grouping.pop()
    
    for body_index in bodies_to_be_sorted:
        is_sorted = False
        body_pcd = _get_pcd(body_index, body_path)
        body_pose = _get_pose_from_id(body_index)
        #body_bb = _get_bbs(body_pcd)
        body_bb = _get_body_bbs(body_index, body_path)
        body_info = _group_info(body_index, body_pcd, body_pose, body_bb)        
        for group in current_grouping:
            if is_sorted:
                break
            for existing_body in group[0].keys():
                if existing_body not in [FLOOR, CEILING, WALL_BACK, WALL_FRONT, WALL_LEFT, WALL_RIGHT]:
                    print "Meiying::_group_bodies: {} and {}".format(existing_body, body_index)
                    if _is_pcd_touch(body_pcd, group[0][existing_body][POINTCLOUD]):
                        is_sorted = True
                        group[0][body_index] = body_info
                        group[0][FLOOR] = _get_floor(group[0])
                        break
        if not is_sorted:          
            current_grouping.append([{}])
            current_grouping[-1][0][body_index] = body_info
            current_grouping[-1][0][FLOOR] = _get_floor(current_grouping[-1][0])
    
    grouping_to_return = []
    for group in current_grouping:
        grouping_to_return.append(group[0])
    
    return grouping_to_return

def _is_position_in_aabb(aabb, position):
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    
    for i in range(len(position)):
        if position[i] < min_bound[i]:
            return False
        if position[i] > max_bound[i]:
            return False
        
    return True

def _is_group_enclosed(pcds, flattened_obbs, target_grouping, body_path):
    # save the result to somewhere and read it here
    data = {}
    data_file_path = "data.json"
    if os.path.exists(data_file_path):
        with open(data_file_path) as f:
            data = json.load(f)
    
    body_indices = [_get_pcd_path(pybullet_id, body_path) for pybullet_id in target_grouping.keys()]
    found = False
    is_enclosed = False
    for key, value in data.items():
        if len(value[PATHS]) == len(target_grouping.keys()):
            found = True
            for target_grouping_key, target_grouping_value in target_grouping.items():
                body_path = _get_pcd_path(target_grouping_key, body_path)
                if body_path not in value[PATHS]:
                    found = False
                    break
                elif target_grouping_value[POSE] not in value[body_path]:
                    found = False
                    break
            if found:
                is_enclosed = value[ENCLOSED]
                break
    
    if found:
        return is_enclosed
    
    # do the calculation
    separate_voxel = 0
    combined_points = []
    for pcd in pcds:
        separate_voxel += _get_pcd_voxel_num(pcd)
        combined_points.extend(list(pcd.points))
    
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(np.array(combined_points))
    combined_voxel = _get_pcd_voxel_num(combined_pcd)
    
    is_enclosed = combined_voxel > separate_voxel * 1.5
    
    # save current result for future usages
    key = time.time()
    data[key] = {}
    data[key][PATHS] = [_get_pcd_path(pybullet_id, body_path) for pybullet_id in target_grouping.keys()]
    data[key][ENCLOSED] = is_enclosed
    for target_grouping_key, target_grouping_value in target_grouping:
        body_path = _get_pcd_path(target_grouping_key, body_path)
        if body_path not in data[key].keys():
            data[key][body_path] = []
        data[key][body_path].append(target_grouping_value[POSE])
    with open(data_file_path, 'w') as json_file:
        json.dump(data, json_file, indent = 4, sort_keys=True)
    
    return is_enclosed

def _get_pcd_voxel_num(pcd):
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 32486        # The port used by the server
    pc_path = "input.ply"
    
    # Create a socket object 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to the server on local computer 
    s.connect((HOST, PORT))
    
    if os.path.exists(pc_path):
        os.remove(pc_path)    
    o3d.io.write_point_cloud(pc_path, pcd)
    
    pc_absollute_path = os.path.abspath(pc_path)
    s.sendall(pc_absollute_path)
    
    data = int(s.recv(1024).decode())
    
    # close the connection 
    s.close() 
    
    return data

def _is_boundary_enclosed(group, body, body_path):
    pcd = None
    for key, value in group.items():
        if key not in [FLOOR, CEILING, WALL_BACK, WALL_FRONT, WALL_LEFT, WALL_RIGHT]:
            if key != body:
                pcd = value[POINTCLOUD]
                break
    
    if pcd is None:
        return False
    
    pcd_bb = _get_bb(pcd)
    pcd_bb_volume = pcd_bb.volume()
    pcd_bb_pcd = generate_cube(pcd_bb.center, pcd_bb.extent, orientation=pcd_bb.R)
    pcd_bb_pcd_voxel_num = _get_pcd_voxel_num(pcd_bb_pcd)
    pcd_voxel_num = _get_pcd_voxel_num(pcd_bb_pcd)
    
    pcd_volume = pcd_voxel_num / pcd_bb_pcd_voxel_num * pcd_bb_volume
    hole_volume = pcd_bb_volume - pcd_volume
    hole_surface_area = hole_volume / min(pcd_bb.extent)
    
    body_pcd = _get_pcd(body, body_path, transformation=np.identity(4))
    body_bb = _get_bb(body_pcd)
    body_bb_extent = body_bb.extent
    body_bb_surface_area = [body_bb.volume() / extent for extent in body_bb_extent]
    
    if hole_surface_area < max(body_bb_surface_area):
        return False
    
    return True

def _find_target_grouping_for_body(grouping, body, pose):
    target_grouping = None
    for group in grouping:
        pcds = []
        for key, value in group.items():
            pcds.append(value[POINTCLOUD])
        pcd_aabb = _get_pcds_aabb(pcds)
        if _is_position_in_aabb(pcd_aabb, pose[0]):
            target_grouping = group
            break    

def _is_body_enclosed_helper(target_grouping, body, body_path):
    if target_grouping is None:
        return False    
    
    obbs = []
    flattened_obbs = []
    pcds = []
    
    for key, value in target_grouping.items():
        obbs.append(value[BOUNDINGBOX])
        flattened_obbs.extend(value[BOUNDINGBOX])
        pcds.append(value[POINTCLOUD])
    
    if CEILING in target_grouping.keys():
        body_pcd = _get_pcd(body, body_path)
        if _is_boundary_enclosed(grouping, body, body_path):
            return True
    else:
        if len(flattened_obbs) <= 2: # one is floor, so there is only one other convex shape, so cannot enclose anything
            return False
        elif _is_group_enclosed(pcds, flattened_obbs, target_grouping, body_path):
            return True
    
    return False    

# grouping already excluded body
def _is_body_enclosed(grouping, body, pose, body_path):
    target_grouping = _find_target_grouping_for_body(grouping, body, pose)
    
    return _is_body_enclosed_helper(target_grouping, body, body_path)

def _remove_body_from_grouping(body, grouping):
    updated_grouping = []
    for group in grouping:
        if body in group.keys():
            if len(group.keys()) <= 2: # since the other one must be FLOOR
                continue
            else:
                updated_grouping.append({})
                for key, value in group.items():
                    if key != body:
                        updated_grouping[-1][key] = value
        else:
            updated_grouping.append({})
            for key, value in group.items():
                updated_grouping[-1][key] = value
    
    return updated_grouping

# start_pose, end_pose is a pose (position, quaternion)
def _is_3d_connected(start_pose, end_pose, body, grouping, body_path):
    # Method 1: Find path
    
    # Method 2: check potential enclosed space
    updated_grouping = _remove_body_from_grouping(body, grouping)
    
    if _is_body_enclosed(updated_grouping, body, start_pose, body_path):
        return False
    
    if _is_body_enclosed(updated_grouping, body, end_pose, body_path):
        return False
    
    return True

def _is_2d_connected():
    pass  

def get_pick_ik_fn(robot, fixed=[], teleport=False, num_attempts=10):
    #movable_joints = get_movable_joints(robot)
    #sample_fn = get_sample_fn(robot, movable_joints)
    def fn(body, pose, grasp, body_traj):
        print("[Meiying::get_pick_object_motion_gen] plan pick ik!!!")
        return None
    return fn

def _get_movable_obstacles_from_args(*args):
    movable_bodies = []
    movable_poses = []
    
    if len(args) > 0:
        if len(args) % 2 != 0:
            raise Exception("Must be body, pose pairs, cannot pass body or pose alone")
        movable_bodies = [args[i * 2] for i in range(len(args) / 2)]
        movable_bodies_poses = [args[i * 2 + 1] for i in range(len(args) / 2)]
        for movable_body_pose in movable_bodies_poses:
            movable_body_pose.assign()
            movable_poses.append(movable_body_pose.pose)
    
    return movable_bodies, movable_poses

def _get_movable_obstacles_from_fluents(fluents, excludes=[]):
    obstacles = []
    for fluent in fluents:
        name, args = fluent[0], fluent[1:]
        if name == 'atpose':
            o, p = args
            if not o in excludes:
                obstacles.append(o)
                p.assign()
        else:
            raise ValueError(name)
    return obstacles

def _get_place_area(body_pcd):
    body_pcd_bb = _get_bb_along_axis(obstacle_pcd, [2], [0, 1])
    body_pcd_bb_extent = body_pcd_bb.extent
    
    return body_pcd_bb_extent[0] * body_pcd_bb_extent[1]    

def _get_place_grouping_from_args(grouping, surface, body, body_path, *args):
    movable_bodies, _ = _get_movable_obstacles_from_args(*args)
    
    return _get_place_grouping_from_movables(grouping, surface, body, body_path, movable_bodies)

def _get_place_grouping_from_movables(grouping, surface, body, body_path, movable_bodies):
    relevant_grouping = _remove_body_from_grouping(body, grouping)
    relevant_grouping = _group_bodies(relevant_grouping, body_path, movables=movable_bodies)
    
    target_grouping = {}
    for group in relevant_grouping:
        if surface in group.keys():
            target_grouping = group
            break
    
    return target_grouping

def _get_place_boundary(body, body_path, surface, transformation=np.identity(4)):
    surface_aabb = _get_pcd(surface, body_path).get_axis_aligned_bounding_box()
    surface_top = surface_aabb.get_extent()[2]
    surface_min_boundary = surface_aabb.get_min_bound()
    surface_max_boundary = surface_aabb.get_max_bound()

    body_aabb = _get_pcd(body, body_path, transformation=transformation).get_axis_aligned_bounding_box()
    body_extent = body_aabb.get_half_extent()

    body_boundary_x = (surface_min_boundary[0] + body_extent[0], surface_max_boundary[0] - body_extent[0])
    body_boundary_y = (surface_min_boundary[1] + body_extent[1], surface_max_boundary[1] - body_extent[1])
    body_position_z = surface_max_boundary[2] + body_extent[2]
    
    return body_boundary_x, body_boundary_y, body_position_z

def get_tool_use_feasible_test(fixed=[], body_path={}, grouping=[]):
    def test(tool, body, pose, *args):
        obstacles = fixed
        movable_bodies, _ = _get_movable_obstacles_from_args(*args)
        obstacles.extend(movable_bodies)
        if _is_body_enclosed(grouping, body, pose.pose, body_path):
            return False
        
        return True

def _get_tool_use_contact_pose(start_pose, end_pose, contact_pose):
    x = np.array(end_pose[0] - start_pose[1])
    x[2] = 0.
    
    z = np.array([0., 0., 1.])
    y = np.cross(z, x)
    
    rotation = np.array([x, y, z]).T
    transformation = _get_transformation_matrix_with_rotation_matrix(rotation, np.array([0., 0., 0.]))
    
    pose_transformation = np.matmul(transformation, contact_pose)
    
    position, quaternion = _decompose_homogeneous_transformation_matrix(pose_transformation)
    
    return (tuple(position), tuple(quaternion))

def _get_tool_use_end_pose(tool_start_pose, start_pose, end_pose):
    tool_start_position = np.array(tool_start_pose[0])
    difference = np.array(end_pose - start_pose)
    difference[2] = 0.
    tool_end_position = tuple(tool_start_position + difference)
    tool_end_pose = (tool_end_position, tool_start_pose[1])
    
    return tool_end_pose

def _get_push_contact_pose(start_pose, end_pose):
    contact_pose = np.identity(4) # TODO: get the contact pose
    tool_start_pose = _get_tool_use_contact_pose(start_pose, end_pose, contact_pose)
    tool_end_pose = _get_tool_use_end_pose(tool_start_pose, start_pose, end_pose)
    
    return tool_start_pose, tool_end_pose
    
def _get_pull_contact_pose(start_pose, end_pose):
    contact_pose = np.identity(4) # TODO: get the contact pose
    tool_start_pose = _get_tool_use_contact_pose(start_pose, end_pose, contact_pose)
    tool_end_pose = _get_tool_use_end_pose(tool_start_pose, start_pose, end_pose)
    
    return tool_start_pose, tool_end_pose

def _get_directions():
    angles = [i for i in range(360)]
    directions = []
    
    for angle in angles:
        direction = np.array([np.cos(np.deg2rad(angle), np.sin(np.deg2rad(angle)), 0.)])
        directions.append(direction)
    
    return directions

def _merge_pcds(pcds):
    points = []
    
    for pcd in pcds:
        points.extend(list(np.asarray(pcd.points)))
    
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.array(points))
    
    return merged_pcd

def _get_transformation_around_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0., 0.],
                     [np.sin(theta),  np.cos(theta), 0., 0.],
                     [           0.,             0., 0., 0.],
                     [           0.,             0., 0., 1.]])

# pose = (position, quaternion)
def _rotate_theta(pose, theta):
    transformation = _get_homogeneous_transformation_matrix_from_quaternion(np.array(pose[1]), np.array(pose[0]))
    
    translation_inverse = _get_homogeneous_transformation_matrix_from_quaternion([0., 0., 0., 1.], -np.array(pose[0]))
    rotation = _get_transformation_around_z(theta)
    translation = _get_homogeneous_transformation_matrix_from_quaternion([0., 0., 0., 1.], np.array(pose[0]))
    
    transformation = np.matmul(translation_inverse, transformation)
    transformation = np.matmul(rotation, transformation)
    transformation = np.matmul(translation, transformation)
    
    position, quaternion = _decompose_homogeneous_transformation_matrix_to_rotation_matrix(transformation)
    
    return (tuple(position), tuple(quaternion))

def _get_random_pose(body, body_path, surface, is_rotate=False): 
    pose = _get_pose_from_id(body)
    quaternion = pose[1]
    if is_rotate:
        pose = _get_pose_from_id(body)
        theta = random.uniform(0., np.pi * 2)
        _, quaternion = _rotate_theta(pose, theta)
    transformation = _get_homogeneous_transformation_matrix_from_quaternion(quaternion, pose[0])
    
    body_boundary_x, body_boundary_y, body_position_z = _get_place_boundary(body, body_path, surface, transformation=transformation)
    
    body_position_x = random.uniform(body_boundary_x[0], body_boundary_x[1])
    body_position_y = random.uniform(body_boundary_y[0], body_boundary_y[1])
    
    return ((body_position_x, body_position_y, body_position_z), quaternion)

def _is_bodies_same(bodies_1, bodies_2):
    if len(bodies_1) != len(bodies_2):
        return False
    
    for body in bodies_1:
        if body not in bodies_2:
            return False
    
    return True

def _is_poses_same(place_value_dict, target_grouping):
    obstacles = target_grouping.keys()
    
    for obstacle in obstacles:
        if not _pose_equal(place_value_dict[str(obstacle)], target_grouping[obstacle][POSE]):
            return False
    
    return True

def _save_placeable(body, surface, target_grouping, place_data, placeable, all_area_feasible, feasible_positions, place_file_path):
    key = time.time()
    place_data[key] = {}
    
    place_data[key][BODY] = body
    place_data[key][SURFACE] = surface
    place_data[key][MOVABLES] = target_grouping.keys()
    
    for obstacle_id, obstacle_info in target_grouping.items():
        place_data[key][obstacle_id] = obstacle_info[POSE]
        
    place_data[key][PLACEABLE] = placeable
    place_data[key][ALL_AREA_FEASIBLE] = all_area_feasible
    place_data[key][FEASIBLE_POSITIONS] = feasible_positions
    
    with open(place_file_path, 'w') as json_file:
        json.dump(place_data, json_file, indent = 4, sort_keys=True)    

def _save_pullable(body, pose, tool, target_grouping, pull_data, is_pullable, all_direction_feasible, feasible_positions, pull_file_path):
    key = time.time()
    pull_data[key] = {}
    
    pull_data[key][BODY] = body
    pull_data[key][BODY_POSE] = pose
    pull_data[key][TOOL] = tool
    pull_data[key][OBSTACLES] = target_grouping.keys()
    
    for obstacle_id, obstacle_info in target_grouping.items():
        pull_data[key][obstacle_id] = obstacle_info[POSE]
        
    pull_data[key][PULLABLE] = is_pullable
    pull_data[key][ALL_DIRECTION_FEASIBLE] = all_direction_feasible
    pull_data[key][FEASIBLE_POSITIONS] = feasible_positions
    
    with open(pull_file_path, 'w') as json_file:
        json.dump(pull_data, json_file, indent = 4, sort_keys=True)       

def _find_place_area(body, surface, obstacles, body_path):
    is_placeable = False
    all_area_feasible = False
    feasible_positions = []
    get_results = False
    
    obstacles_bbs = _get_obstacles_bbs(obstacles, body_path)
    body_bbs = _get_body_bbs(body, body_path, transformation=np.identity(4))
    #is_free_surface = True
    num_samples = 10
    num_not_colliding = 0
    while num_samples > 0:
        #body_pose = _get_random_pose(body, body_path, surface, is_rotate=True)
        #print "[Meiying::_find_place_area] body:", body
        #print "[Meiying::_find_place_area] surface:", surface
        #print "[Meiying::_find_place_area] obstacles:", obstacles
        body_pose = sample_placement(body, surface)
        #print "[Meiying::_find_place_area] body_pose:", body_pose
        #updated_body_bbs = [_transform_bb(bb, _get_homogeneous_transformation_matrix_from_quaternion(body_pose[1], body_pose[0])) for bb in body_bbs]
        current_current_pose = BodyPose(body, body_pose)
        current_current_pose.assign()        
        
        if not _is_obj_env_collide(body, obstacles, body_bbs, body_pose, obstacles_bbs, body_path):
            num_not_colliding += 1
            #print "[Meiying::_find_place_area] not collide with anything #{} !!!!".format(num_not_colliding)
            if num_not_colliding > 5:
                get_results = True
                is_placeable = True
                all_area_feasible = True
            break
        num_samples -= 1

    if not get_results:
        boundary_x, boundary_y, position_z = _get_place_boundary(body, body_path, surface, transformation=np.identity(4))
        position_z += 0.004
        step = 0.01
        x_choices = [boundary_x[0] + i * step for i in range(int((boundary_x[1] - boundary_x[0]) / step))]
        if x_choices[-1] < boundary_x[1]:
            x_choices.append(boundary_x[1])
        y_choices = [boundary_y[0] + i * step for i in range(int((boundary_y[1] - boundary_y[0]) / step))]
        if y_choices[-1] < boundary_y[1]:
            y_choices.append(boundary_y[1])
        for x in x_choices:
            for y in y_choices:
                body_pose = ((x, y, position_z), (0., 0., 0., 1.))
                #r = np.linalg.norm(np.array([x, y]))
                #if r < ROBOT_RANGE[0] or r > ROBOT_RANGE[1]:
                    #continue
                current_current_pose = BodyPose(body, body_pose)
                current_current_pose.assign()
                if not _is_obj_env_collide(body, obstacles, body_bbs, body_pose, obstacles_bbs, body_path):
                    feasible_positions.append(body_pose[0])
        
        if len(feasible_positions) != 0:
            is_placeable = True
            all_area_feasible = False
        else:
            is_placeable = False
            all_area_feasible = False            
    
    return is_placeable, all_area_feasible, feasible_positions

def _get_tool_use_poses(body, tool, body_pose1, body_pose2, usage):
    body_start = transformation_util.get_homogeneous_transformation_matrix_from_quaternion(np.array(body_pose1[1]), np.array(body_pose1[0]))
    body_end   = transformation_util.get_homogeneous_transformation_matrix_from_quaternion(np.array(body_pose2[1]), np.array(body_pose2[0]))
    
    tool_trajectory = usage.get_usage(body_start, body_end)[0]
    #tool_start      = tool_trajectory[2]
    #tool_end        = tool_trajectory[-2]
    tool_start      = tool_trajectory[0]
    tool_end        = tool_trajectory[-1]    
    
    tool_pose1_position, tool_pose1_quaternion = transformation_util.decompose_homogeneous_transformation_matrix(tool_start)
    tool_pose2_position, tool_pose2_quaternion = transformation_util.decompose_homogeneous_transformation_matrix(tool_end)
    
    tool_pose1 = (tuple(tool_pose1_position), tuple(tool_pose1_quaternion))
    tool_pose2 = (tuple(tool_pose2_position), tuple(tool_pose2_quaternion))
    
    return tool_pose1, tool_pose2

def _find_pull_positions(body, pose, tool, obstacles, body_path, usage):
    check_tool_body_collision = False
    #if tool in manipulanda_tool_pose.keys():
        #if body in manipulanda_tool_pose[tool].keys():
            #check_tool_body_collision = True    
    
    is_pullable = False
    all_direction_feasible = False
    feasible_positions = {}
    get_results = False 
    
    obstacles_bbs = _get_obstacles_bbs(obstacles, body_path)
    body_pose_transformation = _get_homogeneous_transformation_matrix_from_quaternion(np.array(pose.pose[1]), np.array(pose.pose[0]))
    #body_bbs = _get_body_bbs(body, body_path, transformation=body_pose_transformation)
    tool_bbs = _get_body_bbs(tool, body_path, transformation=np.identity(4))
    
    if not check_tool_body_collision:
        num_samples = 10
        num_non_colliding_examples = 0
        is_pullable = True
        all_direction_feasible = True
        feasible_positions = {}
        get_results = False    
        while num_samples > 0:
            #print "current num_samples:", num_samples
            theta = random.uniform(0., np.pi * 2.)
            direction = np.array([np.cos(theta), np.sin(theta), 0.])
            distance = 1.
            body_end_position = np.array(pose.pose[0]) + direction * distance
            body_end_pose = (tuple(body_end_position), pose.pose[1])
            tool_start_pose, tool_end_pose = _get_tool_use_poses(body, tool, pose.pose, body_end_pose, usage)
            tool_start_pose_transformation = _get_homogeneous_transformation_matrix_from_quaternion(np.array(tool_start_pose[1]), np.array(tool_start_pose[0]))
            
            set_pose(tool, tool_start_pose)
            
            if _is_obj_env_collide(tool, obstacles, tool_bbs, tool_start_pose, obstacles_bbs, body_path, visualize=False):
                all_direction_feasible = False
                break
    
            num_samples -= 1
    
    #print "finish random sampling!!"
    if all_direction_feasible:
        #return is_pullable, all_direction_feasible, feasible_positions
        #print "all_direction_feasible!!!"
        return True, True, {}

    #print "NOT all_direction_feasible!!!"
    if not all_direction_feasible:
        #print "NOT all_direction_feasible!!!"
        is_pullable = False
        all_direction_feasible = False
        feasible_positions = {}
        get_results = False        
        #raise Exception("did not find results from random sampling. now try to get results systematically!!")
        
        #thetas = [i * 2.0 for i in range(180)]
        #thetas = [0., 90., 180., 270.]
        #thetas = [0., 90., 180., 270.]
        #thetas = [0., 180.]
        thetas = [180.]
        thetas = np.deg2rad(thetas)
        tool_bbs = _get_body_bbs(tool, body_path, transformation=np.identity(4))
        body_bbs = _get_body_bbs(body, body_path, transformation=np.identity(4))
        
        obstacles_pcds = [_get_pcd(i, body_path) for i in obstacles]
        obstacles_pcd = _merge_pcds(obstacles_pcds)
        obstacles_bb = _get_bb_along_axis(obstacles_pcd, [2], [0, 1])
        
        for theta in thetas:
            #print "theta: ", theta
            direction = np.array([np.cos(theta), np.sin(theta), 0.])
            distance = 1.
            body_start_pose = pose.pose
            body_end_position = np.array(pose.pose[0]) + direction * distance
            body_end_pose = (tuple(body_end_position), pose.pose[1])
            #print "\tbody_start_pose:", body_start_pose
            #print "\tbody_end_pose:", body_end_pose
            tool_start_pose, tool_end_pose = _get_tool_use_poses(body, tool, pose.pose, body_end_pose, usage)
            #print "\ttool_start_pose:", tool_start_pose
            #print "\ttool_end_pose:", tool_end_pose
            tool_start_pose_transformation = _get_homogeneous_transformation_matrix_from_quaternion(np.array(tool_start_pose[1]), np.array(tool_start_pose[0]))
            #print "\ttool_start_pose_transformation:", tool_start_pose_transformation
            
            if check_tool_body_collision:
                check_collision_body_pcd = _get_pcd(body, body_path, _get_homogeneous_transformation_matrix_from_quaternion(np.array(body_start_pose[1]), np.array(body_start_pose[0])))
                check_collision_tool_pcd = _get_pcd(tool, body_path, tool_start_pose_transformation)
                
                o3d.visualization.draw_geometries([check_collision_body_pcd, check_collision_tool_pcd], "collide? {}".format(is_pcd_collide(check_collision_body_pcd, check_collision_tool_pcd)))
                
                if is_pcd_collide(check_collision_body_pcd, check_collision_tool_pcd):
                    #print "collide!"
                    continue
            
            set_pose(tool, tool_start_pose)
            set_pose(body, body_start_pose)
            
            #current_tool_pose = BodyPose(tool, tool_start_pose)
            #current_tool_pose.assign()
            
            #print "checking theta: ", theta
            #if _is_obj_env_collide(tool, obstacles, tool_bbs, tool_start_pose, obstacles_bbs, body_path, visualize=True):
                #print "theta collide: ", theta
            
            if not _is_obj_env_collide(tool, obstacles, tool_bbs, tool_start_pose, obstacles_bbs, body_path):
                #print "feasible start theta: ", theta
                min_distance = None
                #for i in [5, 10, 15, 17, 20, 25, 30, 35, 40, 45, 50]:
                for i in range(5, 25):
                    #print "theta: ", theta, "i:", i
                    distance = i * 0.01
                    #print "distance:", distance
                    body_end_position = direction * distance + np.array(body_start_pose[0])
                    #print "\tbody_end_position: ", body_end_position
                    body_end_pose = (tuple(body_end_position), body_start_pose[1])
                    tool_start_pose, tool_end_pose = _get_tool_use_poses(body, tool, pose.pose, body_end_pose, usage)
                    current_body_bbs = [_transform_bb(body_bb, _get_homogeneous_transformation_matrix_from_quaternion(np.array(body_end_pose[1]), np.array(body_end_pose[0]))) for body_bb in body_bbs]
                    current_tool_bbs = [_transform_bb(tool_bb, _get_homogeneous_transformation_matrix_from_quaternion(np.array(tool_end_pose[1]), np.array(tool_end_pose[0]))) for tool_bb in tool_bbs]
                    if not _is_bb_separate(current_body_bbs, obstacles_bb):
                        #print "\tcontained in obstacle"
                        continue
                    found_collision = False
                    
                    set_pose(tool, tool_end_pose)
                    set_pose(body, body_end_pose) 
                    
                    body_pcd = _get_pcd(body, body_path)
                    tool_pcd = _get_pcd(tool, body_path)
                    
                    #temp_is_collide = is_pcd_collide(obstacles_pcd, _merge_pcds([body_pcd, tool_pcd]))
                    #o3d.visualization.draw_geometries([obstacles_pcd, _merge_pcds([body_pcd, tool_pcd])], "collide? {}".format(temp_is_collide))
                    
                    #print "distance: ", min(np.asarray(obstacles_pcd.compute_point_cloud_distance(_merge_pcds([body_pcd, tool_pcd]))))
                    if is_pcd_collide(obstacles_pcd, _merge_pcds([body_pcd, tool_pcd])):
                        #print "\tcollision detected! Find next theta"
                        #o3d.visualization.draw_geometries([obstacles_pcd, _merge_pcds([body_pcd, tool_pcd])])
                        break
            
                    #if not _is_bb_contain(current_body_bbs, obstacles_bb) and not _is_bb_collide(current_body_bb, obstacles_bb):
                    if _is_bb_separate(current_body_bbs, obstacles_bb):    
                        #print "\tfinally not contained! min_distance of this theta is:", distance
                        min_distance = distance
                        feasible_positions[str(tuple(direction))] = min_distance
                        break
        
        if len(feasible_positions.keys()) != 0:
            is_pullable = True
            all_direction_feasible = False
        else:
            is_pullable = False
            all_direction_feasible = False
            #print "no feasible positions"
    
    #print "feasible_positions:"
    #print feasible_positions
    
    #raw_input("continue?")
    
    return is_pullable, all_direction_feasible, feasible_positions

def _format_feasible_positions(feasible_positions):
    formatted_feasible_positions = {}
    
    for key, value in feasible_positions.items():
        positions = tuple([float(i) for i in key[1:-1].split(",")])
        formatted_feasible_positions[positions] = value
    
    return formatted_feasible_positions

# format for place:
# {time_id: {"body":int, "surface":surface, "movables":[], id:pose, id:pose, ..., all_area_reasible: bool, feasible_positions=[]}
def _place_area_feasible(grouping, fixed, body_path, body, surface, place_file_path, *args):
    print "meiying::_place_area_feasible {} on surface {}".format(body, surface)
    
    if surface == fixed[0]: # can always put things on the table
        print "meiying::_place_area_feasible {} on surface {}: True. can always put things on the floor".format(body, surface)
        return True
    
    target_grouping = _get_place_grouping_from_args(grouping, surface, body, body_path, *args)
    
    num_obstacles_on_surface = 0
    obstacle_pcds = []
    for obstacle in target_grouping.keys():
        if obstacle not in [FLOOR, CEILING, WALL_RIGHT, WALL_LEFT, WALL_FRONT, WALL_BACK, surface, body]:
            num_obstacles_on_surface += 1
            obstacle_pcds.append(target_grouping[obstacle][POINTCLOUD])
    
    if num_obstacles_on_surface == 0: # no other object on the surface
        print "meiying::_place_area_feasible {} on surface {}: True. No other objects on the surface".format(body, surface)
        return True
    
    place_data = {}
    if not place_file_path is None:
        try:
            with open(place_file_path) as f:
                place_data = json.load(f)
        except:
            pass
    
    obstacles = target_grouping.keys()
    
    # find whether this has been checked before
    is_found = False
    is_placeable = True
    #i = 0
    #print "# of items:", len(place_data.items())
    for key, value in place_data.items():
        #print "i =", i
        #i += 1
        #print "value"
        #print value
        if value[BODY] == body and value[SURFACE] == surface:
            if _is_bodies_same(obstacles, value[MOVABLES]):
                if _is_poses_same(value, target_grouping):
                    is_found = True
                    is_placeable = value[PLACEABLE]
                    break
                
    if is_found:
        return is_placeable

    is_placeable, all_area_feasible, feasible_positions = _find_place_area(body, surface, obstacles, body_path)
    
    print "meiying::_place_area_feasible {} on surface {}: {}".format(body, surface, is_placeable)
    
    _save_placeable(body, surface, target_grouping, place_data, is_placeable, all_area_feasible, feasible_positions, place_file_path)

    #print "!!!!!!!!!SAVE PLACEABLE to {}!!!!!!!!!!!!!!!!".format(place_file_path)

    return is_placeable

def _find_place_grouping(body, grouping):
    for group in grouping:
        if body in group.keys():
            return group
    
    return None      

# the first one in fixed is always floor
def _place_connection_feasible(body, start_surface, end_surface, body_path, fixed, grouping):
    if start_surface == fixed[0] and end_surface == fixed[0]:
        return True
    
    updated_grouping = _remove_body_from_grouping(body, grouping)
    
    if start_surface == fixed[0]:
        end_grouping = _find_place_grouping(end_surface, updated_grouping)
        if _is_body_enclosed_helper(end_grouping, body, body_path):
            return False
        else:
            return True
        
    if end_surface == fixed[0]:
        start_grouping = _find_place_grouping(start_surface, updated_grouping)
        if _is_body_enclosed_helper(start_grouping, body, body_path):
            return False
        else:
            return True
    
    start_grouping = _find_place_grouping(start_surface, updated_grouping)
    end_grouping = _find_place_grouping(end_surface, updated_grouping)
    if _is_body_enclosed_helper(start_grouping, body, body_path):
        return False
    elif _is_body_enclosed_helper(end_grouping, body, body_path):
        return False
    else:
        return True
    
    return True

def _pose_contain(pose, pose_list):
    for each_pose in pose_list:
        if _pose_equal(pose, each_pose):
            return True
        
    return False

def _pose_equal(pose1, pose2):
    position1, quaternion1 = pose1[0], pose1[1]
    position2, quaternion2 = pose2[0], pose2[1]
    
    for i in range(3):
        if round(position1[i], 4) != round(position2[i], 4):
            return False
    
    for i in range(4):
        if round(quaternion1[i], 4) != round(quaternion2[i], 4):
            return False
    
    return True

# fluents: [[fluent, fluent, ...], [fluent, ...], ...]
# results: {body: [pose, pose, ...], ...}
def _grouping_atpose_fluents(fluents):
    group = {}
    
    for each_group in fluents:
        for fluent in each_group:
            name, args = fluent[0], fluent[1:]
            if name == 'atpose':
                body, body_pose = args[0], args[1]
                if isinstance(body_pose, BodyPose):
                    pose = body_pose.pose
                    if body not in group.keys():
                        group[body] = []
                    if not _pose_contain(pose, group[body]):
                        group[body].append(pose)

    return group

# surface is always fixed
def _get_movable_obstacles_from_fluent_group(body, surface, grouping):
    movables = []
    movable_poses = []
    #movable_and_pose = []
    surface_pose = _get_pose_from_id(surface)
    
    for movable in grouping.keys():
        if movable != body:
            max_distance = 0.
            max_pose = None
            for pose in grouping[movable]:
                distance = np.linalg.norm(np.array(surface_pose[0]) - np.array(pose[0]))
                if distance > max_distance:
                    max_pose = pose
                    max_distance = distance
            if not max_pose is None:
                movables.append(movable)
                movable_poses.append(max_pose)
                #movable_and_pose.append(movable)
                #movable_and_pose.append(max_pose)
                BodyPose(movable, max_pose).assign()
    
    return movables, movable_poses

def _generate_place_pose(body, surface, fixed=[], body_path={}):
    while True:
        pose = sample_placement(body, surface)
        if (pose is None) or any(pairwise_collision(body, b, body_path=body_path) for b in fixed):
            continue
        x, y = pose[0][0], pose[0][1]
        r = np.linalg.norm(np.array([x, y]))
        direction = np.array([x, y]) / r
        if r < ROBOT_RANGE[0]:
            position_2d = direction * ROBOT_RANGE[0]
            x, y = position_2d[0], position_2d[1]
            pose = ((x, y, pose[0][2]), pose[1])
        elif r > ROBOT_RANGE[1]:
            position_2d = direction * ROBOT_RANGE[1]
            x, y = position_2d[0], position_2d[1]
            pose = ((x, y, pose[0][2]), pose[1])
        body_pose = BodyPose(body, pose)
        return (body_pose,)
    
def _get_pull_grouping(body, tool, pose, grouping, fixed, body_path):
    pose.assign()
    relevant_grouping = _remove_body_from_grouping(tool, grouping)
    relevant_grouping = _group_bodies(relevant_grouping, body_path, movables=[body])
    
    target_grouping = {}
    for group in relevant_grouping:
        if body in group.keys():
            target_grouping = group
            break
    
    target_grouping_without_body = {}
    for key, value in target_grouping.items():
        if not key in [body, FLOOR]:
            target_grouping_without_body[key] = value
    
    return target_grouping_without_body

def _get_pick_grouping(body, pose, surface, grouping, body_path):
    #print "body:", body
    #print "pose:", pose.pose
    
    pose.assign()
    
    #print "grouping"
    #for group in grouping:
        #print group
        #print
    
    relevant_grouping = _group_bodies(grouping, body_path, movables=[body])
    
    #print "relevant_grouping"
    #for group in relevant_grouping:
        #print
        #print group
    
    target_grouping = {}
    for group in relevant_grouping:
        if body in group.keys():
            target_grouping = group
            break
    
    #print "target_grouping"
    #print target_grouping    
    
    target_grouping_without_body = {}
    for key, value in target_grouping.items():
        if not key in surface:
            if not key in [body, FLOOR]:
                target_grouping_without_body[key] = value
    
    return target_grouping_without_body    

#######################################################################################################
#############################    Public Functions/Tests/Generators ####################################
#######################################################################################################

"""
Tests
"""
def _get_base_data_dir(task):
    return os.path.join(constants.get_learned_data_dir(), task)

def _get_affordance(task, goal): # non-tool afforfances
    # check this has been trained, or whether it is source tool or sub tool
    tool = "meiying_block_for_pick_and_place_mid_size"
    tool_path = os.path.join(_get_base_data_dir(task), tool)
    if os.path.exists(tool_path): # it must be the gripper
        pass
    else:
        return None

SOURCE_TOOL = "SOURCE_TOOL"
SOURCE_GOAL = "SOURCE_GOAL"
"meiying_block_for_gripper"

TOOL_LEARNED = {"push": {SOURCE_TOOL: ["meiying_T_push_tool"],
                         SOURCE_GOAL: ["meiying_tool_target"]},
                "pull": {SOURCE_TOOL: ["meiying_L_pull_tool", "meiying_L_chain_pull_tool"],
                         SOURCE_GOAL: ["meiying_tool_target"]},
                "pick": {SOURCE_TOOL: ["meiying_block_for_gripper"],
                         SOURCE_GOAL: ["meiying_block_for_pick_and_place_mid_size"]},
                "place": {SOURCE_TOOL: ["meiying_block_for_gripper"],
                          SOURCE_GOAL: ["meiying_block_for_pick_and_place_mid_size"]},              
                }

# provide: short name for tools.
def _get_tool_usage(task, goal, tool=None):
    usage = None
    
    if tool is None:
        tool = "meiying_block_for_gripper"
    
    tool_path = os.path.join(_get_base_data_dir(task), tool)
    goal_path = os.path.join(tool_path, goal)
    base_data_dir = _get_base_data_dir(task)
    #print "tool_path: ", tool_path
    #print "goal_path: ", goal_path
    if os.path.exists(tool_path) and os.path.exists(goal_path): # source tool and goal
        source_usage = tool_usage.SourceToolUsage(task, tool, goal, base_data_dir)
        source_usage.read()
        usage = source_usage
    else: # it is a sub usage
        source_tool = TOOL_LEARNED[task][SOURCE_TOOL][0]
        source_goal = TOOL_LEARNED[task][SOURCE_GOAL][0]
        source_usage = tool_usage.SourceToolUsage(task, source_tool, source_goal, base_data_dir)
        source_usage.read()
        sub_usage = tool_usage.SubstituteToolUsage(task, tool, goal, base_data_dir, source_usage)
        usage = sub_usage
    
    return usage

def _get_usage(task, body, body_path, tool=None):
    goal_name = _get_pcd_name(body, body_path)
    tool_name = None
    if not tool is None:
        tool_name = _get_pcd_name(tool, body_path)
    
    return _get_tool_usage(task, goal_name, tool=tool_name)    

def _check_feasible_from_start(grouping, task, surface, fixed, body_path, file_path, body, pose, start_surface, end_surface, args, tool=None):
    usage = _get_usage(task, body, body_path, tool)
    
    if usage.task_type == constants.TASK_TYPE_OTHER: # nothing will be relocated
        return True
    
    if usage.subsubtask_type == constants.TASK_SUBSUBTYPE_POSE_CHANGE_WORLD_FRAME_SPECIFIC: # pose changes in the world frame
        r = np.linalg.norm(np.array([pose.pose[0][0], pose.pose[0][1]]))
        if r > np.sqrt(2.) * 0.65: # out of robot range
            return False        
        
        #print "body:", body
        #print "pose:", pose.pose
        #print "surface:", surface
        #print "fixed:", fixed
        
        target_grouping = _get_pick_grouping(body, pose, surface, grouping, body_path)
        
        #print "target_grouping"
        #print target_grouping
        
        if len(target_grouping.keys()) == 0: # nothing else in collision other than the platform
            #print "nothing else, return True"
            return True
        
        obstacles = target_grouping.keys()
        movable_bodies, _ = _get_movable_obstacles_from_args(*args)
        obstacles.extend(movable_bodies)
        S = usage.get_goal_axis()
        theta = usage.get_goal_theta()
        result = _is_obj_traj_feasible(body, pose, S, theta, obstacles, body_path, frame=WORLD_FRAME)
        
        #if body == 6:
            #print "pick/place feasible: ", result
            #raw_input("continue?")
        
        return result
    
    #print "task:", task
    #print "subsubtype:", usage.subsubtask_type
    #print "source tool:", usage.source_tool
    #print "source goal:", usage.source_goal
    #print "source tool subsub type:", usage.source_usage.subsubtask_type
    if usage.subtask_type == constants.TASK_SUBTYPE_POSE_CHANGE_SPECIFIC: # pose changes in the goal frame
        r = np.linalg.norm(np.array([pose.pose[0][0], pose.pose[0][1]]))
        if r > np.sqrt(2.) * 0.65: # out of robot range
            return False        
        
        #print "body:", body
        #print "pose:", pose.pose
        #print "surface:", surface
        #print "fixed:", fixed
        
        target_grouping = _get_pick_grouping(body, pose, surface, grouping, body_path)
        
        #print "target_grouping"
        #print target_grouping
        
        if len(target_grouping.keys()) == 0: # nothing else in collision other than the platform
            #print "nothing else, return True"
            return True
        
        obstacles = target_grouping.keys()
        movable_bodies, _ = _get_movable_obstacles_from_args(*args)
        obstacles.extend(movable_bodies)
        S = usage.get_goal_axis()
        theta = usage.get_goal_theta()        
        result = _is_obj_traj_feasible(body, pose, S, theta, obstacles, body_path, frame=BODY_FRAME)
        
        return result
    
    target_grouping = _get_pull_grouping(body, tool, pose, grouping, fixed, body_path)
    
    print "body:", body
    print "pose:", pose
    print "tool:", tool
    print "target_grouping:", target_grouping
    
    num_obstacles_on_surface = 0
    obstacle_pcds = []
    for obstacle in target_grouping.keys():
        if obstacle not in [FLOOR, CEILING, WALL_RIGHT, WALL_LEFT, WALL_FRONT, WALL_BACK, tool, body]:
            num_obstacles_on_surface += 1
            obstacle_pcds.append(target_grouping[obstacle][POINTCLOUD])
        
    if num_obstacles_on_surface == 0: # no other object in this group
        print "no other obstacles on the surface, can pull!!!"
        return True
    
    pull_data = {}
    if not file_path is None:
        try:
            with open(file_path) as f:
                pull_data = json.load(f)
        except:
            pass
    
    obstacles = target_grouping.keys()
    
    print "obstacles:", obstacles
    
    # find whether this has been checked before
    is_found = False
    is_pullable = True
    #i = 0
    #print "# of items:", len(place_data.items())
    for key, value in pull_data.items():
        #print "i =", i
        #i += 1
        #print "value"
        #print value
        if value[BODY] == body and value[TOOL] == tool and _pose_equal(value[BODY_POSE, pose.pose]):
            if _is_bodies_same(obstacles, value[OBSTACLES]):
                if _is_poses_same(value, target_grouping):
                    is_found = True
                    is_pullable = value[PULLABLE]
                    break
                
    if is_found:
        return is_pullable

    is_pullable, all_direction_feasible, feasible_positions = _find_pull_positions(body, pose, tool, obstacles, body_path, usage)
    
    print "meiying::get_pull_feasible_test {} with tool {}: {}".format(body, tool, is_pullable)
    
    _save_pullable(body, pose.pose, tool, target_grouping, pull_data, is_pullable, all_direction_feasible, feasible_positions, file_path)

    #print "!!!!!!!!!SAVE PULLABLE to {}!!!!!!!!!!!!!!!!".format(pull_file_path)

    return is_pullable    
    
def _check_feasible_from_end(grouping, task, fixed, body_path, file_path, body, pose, start_surface, end_surface, args, tool=None):
    print "Meiying::_check_feasible_from_end"
    
    area_feasible = _place_area_feasible(grouping, fixed, body_path, body, end_surface, file_path, *args)
    
    print "[Meiying::area_feasible]:", area_feasible
    if not area_feasible:
        return False
    
    return True

def get_generic_feasible_test(grouping, task="", surface=[], fixed=[], body_path={}, file_path=None):
    def test(body, start_surface, end_surface, *args):
        if task == "move": # do not make this check yet.
            return True
        pose = None
        #print "task:", task
        if len(args) > 0 and isinstance(args[0], BodyPose):
            pose = args[0]
            args = args[1:]
        
        #if task == "place":
            #if not pose is None:
                #raise Exception("place should have no pose!!!!!")
        #if task == "pick":
            #if pose is None:
                #raise Exception("pick should have pose!!!!!")
        
        if not pose is None: # check from start
            return _check_feasible_from_start(grouping, task, surface, fixed, body_path, file_path, body, pose, start_surface, end_surface, args, tool=None)

        return _check_feasible_from_end(grouping, task, fixed, body_path, file_path, body, pose, start_surface, end_surface, args, tool=None)
        
    return test

# this should be theoretically the same as the generic feasible test
# however, due to the code structure from pddlstream, and the extra parameter of the tool
# they are separated as 2 different functions
def get_tool_use_generic_feasible_test(grouping, surface, task="", fixed=[], body_path={}, file_path=None):
    def test(body, tool, pose, *args):
        start_surface = None
        end_surface   = None
        return _check_feasible_from_start(grouping, task, surface, fixed, body_path, file_path, body, pose, start_surface, end_surface, args, tool=tool)
    return test

def get_pick_feasible_test(grouping, surface=[], fixed=[], body_path={}):
    def test(body, start_surface, end_surface, pose, *args):
        if body == 6:
            print("\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pick feasible test for object 6!!")
        
        r = np.linalg.norm(np.array([pose.pose[0][0], pose.pose[0][1]]))
        if r > np.sqrt(2.) * 0.65: # out of robot range
            return False        
        
        #print "body:", body
        #print "pose:", pose.pose
        #print "surface:", surface
        #print "fixed:", fixed
        
        target_grouping = _get_pick_grouping(body, pose, surface, grouping, body_path)
        
        #print "target_grouping"
        #print target_grouping
        
        if len(target_grouping.keys()) == 0: # nothing else in collision other than the platform
            #print "nothing else, return True"
            return True
        
        obstacles = target_grouping.keys()
        movable_bodies, _ = _get_movable_obstacles_from_args(*args)
        obstacles.extend(movable_bodies)
        end_position = (pose.pose[0][0], pose.pose[0][1], pose.pose[0][2] + 0.1)
        result = _is_obj_straightline_traj_feasible(body, pose, end_position, obstacles, body_path)
        
        #if result == False:
            #raise Exception("pick body {} at pose {} is infeasible!!!!".format(body, pose.pose))
        
        return result
        
        #print "[Meiying::get_pick_feasible_test] return True"
        #return True
    return test


# depretecated, this is a manually defined test, used in previous version
def get_place_feasible_test(grouping, fixed=[], body_path={}, file_path=None):
    def test(body, start_furface, end_surface, *args):
        #if body == 8:
            #current_movables, _ = _get_movable_obstacles_from_args(*args)
            #raise Exception("test body 8 for place on surface {} from surface {}, with the other movable {}".format(end_surface, start_surface, current_movables[0]))                

        print "Meiying::get_place_feasible_test"
        
        area_feasible = _place_area_feasible(grouping, fixed, body_path, body, end_surface, file_path, *args)
        
        print "[Meiying::area_feasible]:", area_feasible
        if not area_feasible:
            return False
        
        #connection_feasible = _place_connection_feasible(body, start_furface, end_surface, body_path, fixed, grouping)
        #print "[Meiying::connection_feasible]:", connection_feasible
        #if not connection_feasible:
            #return False
        
        return True 
        #return True
    return test

# depretecated, this is a manually defined test, used in previous version
def get_move_feasible_test(fixed=[], body_path={}):
    def test(body, start_pose, end_pose, start_conf, end_conf, *args):
        return True
    return test

"""
Generators
"""
def _get_grasp_point(body, body_path):
    tool_name = _get_pcd_name(body, body_path)
    
    data_path = os.path.join(constants.get_learned_data_dir(), "tool.json")
    T_json_result = {}
    with open(data_path, "r") as read_file:
        T_json_result = json.load(read_file)
        for task in T_json_result.keys():
            for tool in T_json_result[task].keys():
                if tool == tool_name:
                    grasp = T_json_result[task][tool]["grasp"]
                    return (tuple(grasp[0]), tuple(grasp[1]))
    
    return ((0., 0., 0.), (0., 0., 0., 1.))

def get_grasp_gen(robot, body_path, grasp_name='top'):
    grasp_info = GRASP_INFO[grasp_name]
    tool_link = get_tool_link(robot)
    def gen(body):
        #print "[Meiying::get_grasp_gen] sampling grasps of body {}!!!".format(body)
        #grasp_poses = grasp_info.get_grasps(body)
        # TODO: continuous set of grasps
        #if body == 6:
            #print("\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!sample grasp for object 6!!")
        for grasp_pose in grasp_info.get_grasps(body):          
            Tbodycenter_bodygrasp = ((0., 0., 0.), (0., 0., 0., 1.))
            Tee_bodygrasp = grasp_pose
            Tbodycenter_bodygrasp = _get_grasp_point(body, body_path)
            Tee_bodycenter = multiply(Tee_bodygrasp, invert(Tbodycenter_bodygrasp))
            grasp_pose = Tee_bodycenter
            body_grasp = BodyGrasp(body, grasp_pose, grasp_info.approach_pose, robot, tool_link)
            #if body == 6:
                #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", body_grasp
                #print "Tee_bodycenter:", Tee_bodycenter
                #raw_input("continue?")
            yield (body_grasp,)
    return gen

# generate pick/place motion
def get_ik_fn(robot, fixed=[], teleport=False, num_attempts=10, body_path={}):
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)
    def fn(body, pose, grasp):
        print "[Meiying::meiying_primitives] sampling ik"
        obstacles = [body] + fixed
        gripper_pose = end_effector_from_body(pose.pose, grasp.grasp_pose)
        #if body == 6:
            #print "gripper_pose:", gripper_pose
            #raw_input("continue??")
        approach_pose = approach_from_grasp(grasp.approach_pose, gripper_pose)
        for _ in range(num_attempts):
            set_joint_positions(robot, movable_joints, sample_fn()) # Random seed
            # TODO: multiple attempts?
            q_approach = inverse_kinematics(robot, grasp.link, approach_pose)
            if (q_approach is None) or any(pairwise_collision(robot, b) for b in obstacles):
                if q_approach is None:
                    print "[Meiying::meiying_primitives] q_approach is None"
                else:
                    print "[Meiying::meiying_primitives] cannot approach due to collision"
                continue
            conf = BodyConf(robot, q_approach)
            conf.assign()
            q_grasp = inverse_kinematics(robot, grasp.link, gripper_pose)
            if (q_grasp is None) or any(pairwise_collision(robot, b) for b in obstacles):
                if q_grasp is None:
                    print "[Meiying::meiying_primitives] q_grasp is None"
                else:
                    print "[Meiying::meiying_primitives] cannot grasp due to collision"                
                continue
            if teleport:
                path = [q_approach, q_grasp]
            else:
                conf.assign()
                #direction, _ = grasp.approach_pose
                #path = workspace_trajectory(robot, grasp.link, point_from_pose(approach_pose), -direction,
                #                                   quat_from_pose(approach_pose))
                path = plan_direct_joint_motion(robot, conf.joints, q_grasp, obstacles=obstacles)
                if path is None:
                    print "[Meiying::meiying_primitives] cannot plan a path between q_grasp and q_approach"
                    if DEBUG_FAILURE: wait_if_gui('Approach motion failed')
                    continue
            command = Command([BodyPath(robot, path),
                               Attach(body, robot, grasp.link),
                               BodyPath(robot, path[::-1], attachments=[grasp])])
            body_approach_pose = BodyPose(pose.body, body_from_end_effector(approach_pose, grasp.grasp_pose))
            return (body_approach_pose, conf, command)
            # TODO: holding collisions
        print "[Meiying::meiying_primitives] ik return None!!!!!!"
        return None
    return fn

# generate place location
# TODO: replace
def get_place_object_gen(grouping, fixed=[], movables=[], body_path={}, file_path=None):
    def gen(body, surface, at_pose_fluents=[], *args):
        #print "Meiying::get_place_object_gen body {} on surface {}".format(body, surface)
        #print "Meiying::get_place_object_gen fixed: ", fixed
 
        if surface == fixed[0]: # can always put things on the table
            while True:
                #yield _generate_place_pose(body, surface, fixed=fixed, body_path=body_path)
                pose = sample_placement(body, surface)
                if pose is None:
                    continue
                position = list(pose[0])
                r = np.linalg.norm(np.array([position[0], position[1]]))
                if r > 0.707:
                    direction = np.array([position[0], position[1]]) / r * 0.707
                    pose = ((direction[0], direction[1], pose[0][2]), pose[1])
                if any(pairwise_collision(body, b, body_path=body_path) for b in fixed):
                    continue
                body_pose = BodyPose(body, pose)
                yield (body_pose,)            
        else:
            #print "[Meiying::get_place_object_gen] at_pose_fluents:", at_pose_fluents
            fluent_group = _grouping_atpose_fluents(at_pose_fluents)
            #print "[Meiying::get_place_object_gen] fluent_group:", fluent_group
            movable_bodies, _ = _get_movable_obstacles_from_fluent_group(body, surface, fluent_group)
            #print "[Meiying::get_place_object_gen] body:", body
            #print "[Meiying::get_place_object_gen] surface:", surface
            #print "[Meiying::get_place_object_gen] movable_bodies:", movable_bodies
            
            target_grouping = _get_place_grouping_from_movables(grouping, surface, body, body_path, movable_bodies)
            
            num_obstacles_on_surface = 0
            obstacle_pcds = []
            for obstacle in target_grouping.keys():
                if obstacle not in [FLOOR, CEILING, WALL_RIGHT, WALL_LEFT, WALL_FRONT, WALL_BACK, surface, body]:
                    num_obstacles_on_surface += 1
                    obstacle_pcds.append(target_grouping[obstacle][POINTCLOUD])
            
            if num_obstacles_on_surface == 0: # no other object on the surface
                while True:
                    pose = sample_placement(body, surface)
                    if (pose is None) or any(pairwise_collision(body, b, body_path=body_path) for b in fixed):
                        continue
                    body_pose = BodyPose(body, pose)
                    yield (body_pose,)             
            else:
                obstacles = target_grouping.keys()
                
                place_data = {}
                if not file_path is None:
                    try:
                        with open(file_path) as f:
                            place_data = json.load(f)
                    except:
                        pass
                
                obstacles = target_grouping.keys()
                
                # find whether this has been checked before
                is_found = False
                is_placeable = True
                all_area_feasible = True
                feasible_positions = []
                
                for obstacle in obstacles:
                    if obstacle not in fixed:
                        print "[Meiying::get_place_object_gen] obstacle not in the fixed set:", obstacle
                
                for key, value in place_data.items():
                    if value[BODY] == body and value[SURFACE] == surface and _is_bodies_same(obstacles, value[MOVABLES]) and _is_poses_same(value, target_grouping):
                        is_found = True
                        is_placeable = value[PLACEABLE]
                        all_area_feasible = value[ALL_AREA_FEASIBLE]
                        feasible_positions = value[FEASIBLE_POSITIONS]
                        break
                            
                if not is_found:
                    is_placeable, all_area_feasible, feasible_positions = _find_place_area(body, surface, obstacles, body_path)
                    _save_placeable(body, surface, target_grouping, place_data, is_placeable, all_area_feasible, feasible_positions, file_path)
                    
                if not is_placeable:
                    #yield None
                    print "[Meiying::get_place_object_gen] body {} not placeable on surface {}".format(body, surface)
                    return
                
                if all_area_feasible:
                    print "[Meiying::get_place_object_gen] body {} on surface {} all area feasible".format(body, surface)
                    while True:
                        #yield _generate_place_pose(body, surface, fixed=fixed, body_path=body_path)
                        pose = sample_placement(body, surface)
                        if (pose is None) or any(pairwise_collision(body, b, body_path=body_path) for b in fixed):
                            continue
                        body_pose = BodyPose(body, pose)
                        yield (body_pose,)
                else:
                    while True:
                        position = random.choice(feasible_positions)
                        yield (BodyPose(body, (position, (0., 0., 0., 1.))),) 
                        #theta = random.uniform(0., np.pi * 2)
                        #_, quaternion = _rotate_theta(start_pose.pose, theta)
                        #yield (BodyPose(body, (position, quaternion)),) # TODO
        #"""
                
    return gen

# generate move with an object attached
def get_holding_motion_gen(robot, fixed=[], teleport=False, self_collisions=True):
    def fn(conf1, conf2, body, grasp, pose1, pose2, fluents=[]):
        print "Meiying::get_holding_motion_gen"
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

# contact pose is tool pose in the object frame
#def get_tool_motion_gen(robot, fixed=[], teleport=False, num_attempts=10, body_path={}, contact_poses={}, manipulanda_tool_pose={}, tool_put_down_z={}):
    #movable_joints = get_movable_joints(robot)
    #sample_fn = get_sample_fn(robot, movable_joints)   
    #def fn(body, tool, body_pose1, body_pose2, tool_grasp, *args):
        #if not (tool in contact_poses.keys()):
            #return
        #if not (body in contact_poses[tool].keys()):
            #return
        
        #check_tool_body_collision = False
        #if tool in manipulanda_tool_pose.keys():
            #if body in manipulanda_tool_pose[tool].keys():
                #check_tool_body_collision = True

        #contact_pose = contact_poses[tool][body]
    
        ##print "~~~~~~~~~~~~~~~~~[Meiying::get_tool_motion_gen] body={}, tool={}, body_pose1={}, body_pose2={}~~~~~~~~~~~~~~~~~".format(body, tool, body_pose1.pose, body_pose2.pose)
        #body_pose3 = BodyPose(body, (body_pose2.pose[0], body_pose1.pose[1]))
        #tool_pose1, tool_pose2, contact_grasp_pose = _get_tool_use_config(body, tool, body_pose1.pose, body_pose3.pose, tool_grasp, contact_pose=contact_pose, manipulanda_tool_pose=manipulanda_tool_pose)
        
        #if check_tool_body_collision:
            #body_pcd = _get_pcd(body, body_path, _get_homogeneous_transformation_matrix_from_quaternion(np.array(body_pose1.pose[1]), np.array(body_pose1.pose[0])))
            #tool_pcd = _get_pcd(tool, body_path, _get_homogeneous_transformation_matrix_from_quaternion(np.array(tool_pose1[1]), np.array(tool_pose1[0])))
            #if is_pcd_collide(body_pcd, tool_pcd):
                #return None
            
        ##print "[Meiying::get_tool_motion_gen] tool_pose1:", tool_pose1
        ##print "[Meiying::get_tool_motion_gen] tool_pose2:", tool_pose2
        
        #obstacles = fixed
        
        ## approach object
        #gripper_pose = end_effector_from_body(body_pose1.pose, contact_grasp_pose.grasp_pose)
        ##approach_pose = approach_from_grasp(contact_grasp_pose.approach_pose, gripper_pose) 
        #approach_pose = end_effector_from_body(body_pose1.pose, contact_grasp_pose.approach_pose)
        #tool_approach_pose = body_from_end_effector(approach_pose, tool_grasp.grasp_pose)
        ##print "[Meiying::get_tool_motion_gen] gripper_pose: ", gripper_pose
        ##print "[Meiying::get_tool_motion_gen] approach_pose: ", approach_pose
        #approach_result = _get_straightline_path(robot, movable_joints, 
                                                 #sample_fn, 
                                                 #contact_grasp_pose.link, 
                                                 #approach_pose, 
                                                 #gripper_pose, 
                                                 #num_attempts=num_attempts, 
                                                 #obstacles=obstacles, 
                                                 #body_path=body_path,
                                                 #teleport=teleport)
        ##print "[Meiying::get_tool_motion_gen] approach_result:", approach_result
        #if approach_result is None:
            #return None
        
        #approach_q_start, approach_path, approach_q_end = approach_result
        
        ## tool_use, pull or push
        #tool_start_pose = gripper_pose
        #tool_end_pose = end_effector_from_body(body_pose3.pose, contact_grasp_pose.grasp_pose)
        #tool_result = _get_straightline_path(robot, movable_joints,
                                             #sample_fn, 
                                             #contact_grasp_pose.link, 
                                             #tool_start_pose, 
                                             #tool_end_pose, 
                                             #q_start=approach_q_end,
                                             #num_attempts=num_attempts, 
                                             #obstacles=obstacles, 
                                             #body_path=body_path,
                                             #teleport=teleport)
        ##print "[Meiying::get_tool_motion_gen] tool_result:", tool_result
        #if tool_result is None:
            #return None
        
        #tool_q_start, tool_path, tool_q_end = tool_result
        
        ## put tool down
        #put_start_pose = tool_end_pose
        ## stable z gives a weird result, so just hard code the z for now
        #tool_z = -0.025000000366475428
        #if tool in tool_put_down_z.keys():
            #tool_z = tool_put_down_z[tool]
        ##tool_put_down_pose = ((tool_pose2[0][0], tool_pose2[0][1], tool_put_down_pose_z), tool_pose2[1])
        #tool_put_down_pose = ((tool_pose2[0][0], tool_pose2[0][1], tool_z), tool_pose2[1])
        #put_end_pose = end_effector_from_body(tool_put_down_pose, tool_grasp.grasp_pose)
        #put_result = _get_straightline_path(robot, movable_joints,
                                            #sample_fn, 
                                            #contact_grasp_pose.link, 
                                            #put_start_pose, 
                                            #put_end_pose, 
                                            #q_start=tool_q_end,
                                            #num_attempts=num_attempts, 
                                            #obstacles=obstacles, 
                                            #body_path=body_path,
                                            #teleport=teleport)
        ##print "[Meiying::get_tool_motion_gen] put_result:", put_result
        
        ##print "[Meiying::tool_put_down_pose]:", tool_put_down_pose

        #if put_result is None:
            #return None
        
        #put_q_start, put_path, put_q_end = put_result
        
        #lift_start_pose = put_end_pose
        #lift_end_pose = ((put_end_pose[0][0], put_end_pose[0][1], put_end_pose[0][2] + 0.1), put_end_pose[1])
        #lift_result = _get_straightline_path(robot, movable_joints,
                                             #sample_fn, 
                                             #contact_grasp_pose.link, 
                                             #lift_start_pose, 
                                             #lift_end_pose, 
                                             #q_start=put_q_end,
                                             #num_attempts=num_attempts, 
                                             #obstacles=obstacles, 
                                             #body_path=body_path,
                                             #teleport=teleport)
        
        #if lift_result is None:
            #return None
        
        #lift_q_start, lift_path, lift_q_end = lift_result
        
        ## get return parameters
        #command = Command([BodyPath(robot, approach_path, attachments=[tool_grasp]),
                           #Attach(body, robot, contact_grasp_pose.link),
                           #BodyPath(robot, tool_path, attachments=[contact_grasp_pose, tool_grasp]),
                           #Detach(body, robot, contact_grasp_pose.link),
                           #BodyPath(robot, put_path, attachments=[tool_grasp]),
                           #Detach(tool, robot, tool_grasp.link),
                           #BodyPath(robot, lift_path)])
        #start_conf = BodyConf(robot, approach_q_start)
        #end_conf = BodyConf(robot, lift_q_end)
        #tool_final_body_pose = BodyPose(tool, tool_put_down_pose)
        #tool_apporach_body_pose = BodyPose(tool, tool_approach_pose)
        
        ##print "start_conf:"
        ##print approach_q_start

        #return body_pose3, tool_apporach_body_pose, tool_final_body_pose, start_conf, end_conf, command
        
    #return fn

def _get_tool_use_config(task, body, tool, body_pose1, body_pose2, tool_grasp, body_path):
    usage = _get_usage(task, body, body_path, tool)
    tool_pose1, tool_pose2 = _get_tool_use_poses(body, tool, body_pose1, body_pose2, usage)

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

def get_tool_motion_gen(robot, task="", fixed=[], teleport=False, num_attempts=10, body_path={}, tool_put_down_z={}):
    movable_joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, movable_joints)   
    def fn(body, tool, body_pose1, body_pose2, tool_grasp, *args):
        check_tool_body_collision = False
        #if tool in manipulanda_tool_pose.keys():
            #if body in manipulanda_tool_pose[tool].keys():
                #check_tool_body_collision = True

        #contact_pose = contact_poses[tool][body]
    
        #print "~~~~~~~~~~~~~~~~~[Meiying::get_tool_motion_gen] body={}, tool={}, body_pose1={}, body_pose2={}~~~~~~~~~~~~~~~~~".format(body, tool, body_pose1.pose, body_pose2.pose)
        body_pose3 = BodyPose(body, (body_pose2.pose[0], body_pose1.pose[1]))
        #tool_pose1, tool_pose2, contact_grasp_pose = _get_tool_use_config(body, tool, body_pose1.pose, body_pose3.pose, tool_grasp, contact_pose=contact_pose, manipulanda_tool_pose=manipulanda_tool_pose)
        tool_pose1, tool_pose2, contact_grasp_pose = _get_tool_use_config(task, body, tool, body_pose1.pose, body_pose3.pose, tool_grasp, body_path)
        
        if check_tool_body_collision:
            body_pcd = _get_pcd(body, body_path, _get_homogeneous_transformation_matrix_from_quaternion(np.array(body_pose1.pose[1]), np.array(body_pose1.pose[0])))
            tool_pcd = _get_pcd(tool, body_path, _get_homogeneous_transformation_matrix_from_quaternion(np.array(tool_pose1[1]), np.array(tool_pose1[0])))
            if is_pcd_collide(body_pcd, tool_pcd):
                return None
            
        #print "[Meiying::get_tool_motion_gen] tool_pose1:", tool_pose1
        #print "[Meiying::get_tool_motion_gen] tool_pose2:", tool_pose2
        
        obstacles = fixed
        
        # approach object
        gripper_pose = end_effector_from_body(body_pose1.pose, contact_grasp_pose.grasp_pose)
        #approach_pose = approach_from_grasp(contact_grasp_pose.approach_pose, gripper_pose) 
        approach_pose = end_effector_from_body(body_pose1.pose, contact_grasp_pose.approach_pose)
        tool_approach_pose = body_from_end_effector(approach_pose, tool_grasp.grasp_pose)
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
        
        tool_q_start, tool_path, tool_q_end = tool_result
        
        # put tool down
        put_start_pose = tool_end_pose
        # stable z gives a weird result, so just hard code the z for now
        tool_z = -0.025000000366475428 + 0.055
        if tool in tool_put_down_z.keys():
            tool_z = tool_put_down_z[tool]
        #tool_put_down_pose = ((tool_pose2[0][0], tool_pose2[0][1], tool_put_down_pose_z), tool_pose2[1])
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
        #print "[Meiying::get_tool_motion_gen] put_result:", put_result
        
        #print "[Meiying::tool_put_down_pose]:", tool_put_down_pose

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
        tool_apporach_body_pose = BodyPose(tool, tool_approach_pose)
        
        #print "start_conf:"
        #print approach_q_start

        return body_pose3, tool_apporach_body_pose, tool_final_body_pose, start_conf, end_conf, command
        
    return fn

def get_pull_tool_goal_gen(body_pose_path, grouping, body_path, pull_file_path=None, fixed=[], movable=[], task="pull", manipulanda_tool_pose={}):
    def gen(body, tool, surface, at_pose_fluents=[], *args): 
        # find body pose
        previous_body_poses = {}
        try:
            with open(body_pose_path) as f:
                previous_body_poses = json.load(f)
        except:
            pass
        
        fluent_group = _grouping_atpose_fluents(at_pose_fluents)
        candidates = fluent_group[body]
        current_pose = None
        
        if body not in previous_body_poses.keys():
            current_pose = fluent_group[body][0]
            previous_body_poses[body] = [current_pose]
        else:
            for candidate in candidates:
                if not _pose_contain(candidate, previous_body_poses[body]):
                    current_pose = candidate
                    previous_body_poses[body].append(current_pose)
                    break
        
        with open(body_pose_path, 'w') as json_file:
            json.dump(previous_body_poses, json_file, indent = 4, sort_keys=True)          
            
        if not current_pose is None:
            current_pose = previous_body_poses[body][-1]
        
        pose = BodyPose(body, current_pose)
        # get grouping
        
        target_grouping = _get_pull_grouping(body, tool, pose, grouping, fixed, body_path)
        
        num_obstacles_on_surface = 0
        obstacle_pcds = []
        for obstacle in target_grouping.keys():
            if obstacle not in [FLOOR, CEILING, WALL_RIGHT, WALL_LEFT, WALL_FRONT, WALL_BACK, tool, body]:
                num_obstacles_on_surface += 1
                obstacle_pcds.append(target_grouping[obstacle][POINTCLOUD])
            
        if num_obstacles_on_surface == 0: # no other object in this group
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
        else:
            pull_data = {}
            if not pull_file_path is None:
                try:
                    with open(pull_file_path) as f:
                        pull_data = json.load(f)
                except:
                    pass
            
            obstacles = target_grouping.keys()
            
            # find whether this has been checked before
            is_found = False
            is_pullable = True
            all_direction_feasible = True
            feasible_positions = {}
            #i = 0
            #print "# of items:", len(place_data.items())
            for key, value in pull_data.items():
                #print "i =", i
                #i += 1
                #print "value"
                #print value
                if value[BODY] == body and value[TOOL] == tool and _pose_equal(value[BODY_POSE], pose.pose):
                    if _is_bodies_same(obstacles, value[OBSTACLES]):
                        if _is_poses_same(value, target_grouping):
                            is_found = True
                            is_pullable = value[PULLABLE]
                            all_direction_feasible = value[ALL_DIRECTION_FEASIBLE]
                            feasible_positions = _format_feasible_positions(value[FEASIBLE_POSITIONS])
                            break
                        
            if not is_found:
                usage = _get_usage(task, body, body_path, tool)
                is_pullable, all_direction_feasible, feasible_positions = _find_pull_positions(body, pose, tool, obstacles, body_path, usage)
                _save_pullable(body, pose.pose, tool, target_grouping, pull_data, is_pullable, all_direction_feasible, feasible_positions, pull_file_path) 
            
            print "meiying::get_pull_feasible_test {} with tool {}: {}".format(body, tool, is_pullable)
            
            if not is_pullable:
                return
            
            if all_direction_feasible:
                while True:
                    #pose = sample_placement(body, surface)
                    #if (pose is None) or any(pairwise_collision(body, b, body_path=body_path) for b in fixed):
                        #continue
                    #body_pose = BodyPose(body, pose)
                    #yield (body_pose,)
                    pose = sample_placement(body, surface)
                    if (pose is None):
                        continue
                    position = list(pose[0])
                    position[1] = abs(position[1])
                    position[2] = stable_z(body, surface)
                    r = np.linalg.norm(np.array([position[0], position[1]]))
                    if r > 0.707:
                        direction = np.array([position[0], position[1]]) / r * 0.707
                        position = (direction[0], direction[1], position[2])
                    print "position sampled: ", position
                    pose = (tuple(position), pose[1])
                    set_pose(body, pose)
                    if any(pairwise_collision(body, b) for b in fixed):
                        continue
                    body_pose = BodyPose(body, pose)
                    yield (body_pose,)                    
            else:
                while True:
                    direction = random.choice(feasible_positions.keys())
                    distance = feasible_positions[direction] + 0.02
                    offset = np.array(direction) * distance
                    end_position = tuple(np.array(current_pose[0]) + offset)
                    end_pose = (end_position, current_pose[1])
                    yield (BodyPose(body, end_pose),) 
                    #theta = random.uniform(0., np.pi * 2)
                    #_, quaternion = _rotate_theta(start_pose.pose, theta)
                    #yield (BodyPose(body, (position, quaternion)),) # TODO
                
    return gen

def get_push_tool_goal_gen(body_pose_path, grouping, body_path, pull_file_path=None, fixed=[], movable=[], task="push", manipulanda_tool_pose={}):
    def gen(body, tool, surface, at_pose_fluents=[], *args): 
        # find body pose
        previous_body_poses = {}
        try:
            with open(body_pose_path) as f:
                previous_body_poses = json.load(f)
        except:
            pass
        
        fluent_group = _grouping_atpose_fluents(at_pose_fluents)
        candidates = fluent_group[body]
        current_pose = None
        
        if body not in previous_body_poses.keys():
            current_pose = fluent_group[body][0]
            previous_body_poses[body] = [current_pose]
        else:
            for candidate in candidates:
                if not _pose_contain(candidate, previous_body_poses[body]):
                    current_pose = candidate
                    previous_body_poses[body].append(current_pose)
                    break
        
        with open(body_pose_path, 'w') as json_file:
            json.dump(previous_body_poses, json_file, indent = 4, sort_keys=True)          
            
        if not current_pose is None:
            current_pose = previous_body_poses[body][-1]
        
        pose = BodyPose(body, current_pose)
        # get grouping
        
        target_grouping = _get_pull_grouping(body, tool, pose, grouping, fixed, body_path)
        
        num_obstacles_on_surface = 0
        obstacle_pcds = []
        for obstacle in target_grouping.keys():
            if obstacle not in [FLOOR, CEILING, WALL_RIGHT, WALL_LEFT, WALL_FRONT, WALL_BACK, tool, body]:
                num_obstacles_on_surface += 1
                obstacle_pcds.append(target_grouping[obstacle][POINTCLOUD])
            
        if num_obstacles_on_surface == 0: # no other object in this group
            while True:
                pose = sample_placement(body, surface)
                if (pose is None):
                    continue
                position = list(pose[0])
                position[0] = abs(position[0])
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
                #pose = ((0.3, 0.5, stable_z(body, surface)), (0., 0., 0., 1.))
                #body_pose = BodyPose(body, pose)
                #yield (body_pose,)
        else:
            pull_data = {}
            if not pull_file_path is None:
                try:
                    with open(pull_file_path) as f:
                        pull_data = json.load(f)
                except:
                    pass
            
            obstacles = target_grouping.keys()
            
            # find whether this has been checked before
            is_found = False
            is_pullable = True
            all_direction_feasible = True
            feasible_positions = {}
            #i = 0
            #print "# of items:", len(place_data.items())
            for key, value in pull_data.items():
                #print "i =", i
                #i += 1
                #print "value"
                #print value
                if value[BODY] == body and value[TOOL] == tool and _pose_equal(value[BODY_POSE], pose.pose):
                    if _is_bodies_same(obstacles, value[OBSTACLES]):
                        if _is_poses_same(value, target_grouping):
                            is_found = True
                            is_pullable = value[PULLABLE]
                            all_direction_feasible = value[ALL_DIRECTION_FEASIBLE]
                            feasible_positions = _format_feasible_positions(value[FEASIBLE_POSITIONS])
                            break
                        
            if not is_found:
                usage = _get_usage(task, body, body_path, tool)
                is_pullable, all_direction_feasible, feasible_positions = _find_pull_positions(body, pose, tool, obstacles, body_path, usage)
                _save_pullable(body, pose.pose, tool, target_grouping, pull_data, is_pullable, all_direction_feasible, feasible_positions, pull_file_path) 
            
            print "meiying::get_pull_feasible_test {} with tool {}: {}".format(body, tool, is_pullable)
            
            if not is_pullable:
                return
            
            if all_direction_feasible:
                while True:
                    pose = sample_placement(body, surface)
                    if (pose is None):
                        continue
                    position = list(pose[0])
                    position[1] = abs(position[1])
                    position[2] = stable_z(body, surface)
                    r = np.linalg.norm(np.array([position[0], position[1]]))
                    if r > 0.707:
                        direction = np.array([position[0], position[1]]) / r * 0.707
                        position = (direction[0], direction[1], position[2])
                    print "position sampled: ", position
                    pose = (tuple(position), pose[1])
                    set_pose(body, pose)
                    if any(pairwise_collision(body, b) for b in fixed):
                        continue
                    body_pose = BodyPose(body, pose)
                    yield (body_pose,)                    
            else:
                while True:
                    direction = random.choice(feasible_positions.keys())
                    distance = distance = feasible_positions[direction] + 0.02
                    offset = np.array(direction) * distance
                    end_position = tuple(np.array(current_pose[0]) + offset)
                    end_pose = (end_position, current_pose[1])
                    yield (BodyPose(body, end_pose),) 
                    #theta = random.uniform(0., np.pi * 2)
                    #_, quaternion = _rotate_theta(start_pose.pose, theta)
                    #yield (BodyPose(body, (position, quaternion)),) # TODO
                
    return gen

