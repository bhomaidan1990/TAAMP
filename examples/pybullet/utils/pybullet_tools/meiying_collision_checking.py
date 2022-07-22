#!/usr/bin/env python

import os
import json
import copy

import open3d as o3d
import numpy as np
import transformations as tfs
import pybullet as p

MAX_DISTANCE = 0.0035
BB_INFO = 'models/bb.json'

CENTER = "center"
R = "r"
EXTENT = "extent"

def _is_point_in_bb(bb, point):
    min_bound = bb.get_min_bound()
    max_bound = bb.get_max_bound()
    rotation = bb.R
    
    rotation_inv = np.linalg.inv(rotation)
    
    min_bound_normalized = _rotate_point(rotation_inv, min_bound)
    max_bound_normalized = _rotate_point(rotation_inv, max_bound)
    point_normalized = _rotate_point(rotation_inv, point)
    
    for i in range(len(min_bound_normalized)):
        if point_normalized[i] < min(min_bound_normalized[i], max_bound_normalized[i]) or point_normalized[i] > max(min_bound_normalized[i], max_bound_normalized[i]):
            return False
    
    return True

# bb: o3d OrientedBoundingBox. The transform function is not implemented.
def _transform_bb(bb, transformation):
    copied_bb = copy.deepcopy(bb)
    
    position, rotation = _decompose_homogeneous_transformation_matrix_to_rotation_matrix(transformation)
    
    copied_bb.rotate(rotation)
    copied_bb.translate(position)
    
    return copied_bb

def _rotate_point(rotation, point):
    rotated_point = np.matmul(rotation, np.array([point]).T).T[0]
    
    return rotated_point

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

def _decompose_homogeneous_transformation_matrix_to_rotation_matrix(matrix):
    translation = np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
    rotation_matrix = matrix[:3, :3]
    return translation, rotation_matrix

def _is_bbs_far_away(bb1, bb2):
    bb1_half_diagonal = np.linalg.norm(bb1.extent) / 2.
    bb2_half_diagonal = np.linalg.norm(bb2.extent) / 2.
    
    center_distance = np.linalg.norm(bb1.get_center() - bb2.get_center())
    
    if center_distance >= bb1_half_diagonal + bb2_half_diagonal:
        return True
    
    return False

def _is_near_bbs_collide(bb1, bb2):
    if _is_point_in_bb(bb2, bb1.get_center()) or _is_point_in_bb(bb1, bb2.get_center()):
        return True
    
    for point in np.asarray(bb1.get_box_points()):
        if _is_point_in_bb(bb2, point):
            return True
    
    for point in np.asarray(bb2.get_box_points()):
        if _is_point_in_bb(bb1, point):
            return True
    
    return False    

# this one includes contain
def _is_bb_collide(bb1, bb2):
    if _is_bbs_far_away(bb1, bb2):
        return False
    
    if _is_near_bbs_collide(bb1, bb2):
        return True
    
    return False

def _is_obj_collide(body1, body1_bbs, body2, body2_bbs, body_path, visualize=False):
    if visualize:
        bbs = []
        bbs.append(_get_pcd(body1, body_path))
        bbs.append(_get_pcd(body2, body_path))
        bbs.extend(body1_bbs)
        bbs.extend(body2_bbs)
        o3d.visualization.draw_geometries(bbs, "_is_obj_collide on body1 {} and body2 {}".format(body1, body2))    
    
    #for body1_bb in body1_bbs:
        #for body2_bb in body2_bbs:
            #if visualize:
                #print "_is_bb_collide:", _is_bb_collide(body1_bb, body2_bb)
                #print "_pcd_collision:", _pcd_collision(body1, body2, body_path=body_path)
                #o3d.visualization.draw_geometries([body1_bb, body2_bb], "bb collision: {}; pcd collision {}".format(_is_bb_collide(body1_bb, body2_bb), _pcd_collision(body1, body2, body_path=body_path)))
            #if _is_bb_collide(body1_bb, body2_bb):
                #print "{} and {} bb collide".format(body1, body2)
                #return _pcd_collision(body1, body2, body_path=body_path)
    
    #return False
    is_bbs_far_away = True
    for body1_bb in body1_bbs:
        for body2_bb in body2_bbs:
            if not _is_bbs_far_away(body1_bb, body2_bb):
                is_bbs_far_away = False
                if _is_bb_collide(body1_bb, body2_bb):
                    return True
    
    if is_bbs_far_away:
        return False
    
    return _pcd_collision(body1, body2, body_path=body_path)

def _get_ply_path(urdf_path):
    basepath, urdf_filename = os.path.split(urdf_path)
    filename = urdf_filename.split(".")
    base_filename = filename[0]
    ply_filename = "{}.ply".format(base_filename)
    ply_path = os.path.join(basepath, ply_filename)
    
    return ply_path

def get_model_path(rel_path): # TODO: add to search path
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, '..', rel_path)

def _get_preprocessed_bbs():
    bb_path = get_model_path(BB_INFO)
    if os.path.exists(bb_path):
        with open(bb_path) as f:
            return json.load(f) 
    return None

def _get_pcd_path(pybullet_uniqueid, body_path):
    ply_path = body_path[pybullet_uniqueid]
    if os.path.splitext(ply_path)[1][1:] == "urdf":
        urdf_path = ply_path
        ply_path = get_model_path(_get_ply_path(urdf_path))

    return os.path.abspath(ply_path)

def _get_pose_from_id(pybullet_uniqueid):
    return p.getBasePositionAndOrientation(pybullet_uniqueid)

def _get_homogeneous_transformation_matrix_from_quaternion(rotation_quaternion, translation_vector):
    # translation_vector: np.array([1, 2, 3])
    alpha, beta, gamma = tfs.euler_from_quaternion(rotation_quaternion)
    rotation_matrix = tfs.euler_matrix(alpha, beta, gamma)

    result = rotation_matrix
    result[:3, 3] = translation_vector

    return result

def _get_pcd_path(pybullet_uniqueid, body_path):
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
    for pcd_bb_info in pcd_bbs_info:
        center = np.array(pcd_bb_info[CENTER])
        rotation = np.array(pcd_bb_info[R])
        extent = np.array(pcd_bb_info[EXTENT])
        bb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        
        bb = _transform_bb(bb, transformation)
        bbs.append(bb)
    
    return bbs

def _is_obj_env_collide(body, obstacles, body_bbs, body_pose, obstacles_bbs, body_path):
    transformed_body_bbs = []
    
    position = list(body_pose[0])
    quaternion = list(body_pose[1])
    body_transformation = _get_homogeneous_transformation_matrix_from_quaternion(quaternion, position)    
    
    for body_bb in body_bbs:
        copied_body_bb = _transform_bb(body_bb, body_transformation)
        transformed_body_bbs.append(copied_body_bb)

    for obstacle_index in range(len(obstacles_bbs)):
        if _is_obj_collide(body, transformed_body_bbs, obstacles[obstacle_index], obstacles_bbs[obstacle_index], body_path):
            return True
    
    return False

def _is_pick_obj_env_collide(body_pcd, obstacle_pcd):
    distance = np.asarray(body_pcd.compute_point_cloud_distance(obstacle_pcd))
    
    min_distance = min(distance)
    
    return min(distance) < MAX_DISTANCE, min_distance
    
def _pcd_collision(body1, body2, body_path):
    body1_pcd = _get_pcd(body1, body_path)
    body2_pcd = _get_pcd(body2, body_path)
    
    return is_pcd_collide(body1_pcd, body2_pcd)

def is_pcd_collide(pcd1, pcd2):
    distance = np.asarray(pcd1.compute_point_cloud_distance(pcd2))

    return min(distance) < MAX_DISTANCE    

def object_collision(body1, body2, body_path):
    body1_bbs = _get_body_bbs(body1, body_path) 
    body2_bbs = _get_body_bbs(body2, body_path)
    
    #if (body1 == 5 and body2 == 1) or (body1 == 1 and body2 == 5):
        #result = _is_obj_collide(body1, body1_bbs, body2, body2_bbs, body_path)
        #body1_pcd = _get_pcd(body1, body_path=body_path)
        #body2_pcd = _get_pcd(body2, body_path=body_path)
        #to_visualize = [body1_bbs[0], body2_bbs[0], body1_pcd, body2_pcd]
        #o3d.visualization.draw_geometries(to_visualize, "body 4 and 5 are collide? {}".format(result))
        
    return _is_obj_collide(body1, body1_bbs, body2, body2_bbs, body_path)