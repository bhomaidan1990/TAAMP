#!/usr/bin/env python

from __future__ import print_function

from timeit import default_timer as timer
import numpy as np

from pddlstream.algorithms.meta import solve, create_parser
from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, get_movable_collision_test, get_tool_link, \
    get_tool_motion_gen, get_cfree_tool_pose_test, get_cfree_tool_traj_test, get_pull_tool_goal_gen, get_push_tool_goal_gen
from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_body, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, \
    BLOCK_URDF, SMALL_BLOCK_URDF, TOOL_TARGET_URDF, PULL_CHAIN_TOOL_URDF, PUSH_TOOL_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, OCCLUDER_URDF, TUNNEL_CHAIN_URDF, TUNNEL_URDF, get_bodies, HideOutput, wait_for_user, KUKA_IIWA_URDF, add_data_path, load_pybullet, \
    LockRenderer, has_gui, draw_pose, draw_global_system
#from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_body, get_pose, set_pose, Pose, \
     #Point, set_default_camera, stable_z, \
     #BLOCK_OBJ, SMALL_BLOCK_OBJ, get_configuration, SINK_OBJ, STOVE_OBJ, load_model, is_placement, get_body_name, \
     #disconnect, DRAKE_IIWA_URDF, get_bodies, HideOutput, wait_for_user, KUKA_IIWA_URDF, add_data_path, load_pybullet, \
     #LockRenderer, has_gui, draw_pose, draw_global_system
from pddlstream.language.generator import from_gen_fn, from_fn, empty_gen, from_test, universe_test
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object, negate_test
from pddlstream.language.constants import print_solution, PDDLProblem
from examples.pybullet.tamp.streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    move_cost_fn, get_cfree_obj_approach_pose_test

def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed

def place_movable(certified):
    placed = []
    for literal in certified:
        if literal[0] == 'not':
            fact = literal[1]
            if fact[0] == 'trajcollision':
                _, b, p = fact[1:]
                set_pose(b, p.pose)
                placed.append(b)
    return placed

def get_free_motion_synth(robot, movable=[], teleport=False):
    fixed = get_fixed(robot, movable)
    def fn(outputs, certified):
        assert(len(outputs) == 1)
        q0, _, q1 = find_unique(lambda f: f[0] == 'freemotion', certified)[1:]
        obstacles = fixed + place_movable(certified)
        free_motion_fn = get_free_motion_gen(robot, obstacles, teleport)
        return free_motion_fn(q0, q1)
    return fn

def get_holding_motion_synth(robot, movable=[], teleport=False):
    fixed = get_fixed(robot, movable)
    def fn(outputs, certified):
        assert(len(outputs) == 1)
        q0, _, q1, o, g = find_unique(lambda f: f[0] == 'holdingmotion', certified)[1:]
        obstacles = fixed + place_movable(certified)
        holding_motion_fn = get_holding_motion_gen(robot, obstacles, teleport)
        return holding_motion_fn(q0, q1, o, g)
    return fn

def get_pick_object_traj_gen():
    def fn(o, p, fluents=[]):
        print("[Meiying::get_pick_object_traj_gen] PIKA sampling pick motion!!!")
        object_pose = p.pose
        obj_traj = None
        #yield
        return []
    return fn 

#######################################################

def pddlstream_from_problem(robot, surface, body_path, tool_put_down_z={}, manipulanda_tool_pose={}, tool_grasp_pose={}, contact_poses={}, movable=[], pull_tool=None, push_tool=None, teleport=False, grasp_name='top'):
    #assert (not are_colliding(tree, kin_cache))

    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'stream.pddl'))
    constant_map = {}

    print('Robot:', robot)
    conf = BodyConf(robot, get_configuration(robot))
    init = [('CanMove',),
            ('Conf', conf),
            ('AtConf', conf),
            ('HandEmpty',)]

    fixed = get_fixed(robot, movable)
    print('Movable:', movable)
    print('Fixed:', fixed)
    init += [('Floor', fixed[0])] # the first one must be floor
    
    for body in movable:
        pose = BodyPose(body, get_pose(body))
        init += [('Graspable', body),
                 ('Pose', body, pose),
                 ('AtPose', body, pose),
                 ('Movable', body)]
        for top in surface:
            init += [('Stackable', body, top)]
            if is_placement(body, top):
                init += [('Supported', body, pose, top)]

    for body in fixed:
        name = get_body_name(body)
        if 'sink' in name:
            init += [('Sink', body)]
        if 'stove' in name:
            init += [('Stove', body)]

    if not pull_tool is None:
        init += [('PullTool', pull_tool)]
    if not push_tool is None:
        init += [('PushTool', push_tool)]    
        
    for body1 in movable:
        for body2 in movable:
            if body1 != body2:
                init += [('Different', body1, body2)]
        
    #if not push_tool is None:
        #init += [('Tool', push_tool)]

    body = movable[0]
    init += [('Radish', movable[1])]
    init += [('Manipulanda', movable[0])]
    init += [('Manipulanda', pull_tool)]
    #goal = ('and',
            #('AtConf', conf),
            ##('Holding', body),
            ##('On', body, fixed[1]),
            ##('On', body, fixed[2]),
            ##('Cleaned', body),
            #('Cooked', body),
    #)
    goal = ('and',
            ('AtConf', conf),
            #('Holding', body),
            #('Holding', pull_tool),
            #('Pull', pull_tool),
            #('On', body, fixed[1]),
            #('On', body, fixed[2]),
            #('Cleaned', body),
            ('Cooked', body),
    )

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(fixed, movable)),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name, tool_grasp_pose=tool_grasp_pose)),
        #'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport, body_path={})),
        'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport, body_path=body_path)),
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),
        
        'sample-pull-tool-goal': from_gen_fn(get_pull_tool_goal_gen(fixed, movable)),
        'plan-pull-ik': from_fn(get_tool_motion_gen(robot, fixed=fixed, body_path=body_path, contact_poses=contact_poses, tool_put_down_z=tool_put_down_z)),
        'test-cfree-pull-pose': from_test(get_cfree_tool_pose_test(body_path=body_path)),
        'test-cfree-pull-traj-pose': from_test(get_cfree_tool_traj_test(body_path=body_path)),
        
        'sample-push-tool-goal': from_gen_fn(get_push_tool_goal_gen(fixed, movable, manipulanda_tool_pose=manipulanda_tool_pose)),
        'plan-push-ik': from_fn(get_tool_motion_gen(robot, fixed=fixed, body_path=body_path, contact_poses=contact_poses, manipulanda_tool_pose=manipulanda_tool_pose)),
        'test-cfree-push-pose': from_test(get_cfree_tool_pose_test(body_path=body_path)),
        'test-cfree-push-traj-pose': from_test(get_cfree_tool_traj_test(body_path=body_path)),        
        
        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(body_path=body_path)),
        'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test(body_path=body_path)),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test(body_path=body_path))), #get_cfree_traj_pose_test()),

        'TrajCollision': get_movable_collision_test(body_path=body_path),
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


#######################################################

def load_world():
    # TODO: store internal world info here to be reloaded
    set_default_camera()
    draw_global_system()
    with HideOutput():

        #add_data_path()
        robot = load_model(DRAKE_IIWA_URDF, fixed_base=True) # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        floor = load_model('models/meiying_short_floor.urdf')
        sink = load_model(SINK_URDF, pose=Pose(Point(x=-0.5, y=-0.3)))
        stove = load_model(STOVE_URDF, pose=Pose(Point(x=+0.5, y=-0.3)))
        celery = load_model(TOOL_TARGET_URDF, fixed_base=False)
        radish = load_model(SMALL_BLOCK_URDF, fixed_base=False)
        pull_tool = load_model(PULL_CHAIN_TOOL_URDF, fixed_base=False)
        push_tool = load_model(PUSH_TOOL_URDF, fixed_base=False)
        tunnel = load_model(TUNNEL_CHAIN_URDF, pose=Pose(Point(x=0.09, y=0.5)))
        
        #occluder = load_model(OCCLUDER_URDF, fixed_base=False)
        #set_pose(occluder, Pose(Point(x=-0.66, y=0.66, z=stable_z(occluder, floor))))        
        #cup = load_model('models/dinnerware/cup/cup_small.urdf',
        # Pose(Point(x=+0.5, y=+0.5, z=0.5)), fixed_base=False)
        """
        #add_data_path()
        robot = load_model(DRAKE_IIWA_URDF, fixed_base=True) # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        floor = load_model('models/short_floor.obj')
        sink = load_model(SINK_OBJ, pose=Pose(Point(x=-0.5)))
        stove = load_model(STOVE_OBJ, pose=Pose(Point(x=+0.5)))
        celery = load_model(BLOCK_OBJ, fixed_base=False)
        radish = load_model(SMALL_BLOCK_OBJ, fixed_base=False)
        """

    draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot)) # TODO: not working
    # dump_body(robot)
    # wait_for_user()

    body_names = {
        sink: 'sink',
        stove: 'stove',
        celery: 'celery',
        radish: 'radish',
        pull_tool: "pull_tool",
        push_tool: "push_tool",
        tunnel: "tunnel"
        #occluder: "occluder"
    }
    
    body_path = {
        floor: 'models/meiying_short_floor.urdf',
        sink: SINK_URDF,
        stove: STOVE_URDF,
        celery: TOOL_TARGET_URDF,
        radish: SMALL_BLOCK_URDF,
        pull_tool: PULL_CHAIN_TOOL_URDF,
        push_tool: PUSH_TOOL_URDF,
        tunnel: TUNNEL_CHAIN_URDF,
        #occluder: OCCLUDER_URDF
    }    
    movable_bodies = [celery, radish, push_tool, pull_tool]

    #set_pose(pull_tool, Pose(Point(y=0.3, z=stable_z(pull_tool, floor))))
    #print("!!!!!!!!!!!!!!!!!pull_tool:", pull_tool)
    #print("!!!!!!!!!!!!!!!!!floor:", floor)
    #print("!!!!!!!!!!!!!!!!!pull_tool z:", stable_z(pull_tool, floor))
    #set_pose(celery, Pose(Point(x=-0.66, y=0.66, z=stable_z(celery, floor))))
    #set_pose(celery, Pose(Point(y=1.0, z=stable_z(celery, floor))))
    
    set_pose(pull_tool, Pose(Point(x=-0.05, y=0.4, z=stable_z(pull_tool, floor))))
    set_pose(push_tool, Pose(Point(x=-0.25, z=stable_z(push_tool, floor))))
    set_pose(celery, Pose(Point(x=-0.66, y=0.66, z=stable_z(celery, floor))))
    
    set_pose(radish, Pose(Point(y=-0.5, z=stable_z(radish, floor))))
    
    Tpush_tool = np.array([[1., 0., 0., -0.03 ],
                           [0., 1., 0.,  0.0   ],
                           [0., 0., 0., -0.06 ],
                           [0., 0., 0.,  1.    ]])
    
    Tpull_celery = np.array([[1., 0., 0.,  0.11 ],
                             [0., 1., 0.,  0.   ],
                             [0., 0., 0.,  0.055],
                             [0., 0., 0.,  1.   ]])
    
    contact_poses = {pull_tool: {celery: Tpull_celery}, push_tool: {pull_tool: Tpush_tool}}
    
    surface = [floor, sink, stove]
    
    tool_grasp_pose = {push_tool: ((-0.19, 0., 0.), (0., 0., 0., 1.)), pull_tool: ((0.1, 0.1, 0.), (0., 0., 0., 1.))}

    Tpull_manipulanda_point = ((0.175, 0.1, 0.), (0., 0., 0., 1.))
    manipulanda_tool_pose = {push_tool: {pull_tool: [Tpull_manipulanda_point, 0.05]}}
    
    tool_put_down_z = {pull_tool: 0.034}
    
    return robot, body_names, movable_bodies, body_path, pull_tool, push_tool, contact_poses, surface, tool_grasp_pose, manipulanda_tool_pose, tool_put_down_z

def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name == 'place':
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick', 'pull', 'push']:
            paths += args[-1].body_paths
    return Command(paths)

def get_command_paths(step):
    paths = []
    name, args = step
    if name == 'place':
        paths += args[-1].reverse().body_paths
    elif name in ['move', 'move_free', 'move_holding', 'pick', 'pull', 'push']:
        paths += args[-1].body_paths
    return paths

#######################################################

def main():
    parser = create_parser()
    parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
    parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
    parser.add_argument('-simulate', action='store_true', help='Simulates the system')
    parser.add_argument('-viewer', action='store_true', help='Enable the viewer and visualizes the plan')
    args = parser.parse_args()
    print('Arguments:', args)

    connect(use_gui=args.viewer)
    robot, names, movable, body_path, pull_tool, push_tool, contact_poses, surface, tool_grasp_pose, manipulanda_tool_pose, tool_put_down_z = load_world()
    print('Objects:', names)
    saver = WorldSaver()

    problem = pddlstream_from_problem(robot, surface, body_path, tool_put_down_z=tool_put_down_z, manipulanda_tool_pose=manipulanda_tool_pose, tool_grasp_pose=tool_grasp_pose, contact_poses=contact_poses, movable=movable, pull_tool=pull_tool, push_tool=push_tool, teleport=args.teleport)
    _, _, _, stream_map, init, goal = problem
    print('Init:', init)
    print('Goal:', goal)
    print('Streams:', str_from_object(set(stream_map)))

    start = timer()
    with Profiler():
        with LockRenderer(lock=not args.enable):
            solution = solve(problem, algorithm=args.algorithm, max_time=600., unit_costs=args.unit, success_cost=INF)
            saver.restore()
    print_solution(solution)
    plan, cost, evaluations = solution
    end = timer()
    print("=========================time: {}s==========================".format(end - start))

    """
    if not plan is None:
        command = postprocess_plan(plan)
        import os
        current_directory = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
        from examples.pybullet.utils.pybullet_tools.utils import save_command
        save_command(current_directory, command)

    """
    wait_for_user('Get result?')
    if (plan is None) or not has_gui():
        disconnect()
        return

    command = postprocess_plan(plan)
    if args.simulate:
        wait_for_user('Simulate?')
        command.control()
    else:
        wait_for_user('Execute?')
        #command.step()
        #for step in plan:
            #wait_for_user('Execute {}?'.format(step))
            #paths = get_command_paths(step)
            #for path in paths:
                #wait_for_user('\t{}?'.format(path))
                #current_command = Command([path])
                #current_command.refine(num_steps=10).execute(time_step=0.001)
        command.refine(num_steps=10).execute(time_step=0.001)
    wait_for_user('Finish?')

    disconnect()

if __name__ == '__main__':
    main()
