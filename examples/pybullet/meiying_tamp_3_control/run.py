#!/usr/bin/env python

from __future__ import print_function

from timeit import default_timer as timer

from pddlstream.algorithms.meta import solve, create_parser
from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_stable_gen, get_ik_fn, get_free_motion_gen, \
    get_holding_motion_gen, get_movable_collision_test, get_tool_link
from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_body, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, \
    BLOCK_URDF, SMALL_BLOCK_URDF, SHORT_SMALL_BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, OCCLUDER_URDF, DUCK_URDF, get_bodies, HideOutput, wait_for_user, KUKA_IIWA_URDF, add_data_path, load_pybullet, \
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

def pddlstream_from_problem(robot, surface, body_path, movable=[], teleport=False, grasp_name='top'):
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

    body = movable[0]
    goal = ('and',
            ('AtConf', conf),
            #('Holding', body),
            #('On', body, fixed[1]),
            #('On', body, fixed[2]),
            #('Cleaned', body),
            ('Cooked', body),
    )

    stream_map = {
        'sample-pose': from_gen_fn(get_stable_gen(fixed, movable)),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, grasp_name)),
        'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport, body_path=body_path)),
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test()),
        'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test()),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test())), #get_cfree_traj_pose_test()),

        'TrajCollision': get_movable_collision_test(),
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
        sink = load_model(SINK_URDF, pose=Pose(Point(x=-0.5)))
        stove = load_model(STOVE_URDF, pose=Pose(Point(x=+0.5)))
        celery = load_model(BLOCK_URDF, fixed_base=False)
        #duck = load_model(DUCK_URDF, fixed_base=False)
        radish_1 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)
        radish_2 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)
        radish_3 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)
        radish_4 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)
        radish_5 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)
        radish_6 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)
        radish_7 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)
        radish_8 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)
        #radish_9 = load_model(SHORT_SMALL_BLOCK_URDF, fixed_base=False)

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
        #duck: 'duck',
        radish_1: 'radish_1',
        radish_2: 'radish_2',
        radish_3: 'radish_3',
        radish_4: 'radish_4',
        radish_5: 'radish_5',
        radish_6: 'radish_6',
        radish_7: 'radish_7',
        radish_8: 'radish_8',
        #radish_9: 'radish_9',
    }
    body_path = {
        floor: 'models/meiying_short_floor.urdf',
        sink: SINK_URDF,
        stove: STOVE_URDF,
        celery: BLOCK_URDF,
        radish_1: SHORT_SMALL_BLOCK_URDF,
        radish_2: SHORT_SMALL_BLOCK_URDF,
        radish_3: SHORT_SMALL_BLOCK_URDF,
        radish_4: SHORT_SMALL_BLOCK_URDF,
        radish_5: SHORT_SMALL_BLOCK_URDF,
        radish_6: SHORT_SMALL_BLOCK_URDF,
        radish_7: SHORT_SMALL_BLOCK_URDF,
        radish_8: SHORT_SMALL_BLOCK_URDF,
        #radish_9: SHORT_SMALL_BLOCK_URDF
    }     
    #movable_bodies = [celery, radish_1, radish_2, radish_3, radish_5, radish_6, radish_7, radish_8, radish_9]
    movable_bodies = [celery]
    
    """
    space = 0.1-0.02
    set_pose(celery, Pose(Point(y=0.5, z=stable_z(celery, floor))))
    #set_pose(celery, Pose(Point(y=1.0, z=stable_z(celery, floor))))
    #set_pose(duck, Pose(Point(x=-0.5, z=stable_z(duck, sink))))
    set_pose(radish_1, Pose(Point(x=-0.5+space, z=stable_z(radish_1, sink))))
    set_pose(radish_2, Pose(Point(x=-0.5-space, z=stable_z(radish_2, sink))))
    set_pose(radish_3, Pose(Point(x=-0.5, y=space, z=stable_z(radish_3, sink))))
    set_pose(radish_4, Pose(Point(x=-0.5+space, y=space, z=stable_z(radish_4, sink))))
    set_pose(radish_5, Pose(Point(x=-0.5-space, y=space, z=stable_z(radish_5, sink))))
    set_pose(radish_6, Pose(Point(x=-0.5, y=-space, z=stable_z(radish_6, sink))))
    set_pose(radish_7, Pose(Point(x=-0.5+space, y=-space, z=stable_z(radish_7, sink))))
    set_pose(radish_8, Pose(Point(x=-0.5-space, y=-space, z=stable_z(radish_8, sink))))
    #set_pose(radish_9, Pose(Point(x=-0.5, z=stable_z(radish_9, sink))))
    """
    
    space = 0.1-0.02
    set_pose(celery, Pose(Point(y=0.5, z=stable_z(celery, floor))))
    set_pose(radish_1, Pose(Point(x=-0.5+space, z=stable_z(radish_1, sink))))
    #set_pose(radish_2, Pose(Point(x=-0.5-space-0.02, z=stable_z(radish_2, sink))))
    set_pose(radish_2, Pose(Point(x=-0.5-space, z=stable_z(radish_2, sink))))
    set_pose(radish_3, Pose(Point(x=-0.5, y=space+0.02, z=stable_z(radish_3, sink))))
    set_pose(radish_4, Pose(Point(x=-0.5+space, y=space, z=stable_z(radish_4, sink))))
    set_pose(radish_5, Pose(Point(x=-0.5-space, y=space, z=stable_z(radish_5, sink))))
    set_pose(radish_6, Pose(Point(x=-0.5, y=-space-0.02, z=stable_z(radish_6, sink))))
    set_pose(radish_7, Pose(Point(x=-0.5+space, y=-space, z=stable_z(radish_7, sink))))
    set_pose(radish_8, Pose(Point(x=-0.5-space, y=-space, z=stable_z(radish_8, sink))))
    #set_pose(radish_9, Pose(Point(x=-0.5, z=stable_z(radish_9, sink))))
    
    surface = [floor, sink, stove]
    
    return robot, body_names, movable_bodies, body_path, surface

def postprocess_plan(plan):
    paths = []
    for name, args in plan:
        if name == 'place':
            paths += args[-1].reverse().body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            paths += args[-1].body_paths
    return Command(paths)

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
    robot, names, movable, both_path, surface = load_world()
    print('Objects:', names)
    saver = WorldSaver()

    problem = pddlstream_from_problem(robot, surface, both_path, movable=movable, teleport=args.teleport)
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
        command.refine(num_steps=10).execute(time_step=0.001)
    wait_for_user('Finish?')

    disconnect()

if __name__ == '__main__':
    main()
