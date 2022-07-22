#!/usr/bin/env python

from __future__ import print_function

from timeit import default_timer as timer

from pddlstream.algorithms.meta import solve, create_parser
from examples.pybullet.utils.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, \
    get_free_motion_gen, get_movable_collision_test, get_tool_link
from examples.pybullet.utils.pybullet_tools.utils import WorldSaver, connect, dump_body, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, \
    BLOCK_URDF, SMALL_BLOCK_URDF, DUCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, get_bodies, HideOutput, wait_for_user, KUKA_IIWA_URDF, add_data_path, load_pybullet, \
    LockRenderer, has_gui, draw_pose, draw_global_system
from pddlstream.language.generator import from_gen_fn, from_fn, empty_gen, from_test, universe_test
from pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object, negate_test
from pddlstream.language.constants import print_solution, PDDLProblem
from examples.pybullet.tamp.streams import get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    move_cost_fn, get_cfree_obj_approach_pose_test
from examples.pybullet.utils.pybullet_tools.meiying_primitives_affordance import FLOOR, CEILING, WALL_FRONT, WALL_BACK, WALL_LEFT, WALL_RIGHT
from examples.pybullet.utils.pybullet_tools.meiying_primitives_affordance import get_pick_feasible_test, get_generic_feasible_test, get_ik_fn, \
     get_holding_motion_gen, get_place_feasible_test, get_grasp_gen, get_place_object_gen, get_move_feasible_test, group_initial_bodies

def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed

def _get_fixed_in_bodies(bodies, fixed):
    fixed_in_bodies = []
    for body in bodies:
        if body in fixed:
            fixed_in_bodies.append(body)
    
    return fixed_in_bodies

def pddlstream_from_problem(robot, body_path, grouping, movable=[], teleport=False, grasp_name='top'):
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
        for surface in fixed:
            init += [('Stackable', body, surface)]
            if is_placement(body, surface):
                init += [('Supported', body, pose, surface)]

    init += [('Floor', fixed[0])]
    for surface in fixed:
        init += [('Region', surface)]

    for body in fixed:
        name = get_body_name(body)
        if 'sink' in name:
            init += [('Sink', body)]
        if 'stove' in name:
            init += [('Stove', body)]

    body = movable[0]
    init += [('Radish', movable[1])]
    goal = ('and',
            ('AtConf', conf),
            #('Holding', body),
            #('On', body, fixed[1]),
            #('On', body, fixed[2]),
            #('Cleaned', body),
            ('Cooked', body),
    )

    stream_map = {
        'sample-pose': from_gen_fn(get_place_object_gen(grouping, fixed, movable, body_path)),
        'sample-grasp': from_gen_fn(get_grasp_gen(robot, body_path, grasp_name)),

        'inverse-kinematics': from_fn(get_ik_fn(robot, fixed, teleport)),
        'plan-free-motion': from_fn(get_free_motion_gen(robot, fixed, teleport)),
        'plan-holding-motion': from_fn(get_holding_motion_gen(robot, fixed, teleport)),

        'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(body_path=body_path)),
        'test-cfree-approach-pose': from_test(get_cfree_obj_approach_pose_test(body_path=body_path)),
        'test-cfree-traj-pose': from_test(negate_test(get_movable_collision_test(body_path=body_path))), #get_cfree_traj_pose_test()),

        'TrajCollision': get_movable_collision_test(body_path=body_path),
        
        #'sample-pick-object-traj': from_gen_fn(get_pick_object_traj_gen()),
        #'test-pick-feasible': from_test(get_pick_feasible_test(grouping, fixed, fixed, body_path)),
        #'test-place-feasible': from_test(get_place_feasible_test(grouping, fixed, body_path)),
        'test-pick-feasible': from_test(get_generic_feasible_test(grouping, task="pick", surface=fixed, fixed=fixed, body_path=body_path)),
        'test-place-feasible': from_test(get_generic_feasible_test(grouping, task="place", surface=fixed, fixed=fixed, body_path=body_path)),
        'test-move-feasible': from_test(get_move_feasible_test(fixed, body_path)),
        #'test-place-connection-feasible': from_test(get_place_connection_feasible_test(grouping, fixed, body_path)),

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
        #sink = load_model(SINK_URDF, pose=Pose(Point(y=0.5, z=0.3))) sink on top of the object
        stove = load_model(STOVE_URDF, pose=Pose(Point(x=+0.5)))
        celery = load_model(BLOCK_URDF, fixed_base=False)
        radish = load_model(SMALL_BLOCK_URDF, fixed_base=False)
        #radish = load_model(DUCK_URDF, fixed_base=False)
        #cup = load_model('models/dinnerware/cup/cup_small.urdf',
        # Pose(Point(x=+0.5, y=+0.5, z=0.5)), fixed_base=False)   

    draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot)) # TODO: not working
    # dump_body(robot)
    # wait_for_user()

    body_names = {
        sink: 'sink',
        stove: 'stove',
        celery: 'celery',
        radish: 'radish',
    }
    
    body_path = {
        floor: 'models/meiying_short_floor.urdf',
        sink: SINK_URDF,
        stove: STOVE_URDF,
        celery: BLOCK_URDF,
        radish: SMALL_BLOCK_URDF
    }
    movable_bodies = [celery, radish]

    set_pose(celery, Pose(Point(y=0.5, z=stable_z(celery, floor))))
    #set_pose(celery, Pose(Point(y=1.0, z=stable_z(celery, floor))))
    set_pose(radish, Pose(Point(y=-0.5, z=stable_z(radish, floor))))
    print("radish pose z:", stable_z(radish, floor))
    #set_pose(radish, Pose(Point(x=0.5, y=0.5, z=stable_z(radish, floor))))

    initial_index_grouping =[[sink], [stove], [celery], [radish]]
    fixed = get_fixed(robot, movable_bodies)
    grouping = group_initial_bodies(body_path, initial_index_grouping, fixed)

    return robot, body_names, movable_bodies, body_path, grouping

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
    robot, names, movable, body_path, grouping = load_world()
    print('Objects:', names)
    saver = WorldSaver()

    problem = pddlstream_from_problem(robot, body_path, grouping, movable=movable, teleport=args.teleport)
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

    if not plan is None:
        command = postprocess_plan(plan)
        import os
        current_directory = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
        from examples.pybullet.utils.pybullet_tools.utils import save_command
        save_command(current_directory, command)

    """
    if (plan is None) or not has_gui():
        disconnect()
        return

    step_demo = False
    command = postprocess_plan(plan)
    if args.simulate:
        wait_for_user('Simulate?')
        command.control()
    else:
        wait_for_user('Execute?')
        #command.step()
        if step_demo:
            for step_plan in plan:
                wait_for_user('\tshow next step?: {}'.format(step_plan))
                step_command = postprocess_plan([step_plan])
                step_command.refine(num_steps=10).execute(time_step=0.001)
        else:
            command.refine(num_steps=10).execute(time_step=0.001)
    wait_for_user('Finish?')
    """
    
    disconnect()


if __name__ == '__main__':
    main()
