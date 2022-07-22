import open3d as o3d
import numpy as np

def _merge_pcds(pcds):
    points = []
    
    for pcd in pcds:
        points.extend(list(np.asarray(pcd.points)))
    
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.array(points))
    
    return merged_pcd

# extent: half of the actual extent
def _get_axis_range(extent, n, is_fixed=False):
    if is_fixed:
        return np.ones(n) * extent
    return np.random.uniform(-extent, extent, size=(n, ))

def get_circle_array(center, radius, n=10000):
    n_pnts = int(4 * radius * radius * n)
    square = np.array([_get_axis_range(radius, n_pnts), _get_axis_range(radius, n_pnts), -_get_axis_range(0., n_pnts, is_fixed=True)]).T
    
    length = np.linalg.norm(square, axis=1)
    circle_index = np.where(length <= radius)
    
    circle = square[circle_index]
    circle += center
    
    return circle

def generate_cylinder(center, radius, height, n=10000):
    top = get_circle_array(np.array([0., 0., height / 2.]), radius, n=n)
    bottom = get_circle_array(np.array([0., 0., -height / 2.]), radius, n=n)
    
    n_pnts = int(2 * np.pi * radius * height * n)
    angles = _get_axis_range(np.pi, n_pnts)
    zs = _get_axis_range(height / 2., n_pnts)
    
    side_points = []
    for i in range(n_pnts):
        angle = angles[i]
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        z = zs[i]
        side_points.append([x, y, z])
    side = np.array(side_points)
    
    cylinder = np.vstack([top, bottom, side])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cylinder)
    pcd.translate(center)
    
    return pcd    

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

tool_use_target = generate_cylinder(np.array([0., 0., 0.]), 0.079 / 2., 0.199)
tool_use_target.paint_uniform_color((0.1, 0.8, 0.1))
#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(circle)
#o3d.visualization.draw_geometries([cylinder])

L_pull_tool_function = generate_cube([-0.025, 0., 0.055], [0.049, 0.249, 0.049])
L_pull_tool_handle = generate_cube([0.225, 0.1, 0.055], [0.449, 0.049, 0.049])
L_pull_tool = _merge_pcds([L_pull_tool_function, L_pull_tool_handle])
L_pull_tool.translate(np.array([-0.04, 0., 0.]))
L_pull_tool.paint_uniform_color((0., 0.392, 0.,))
#o3d.visualization.draw_geometries([L_pull_tool, tool_use_target])

L_push_tool_function = generate_cube([-0.025, 0., 0.055], [0.049, 0.249, 0.049])
L_push_tool_handle = generate_cube([-0.225, -0.1, 0.055], [0.449, 0.049, 0.049])
L_push_tool = _merge_pcds([L_push_tool_function, L_push_tool_handle])
L_push_tool.translate(np.array([-0.04, 0., 0.]))
L_push_tool.paint_uniform_color((0., 0.392, 0.,))
o3d.visualization.draw_geometries([L_push_tool, L_pull_tool, tool_use_target])