import random
import numpy as np

def close_to(m, n, error=1e-6):
    return m >= n - error and m <= n + error

def is_unit_vector(v, error=1e-6):
    """
    given a vector, return whether it is a unit vector

    Parameters:
    v: vector

    returns:
    boolean
    """
    #print "length: ", np.linalg.norm(v)
    #print "length - 1.: ", np.linalg.norm(v) - 1
    #print "error: ", error
    return close_to(np.linalg.norm(v), 1, error)

def is_zero_vector(v, error=1e-6):
    """
    given a vector, return whether it is a zero vector

    Parameters:
    v: vector

    returns:
    boolean
    """
    #print "[kinematics][is_zero_vector] v length: ", np.linalg.norm(v)
    return close_to(np.linalg.norm(v), 0, error)

def is_identity_matrix(R, error=1e-6):
    assert(R.shape[0] == R.shape[1])

    return R[0][0] >= 1 - error and R[1][1] >= 1 - error and R[2][2] >= 1 - error
    #return np.allclose(R, np.identity(R.shape[0]), 0, error)

def get_distance(a, b):
    return np.linalg.norm(a - b)

def get_skew_symmetric_matrix(w):
    assert(w.shape[0] == 1)
    return np.cross(w, -np.identity(w.shape[1]))

def get_vector_from_skew_symmetric_matrix(R):
    """
    currently only works for 3*3 matrix
    """
    return np.array([[R[2, 1], R[0, 2], R[1, 0]]])

def get_homogenous_transformation_matrix(R, p):
    assert(R.shape[0] == R.shape[1])
    assert(R.shape[0] == p.shape[1])
    #R = np.r_[R, np.zeros((1, R.shape[0]))]
    #p = np.r_[p.T, [[1]]]
    return np.c_[np.r_[R, np.zeros((1, R.shape[0]))], np.r_[p.T, [[1]]]]

def get_homogenous_transformation_matrix_inverse(T):
    R, p = get_R_p_from_matrix(T)
    return get_homogenous_transformation_matrix(R.T, -np.matmul(R.T, p.T).T)

def get_R_p_from_matrix(T):
    #R = T[0:-1, 0:-1]
    #p = np.array([T[0:-1, -1]])
    return T[0:-1, 0:-1], np.array([T[0:-1, -1]])

def get_rotation_matrix_from_exponential_representation(w, theta):
    return np.identity(w.shape[1]) + np.sin(theta) * get_skew_symmetric_matrix(w) + (1 - np.cos(theta)) * np.matmul(get_skew_symmetric_matrix(w), get_skew_symmetric_matrix(w))

def get_exponential_representation_from_rotation_matrix(R, default=None):
    """
    current only work with 3 * 3 matrix
    """
    theta = 0
    w = np.zeros((1, R.shape[0]))

    if is_identity_matrix(R):
        pass
    elif close_to(np.trace(R), -1, error=1e-3):
        #print "[kinematics][get_exponential_representation_from_rotation_matrix] close to trace"
        theta = np.pi
        option = random.randint(0, 2)
        sign = (-1) ** random.randint(0, 1)
        if default is None:
            w = sign * 1 / np.sqrt(2 * (1 + R[option, option])) * np.array([[R[0, option], R[1, option], 1 + R[2, option]]])
        else:
            w = np.array([default])
    else:
        theta = np.arccos((np.trace(R) - 1) / 2)
        w = get_vector_from_skew_symmetric_matrix(0.5 / np.sin(theta) * (R - R.T))

    return w, theta

def get_twist(w, v):
    return np.c_[w, v]

def get_w_v_from_twist(V):
    return np.array(V[:, :V.shape[1]/2]), np.array(V[:, V.shape[1]/2:])

def _get_twist_matrix(w, v):
    return np.r_[np.c_[get_skew_symmetric_matrix(w), v.T], np.zeros((1, w.shape[1] + 1))]

def get_twist_matrix(V):
    w, v = get_w_v_from_twist(V)
    return _get_twist_matrix(w, v)

def get_radian_from_degree(theta):
    return theta / 180.0 * np.pi

def _get_adjoint_representation_matrix(R, p):
    return np.r_[np.c_[R, np.zeros(R.shape)], np.c_[np.matmul(get_skew_symmetric_matrix(p), R), R]]

def get_adjoint_representation_matrix(T):
    return _get_adjoint_representation_matrix(np.array(T[:-1, :-1]), np.array([T[:-1, -1]]))

def get_v(w, r):
    return np.cross(w, -r)

def _get_screw_axis(w, v):
    if is_zero_vector(w):
        return get_twist(w, v) / np.linalg.norm(v)
    else:
        return get_twist(w, v) / np.linalg.norm(w)

def get_screw_axis(V):
    w, v = get_w_v_from_twist(V)
    return _get_screw_axis(w, v)

def get_homogenous_transformation_matrix_from_exponential_representation(S, theta):
    w = np.array(S[:, :S.shape[1]/2])
    v = np.array(S[:, S.shape[1]/2:])
    
    #print "w: ", w
    #print "v: ", v

    R = np.identity(w.shape[1])
    p = np.zeros(w.shape)

    if is_unit_vector(w, error=1e-4):
        #print "w is unit vector"
        R = get_rotation_matrix_from_exponential_representation(w, theta)
        #print "R"
        #print R
        p = np.matmul(np.identity(w.shape[1]) * theta + (1 - np.cos(theta)) * get_skew_symmetric_matrix(w) + (theta - np.sin(theta)) * np.matmul(get_skew_symmetric_matrix(w), get_skew_symmetric_matrix(w)), v.T).T
    elif is_zero_vector(w, error=1e-4) and is_unit_vector(v, error=1e-4):
        #print "w is zero and v is unit vector"
        p = v * theta
    #else:
        #print "something is not right"

    return get_homogenous_transformation_matrix(R, p)

def get_exponential_representation_from_homogenous_transformation_matrix(T, threshold=1e-6, default=None):
    R, p = get_R_p_from_matrix(T)

    w = np.zeros((1, R.shape[0]))
    v = np.zeros((1, R.shape[0]))
    theta = 0.

    if is_identity_matrix(R, threshold):
        #print "[kinematics][get_exponential_representation_from_homogenous_transformation_matrix] is_identity_matrix"
        if is_zero_vector(p):
            v = np.zeros((1, R.shape[0]))
        else:
            v = p / np.linalg.norm(p)
        theta = np.linalg.norm(p)
    else:
        #print "[kinematics][get_exponential_representation_from_homogenous_transformation_matrix] not identity_matrix"
        w, theta = get_exponential_representation_from_rotation_matrix(R, default=default)
        v = np.matmul(np.identity(R.shape[0]) / theta - get_skew_symmetric_matrix(w) / 2 + (1 / theta - 0.5 / np.tan(theta / 2)) * np.matmul(get_skew_symmetric_matrix(w), get_skew_symmetric_matrix(w)), p.T).T
    
    return get_screw_axis(get_twist(w, v)), theta

def forward_kinematics_s_frame(S_list, theta_list, M):
    result_matrix = np.identity(M.shape[0])
    for i in range(S_list.shape[0]):
        result_matrix = np.matmul(result_matrix, get_homogenous_transformation_matrix_from_exponential_representation(np.array([S_list[i, :]]), theta_list[:, i]))
    result_matrix = np.matmul(result_matrix, M)
    return result_matrix

def forward_kinematics_s_frame_all_joints(S_list, theta_list, M_list):
    all_joints_result_matrix = []
    result_matrix = np.identity(M_list[-1].shape[0])
    for i in range(S_list.shape[0]):
        result_matrix = np.matmul(result_matrix, get_homogenous_transformation_matrix_from_exponential_representation(np.array([S_list[i, :]]), theta_list[:, i]))
        all_joints_result_matrix.append(np.matmul(result_matrix, M_list[i]))
    #result_matrix = np.matmul(result_matrix, M)
    return all_joints_result_matrix

def forward_kinematics_s_frame_position(S_list, theta_list, M):
    result_matrix = forward_kinematics_s_frame(S_list, theta_list, M)
    return np.array([result_matrix[:-1, -1]])

def forward_kinematics_b_frame(B_list, theta_list, M):
    result_matrix = M
    for i in range(B_list.shape[0]):
        result_matrix = np.matmul(result_matrix, get_homogenous_transformation_matrix_from_exponential_representation(np.array([B_list[i, :]]), theta_list[:, i]))
    return result_matrix

def space_jacobian(S_list, theta_list):
    S_list_comp = np.r_[[[1, 0, 0, 0, 0, 0]], S_list]
    theta_list_comp = np.c_[[[0]], theta_list]

    result_matrix = np.zeros(S_list.T.shape)
    T = np.identity(S_list.shape[1] / 2 + 1)

    for i in range(1, S_list_comp.shape[0]):
        T = np.matmul(T, get_homogenous_transformation_matrix_from_exponential_representation(np.array([S_list_comp[i - 1, :]]), theta_list_comp[:, i - 1]))
        result_matrix[:, i - 1] = np.matmul(get_adjoint_representation_matrix(T), np.array([S_list_comp[i, :]]).T)[:, 0]

    return result_matrix

def body_jacobian(B_list, theta_list):
    B_list_comp = np.r_[B_list, [[1, 0, 0, 0, 0, 0]]]
    theta_list_comp = np.c_[theta_list, [[0]]]

    result_matrix = np.zeros(B_list.T.shape)
    T = np.identity(B_list.shape[1] / 2 + 1)

    for i in range(B_list.shape[0], 0, -1):
        T = np.matmul(T, get_homogenous_transformation_matrix_from_exponential_representation(-np.array([B_list_comp[i, :]]), theta_list_comp[:, i]))
        result_matrix[:, i - 1] = np.matmul(get_adjoint_representation_matrix(T), np.array([B_list_comp[i - 1, :]]).T)[:, 0]

    return result_matrix

def _matrix_ellipspod_singularity_analysis(A):
    eigenvalues = np.linalg.eigvals(A).tolist()
    criteria_2 = max(eigenvalues) * 1.0 / min(eigenvalues)
    criteria_1 = np.sqrt(criteria_1)
    criteria_3 = np.linalg.det(A)
    return criteria_1, criteria_2, criteria_3

def manipulability_linear_analysis(body_jacobian):
    return _matrix_ellipspod_singularity_analysis(np.matmul(body_jacobian[S.shape[1]/2:, :], body_jacobian[S.shape[1]/2:, :].T))

def manipulability_angular_analysis(body_jacobian):
    return _matrix_ellipspod_singularity_analysis(np.matmul(body_jacobian[0:S.shape[1]/2, :], body_jacobian[0:S.shape[1]/2, :].T))

def ik_newton_raphson_b_frame(B_list, theta_list, M, target, num_iteration=20):
    to_stop = False
    i = 0

    while not to_stop and i < num_iteration:
        Tbs = get_homogenous_transformation_matrix_inverse(forward_kinematics_b_frame(B_list, theta_list, M))
        Sb, theta = get_exponential_representation_from_homogenous_transformation_matrix(np.matmul(Tbs, target))
        Vb = Sb * theta
        w, v = get_w_v_from_twist(Vb)
        if not is_zero_vector(w, 0.001) or not is_zero_vector(v, 0.001):
            theta_list += np.matmul(np.linalg.pinv(body_jacobian(B_list, theta_list)), Vb.T).T
        else:
            to_stop = True
        i += 1
    return theta_list, to_stop

def ik_newton_raphson_s_frame(S_list, theta_list, M, target, num_iteration=20):
    to_stop = False
    i = 0

    while not to_stop and i < num_iteration:
        Tsb = forward_kinematics_s_frame(S_list, theta_list, M)
        Tbs = get_homogenous_transformation_matrix_inverse(Tsb)
        Sb, theta = get_exponential_representation_from_homogenous_transformation_matrix(np.matmul(Tbs, target))
        Vb = Sb * theta
        Vs = np.matmul(get_adjoint_representation_matrix(Tsb), Vb.T).T
        w, v = get_w_v_from_twist(Vs)
        if not is_zero_vector(w, 0.001) or not is_zero_vector(v, 0.001):
            theta_list += np.matmul(np.linalg.pinv(space_jacobian(S_list, theta_list)), Vs.T).T
        else:
            to_stop = True
        i += 1
    return theta_list, to_stop

def ik_newton_raphson_s_frame_get_direction(S_list, theta_list, M, target):
    Tsb = forward_kinematics_s_frame(S_list, theta_list, M)
    Tbs = get_homogenous_transformation_matrix_inverse(Tsb)
    Sb, theta = get_exponential_representation_from_homogenous_transformation_matrix(np.matmul(Tbs, target))
    Vb = Sb * theta
    Vs = np.matmul(get_adjoint_representation_matrix(Tsb), Vb.T).T
    return np.matmul(np.linalg.pinv(space_jacobian(S_list, theta_list)), Vs.T).T

def ik_newton_raphton_s_frame_follow_trajectory(S_list, theta_list, M, target, step=0.001, num_iteration=20):
    results = []
    temp_angles = np.array(theta_list)

    current_position = forward_kinematics_s_frame_position(S_list, theta_list, M)
    target_position = np.array([target[:-1, -1]])
    direction = target_position - current_position
    distance = np.linalg.norm(direction)

    num_sub_targets = int(round(distance / step))

    subtargets = [current_position + direction / num_sub_targets * (i + 1) for i in range(num_sub_targets)]

    for i in range(num_sub_targets):
        subtarget = np.array(target)
        subtarget[:-1, -1] = subtargets[i]
        temp_angles, temp_is_success = ik_newton_raphson_s_frame(S_list, temp_angles, M, subtarget, num_iteration)
        results.append((np.array(temp_angles), temp_is_success))
    return results

def test_screw_axis(Tworld_tool1start, Tworld_tool1end, Tworld_goalstart, Tworld_tool2start, Tworld_tool2end):
    #print "Tworld_tool1start"
    #print Tworld_tool1start
    #print "Tworld_tool1end"
    #print Tworld_tool1end   
    Ttool1start_tool1end = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_tool1start), Tworld_tool1end)
    #print "Ttool1start_tool1end"
    #print Ttool1start_tool1end
    
    Ttool1start_goalstart = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_tool1start), Tworld_goalstart)
    
    #print "Ttool1start_goalstart"
    #print Ttool1start_goalstart
    
    Tgoalstart_goalend = np.matmul(np.matmul(get_homogenous_transformation_matrix_inverse(Ttool1start_goalstart), Ttool1start_tool1end), Ttool1start_goalstart)
    
    #print "Tgoalstart_goalend"
    #print Tgoalstart_goalend
    
    Tworld_goalend = np.matmul(Tworld_goalstart, Tgoalstart_goalend)
    #print "Tworld_goalend"
    #print Tworld_goalend

    #print "-------------------------------------------------"
    
    #print "Tworld_tool2start"
    #print Tworld_tool2start
    #print "Tworld_tool2end"
    #print Tworld_tool2end   
    Ttool2start_tool2end = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_tool2start), Tworld_tool2end)
    #print "Ttool2start_tool2end"
    #print Ttool2start_tool2end
    
    #Tworld_goalstart = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])
    
    Ttool2start_goalstart = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_tool2start), Tworld_goalstart)
    
    #print "Ttool2start_goalstart"
    #print Ttool2start_goalstart
    
    Tgoalstart_goalend = np.matmul(np.matmul(get_homogenous_transformation_matrix_inverse(Ttool2start_goalstart), Ttool2start_tool2end), Ttool2start_goalstart)
    
    #print "Tgoalstart_goalend"
    #print Tgoalstart_goalend
    
    Tworld_goalend = np.matmul(Tworld_goalstart, Tgoalstart_goalend)
    #print "Tworld_goalend"
    #print Tworld_goalend
    
    #print "-------------------------------------"
    
    T2_1 = np.matmul(Ttool2start_goalstart, get_homogenous_transformation_matrix_inverse(Ttool1start_goalstart))
    
    D = Ttool1start_tool1end
    #D2 = Ttool2start_tool2end
    
    #print "T2_1"
    #print T2_1
    
    #print "D"
    #print D
    
    #print "T2_1 * D"
    #print np.matmul(T2_1, D)
    
    #print "D * T2_1"
    #print np.matmul(D, T2_1)
    
    A = Tgoalstart_goalend
    #print "A"
    #print A
    
    # http://www.cs.cornell.edu/projects/caliber/manual.pdf
    
    
    S, theta = get_exponential_representation_from_homogenous_transformation_matrix(A)
    #print "A"
    #print S
    #print theta
    
    S, theta = get_exponential_representation_from_homogenous_transformation_matrix(D)
    #print "D"
    #print S
    #print theta
    
    S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T2_1)
    #print "T2_1"
    #print S
    #print theta
    
    #delta_T = get_homogenous_transformation_matrix_from_exponential_representation(S, 0.707)
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(np.matmul(delta_T, delta_T))
    #print "delta_T * delta_T"
    #print S
    #print theta    
    
    #Tn_1 = get_homogenous_transformation_matrix_from_exponential_representation(S, 3.14)
    #Tn = np.matmul(Tn_1, Ttool1start_goalstart)
    #print "Ttoolstartn_goalstart"
    #print Tn
    
    w, v = get_w_v_from_twist(S)
    w_matrix = get_skew_symmetric_matrix(w)
    Rd, pd = get_R_p_from_matrix(D)
    
    #print "pd: ", pd
    
    #print "[w] * Rd"
    #print np.matmul(w_matrix, Rd)
    
    #print "Rd * [w]"
    #print np.matmul(Rd, w_matrix)
    
    #print "v = ", v
    
    #print "[w] * [x, y, z]T"
    #print np.matmul(w_matrix, pd.T)
    
def get_test_parameters(tool_type):
    Tworld_tool1start = None
    Tworld_tool1end = None
    Tworld_goalstart = None
    Tworld_tool2start = None
    Tworld_tool2end = None
    
    if tool_type == "hammer_around":
        Tworld_tool1start = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])
        Tworld_tool1end = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_goalstart = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])
        Tworld_tool2start = np.array([[0, 1, 0, 0], [-1, 0, 0, 2], [0, 0, 1, 2], [0, 0, 0, 1]])
        Tworld_tool2end = np.array([[0, 1, 0, 0], [-1, 0, 0, 2], [0, 0, 1, 0], [0, 0, 0, 1]]) 
    
    if tool_type == "hammer":
        Tworld_tool1start = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])
        Tworld_tool1end = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_goalstart = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])
        Tworld_tool2start = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])
        Tworld_tool2end = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    if tool_type == "push_origin":
        Tworld_tool1start = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_tool1end = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_goalstart = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_tool2start = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_tool2end = np.array([[0, -1, 0, 2], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    if tool_type == "push_consistent":
        Tworld_tool1start = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_tool1end = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_goalstart = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        Tworld_tool2start = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        Tworld_tool2end = np.array([[1, 0, 0, 2], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])     
    
    return Tworld_tool1start, Tworld_tool1end, Tworld_goalstart, Tworld_tool2start, Tworld_tool2end

if __name__ == "__main__":
    #print get_skew_symmetric_matrix(np.array([1, 2, 3], ndmin=2))

    #R = get_skew_symmetric_matrix(np.array([[1, 2, 3]]))
    #w = get_vector_from_skew_symmetric_matrix(R)
    #print w

    #R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    #p = np.array([1, 2, 3], ndmin=2)
    #print  np.linalg.inv(get_homogenous_transformation_matrix(R, p))
    #print np.linalg.inv(R)
    #print np.matmul(-R.T, p.T)

    #T = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 3], [0, 0, 0, 1]])
    #print get_homogenous_transformation_matrix_inverse(T)

    #print get_radian_from_degree(30)
    #print get_rotation_matrix_from_exponential_representation(np.array([[0, 0.866, 0.5]]), get_radian_from_degree(30))

    #T = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 1]])
    #R, p = get_R_p_from_matrix(T)
    #print R
    #print p

    #print get_twist(np.array([1, 2, 3], ndmin=2), np.array([4, 5, 6], ndmin=2))

    #V = np.array([[1, 2, 3, 4, 5, 6]])
    #w, v = get_w_v_from_twist(V)
    #print w, v

    #print _get_twist_matrix(np.array([1, 2, 3], ndmin=2), np.array([4, 5, 6], ndmin=2))

    #print get_twist_matrix(np.array([[1, 2, 3, 4, 5, 6]]))

    #R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    #p = np.array([[1, 2, 3]])
    #T = get_homogenous_transformation_matrix(R, p)
    #print _get_adjoint_representation_matrix(R, p)
    #T = get_homogenous_transformation_matrix_from_exponential_representation(np.array([[1, 0, 0, 0, 0, 0]]), 0)
    #print get_adjoint_representation_matrix(T)

    #print get_v(np.array([[0, 0, 2]]), np.array([[2, -1, 0]]))

    #w = np.array([[0, 0, 0]])
    #v = np.array([[4, 5, 6]])
    #print get_screw_axis(w, v)

    #w = np.array([[0, 1, 0]])
    #v = np.array([[1, 0, 0]])
    #T = get_homogenous_transformation_matrix_from_exponential_representation(get_screw_axis(get_twist(w, v)), 1)
    #print T
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T)
    #print S
    #print theta

    #M = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]])
    #Blist = np.array([[0, 0, -1, 2, 0,   0],
                        #[0, 0,  0, 0, 1,   0],
                        #[0, 0,  1, 0, 0, 0.1]])
    #thetalist = np.array([[np.pi / 2.0, 3, np.pi]])
    #print forward_kinematics_b_frame(Blist, thetalist, M)

    """
    Method 1 (space frame): add a rotation matrix in the end, Tab * Tbc
    or: at initial position: Tbc = Tab-1 * Tac
    """
    #theta = get_radian_from_degree(45)
    #Tbc = np.array([[np.cos(theta), -np.sin(theta), 0, np.cos(theta)], [np.sin(theta), np.cos(theta), 0, np.sin(theta)], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print Tbc
    #M = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #S_list = np.array([[0, 0, 1,  0,  0,  0],
                        #[0, 0, 1, 0, -1, 0]])
    #theta_list = np.array([[0, 0]])
    #Tab = forward_kinematics_s_frame(S_list, theta_list, M)
    #print Tab
    #print np.matmul(Tab, Tbc)

    """
    Method 2 (space frame): treat the last one as a joint but with fixed theta
    """
    #theta = get_radian_from_degree(45)
    #l1 = 1
    #l2 = 1
    #tool = 1
    #M = np.array([[1, 0, 0, l1 + l2 + tool], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #S_list = np.array([[0, 0, 1,  0,  0,  0],
                        #[0, 0, 1, 0, -l1, 0],
                        #[0, 0, 1, 0, -l1 - l2, 0]]) # the last one is the tool
    #theta_list = np.array([[0, np.pi/4, theta]])
    #print forward_kinematics_s_frame(S_list, theta_list, M)

    #Slist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                      #[1, 0, 0,   2,   0,   3],
                      #[0, 1, 0,   0,   2,   1],
                      #[1, 0, 0, 0.2, 0.3, 0.4]])
    #thetalist = np.array([[0.2, 1.1, 0.1, 1.2]])
    #print space_jacobian(Slist, thetalist)

    #Blist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                      #[1, 0, 0,   2,   0,   3],
                      #[0, 1, 0,   0,   2,   1],
                      #[1, 0, 0, 0.2, 0.3, 0.4]])
    #thetalist = np.array([[0.2, 1.1, 0.1, 1.2]])
    #print body_jacobian(Blist, thetalist)

    #B_list = np.array([[0, 0, -1, 2, 0,   0],
                      #[0, 0,  0, 0, 1,   0],
                      #[0, 0,  1, 0, 0, 0.1]])
    #M = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]])
    #target = np.array([[0, 1,  0,     -5],
                  #[1, 0,  0,      4],
                  #[0, 0, -1, 1.6858],
                  #[0, 0,  0,      1]])
    #theta_list = np.array([[1.5, 2.5, 3]])
    #result, correctness = ik_newton_raphson_b_frame(B_list, theta_list, M, target)
    #print result
    #print correctness

    # S_list = np.array([[0, 0,  1,  4, 0,    0],
    #                   [0, 0,  0,  0, 1,    0],
    #                   [0, 0, -1, -6, 0, -0.1]])
    # M = np.array([[-1, 0, 0, 0], [0, 1, 0, 6], [0, 0, -1, 2], [0, 0, 0, 1]])
    # target = np.array([[0, 1,  0,     -5],
    #               [1, 0,  0,      4],
    #               [0, 0, -1, 1.6858],
    #               [0, 0,  0,      1]])
    # theta_list = np.array([[1.5, 2.5, 3]])
    # result, correctness = ik_newton_raphson_s_frame(S_list, theta_list, M, target)
    # print result
    # print correctness

    # case 1: rotate together
    #Tworld_1 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_1"
    #print Tworld_1
    #Tworld_2 = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1]])
    #print "Tworld_2"
    #print Tworld_2   
    #T1_2 = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_1), Tworld_2)
    #print "T1_2"
    #print T1_2
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T1_2)
    #print S
    #print theta
    
    #print "================================================="
    
    #Tworld_3 = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_3"
    #print Tworld_3
    #Tworld_4 = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.5], [0, 0, 0, 1]])
    #print "Tworld_4"
    #print Tworld_4
    #T3_4 = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_3), Tworld_4)
    #print "T3_4"
    #print T3_4    
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T3_4)
    #print S
    #print theta
    
    
    
    # case 2: translate together
    #Tworld_1 = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_1"
    #print Tworld_1
    #Tworld_2 = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])
    #print "Tworld_2"
    #print Tworld_2   
    #T1_2 = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_1), Tworld_2)
    #print "T1_2"
    #print T1_2
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T1_2)
    #print S
    #print theta
    
    #print "================================================="
    
    #Tworld_3 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_3"
    #print Tworld_3
    #Tworld_4 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    #print "Tworld_4"
    #print Tworld_4
    #T3_4 = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_3), Tworld_4)
    #print "T3_4"
    #print T3_4    
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T3_4)
    #print S
    #print theta
    
    # case 3: rotate the same, one translate faster
    #Tworld_1 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_1"
    #print Tworld_1
    #Tworld_2 = np.array([[0, 0, -1, 0], [0, 1, 0, 2], [1, 0, 0, 1], [0, 0, 0, 1]])
    #print "Tworld_2"
    #print Tworld_2   
    #T1_2 = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_1), Tworld_2)
    #print "T1_2"
    #print T1_2
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T1_2)
    #print S
    #print theta
    
    #print "================================================="
    
    #Tworld_3 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_3"
    #print Tworld_3
    #Tworld_4 = np.array([[0, 0, -1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    #print "Tworld_4"
    #print Tworld_4
    #T3_4 = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_3), Tworld_4)
    #print "T3_4"
    #print T3_4    
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T3_4)
    #print S
    #print theta
    
    # case 4: axis not at origin
    #Tworld_1 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_1"
    #print Tworld_1
    #Tworld_2 = np.array([[0, 0, -1, -1], [0, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 1]])
    #print "Tworld_2"
    #print Tworld_2   
    #T1_2 = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_1), Tworld_2)
    #print "T1_2"
    #print T1_2
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T1_2)
    #print S
    #print theta
    
    #print "================================================="
    
    #Tworld_3 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_3"
    #print Tworld_3
    #Tworld_4 = np.array([[0, 0, -1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]])
    #print "Tworld_4"
    #print Tworld_4
    #T3_4 = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_3), Tworld_4)
    #print "T3_4"
    #print T3_4    
    #S, theta = get_exponential_representation_from_homogenous_transformation_matrix(T3_4)
    #print S
    #print theta
    
    #Tworld_toolstart = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #print "Tworld_toolstart"
    #print Tworld_toolstart
    #Tworld_toolend = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 2], [0, 0, 0, 1]])
    #print "Tworld_toolend"
    #print Tworld_toolend   
    #Ttoolstart_toolend = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_toolstart), Tworld_toolend)
    #print "Ttoolstart_toolend"
    #print Ttoolstart_toolend
    
    #Tworld_goalstart = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    #Ttoolstart_goalstart = np.matmul(get_homogenous_transformation_matrix_inverse(Tworld_toolstart), Tworld_goalstart)
    
    #Tgoalstart_goalend = np.matmul(np.matmul(get_homogenous_transformation_matrix_inverse(Ttoolstart_goalstart), Ttoolstart_toolend), Ttoolstart_goalstart)
    
    #print "Tgoalstart_goalend"
    #print Tgoalstart_goalend
    
    #Tworld_goalend = np.matmul(Tworld_goalstart, Tgoalstart_goalend)
    #print "Tworld_goalend"
    #print Tworld_goalend
    
    Tworld_tool1start, Tworld_tool1end, Tworld_goalstart, Tworld_tool2start, Tworld_tool2end = get_test_parameters("push_consistent")
    test_screw_axis(Tworld_tool1start, Tworld_tool1end, Tworld_goalstart, Tworld_tool2start, Tworld_tool2end)
    