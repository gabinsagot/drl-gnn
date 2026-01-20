import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial import distance_matrix
import shutil
from pathlib import Path
import subprocess

def get_closest_point(points, mesh):
    """
    Returns np.ndarray of shape (M,2) containing the closest point in 'mesh' for each point in 'points'.
    Args: 
        point: np.ndarray of shape (M,2)
        mesh: np.ndarray of shape (N, 2)
    """
    closest_points = []

    for point in points :
        distances = np.linalg.norm(mesh - point, axis=1)
        closest_index = np.argmin(distances)
        closest_point = mesh[closest_index]
        closest_points.append(closest_point)

    # print("Points les + proches : ", closest_points)
    _, counts = np.unique(closest_points, return_counts=True)
    if np.any(counts > 1) :
        print("Warning: Some control points map to the same mesh point.")

    closest_points = np.array(closest_points)
    return closest_points



def extract_points(t_file : str):
    """
    Extract the points from a .t file

    Args:
        t_file: The .t file to extract points from.

    Returns:
        A np.ndarray of shape (N, 2) of points extracted from the .t file.
    """
    points = []
    with open(t_file, 'r') as f:
        for line in f:
            l = line.split()
            points.append(np.array([float(l[0]), float(l[1])],dtype=np.float64)) if len(l) == 2 else None
        points = np.array(points, dtype=np.float64)
    return points


def idw(mesh, control_points, init_displacements, p, take_edges=True):
    """
    Args :
        mesh : np.ndarray of shape (N, 2)
        control_points : np.ndarray of shape (M, 2)
        init_displacements : np.ndarray of shape (M, 2)
        p : power parameter for inverse-distance weighting

    Returns :
        new_mesh : np.ndarray of shape (N, 2)
    """
    start_time = time.perf_counter()

    H = max(mesh, key=lambda x: x[1])[1]  # hauteur
    l = max(mesh, key=lambda x: x[0])[0]  # largeur (max x)
    null = np.array([0.0, 0.0])

    
    def is_edge(point):
        x, y = point[0], point[1]
        # use isclose to avoid floating-point equality issues
        return np.isclose(x, 0.0) or np.isclose(y, 0.0) or np.isclose(x, l) or np.isclose(y, H)
    
    if take_edges:
        for point in mesh:
            if is_edge(point):
                control_points = np.vstack((control_points, point))
                init_displacements = np.vstack((init_displacements, null))

    distances = distance_matrix(mesh, control_points, threshold=1e8)
    #if there is a 0 in a line, it means the point is a control_point
    #In this case, we can use the control point's displacement directly
    # compute inverse-distance weights, handling zeros so that a row with a control point
    # becomes one-hot (1 for coincident control(s), 0 for others)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / (distances ** p)

    zero_mask = (distances == 0)
    if zero_mask.any():
        # clear any inf/NaN produced by division by zero
        weights[zero_mask] = 0.0
        # for rows that contain one or more exact matches, set the row to the mask (1.0 where match)
        rows_with_zero = zero_mask.any(axis=1)
        weights[rows_with_zero] = zero_mask[rows_with_zero].astype(np.float64)

    #make weights sum to 1
    weights /= weights.sum(axis=1, keepdims=True)

    displacements = weights @ init_displacements
    new_mesh = mesh + displacements
    #displacements is of shape (N, 2). We need to create a list of np.ndarrays of shape (2,)
    #print(displacements)

    end_time = time.perf_counter()
    print(f"IDW computation time: {end_time - start_time:.4f} seconds")

    return new_mesh


    """
    for point in points:
        #ON CHOISIT UN X
        weighted_disp = np.array([0.0, 0.0])
        weights = []
        exact_match = False

        for control_point, s in transformation:
            #ON ITERE SUR I
            d = np.linalg.norm(point - control_point, ord=2)
            # exact match -> use the control displacement directly
            if d == 0:
                new_points.append(point + s)
                exact_match = True
                break
            else:
                w = 1.0 / (d ** p)
                weighted_disp = weighted_disp + w * s
                weights.append(w)

        if exact_match:
            continue

        S = sum(weights)
        if S == 0:
            new_points.append(point.copy())
        else:
            disp = weighted_disp / S
            new_points.append(point + disp)
    """



    return new_points
            

def replace_points(input_t_file_path : str , output_t_file_path : str, new_points):
    """
    Changes the points of a .t file and replaces them with new_points
    Args:
        input_t_file: Path to the .t file to take initial points
        output_t_file: Path to the .t file where to modify
        new_points: np.ndarray of shape (N, 2) containing the new points.
    Returns: 
        str: path to the new t_file
    """
    #new_file = input_t_file_path.replace('.t', '_idw.t')
    new_file = output_t_file_path

    with open(input_t_file_path, 'r') as f_in, open(new_file, 'w') as f_out:

        lines = f_in.readlines()
        point_index = 0

        for i in range(len(lines)):

            l = lines[i].split()

            if len(l) == 2:
                
                np_point = new_points[point_index]
                lines[i] = f"{np_point[0]} {np_point[1]}\n"
                point_index += 1
            
        f_out.writelines(lines)

    return new_file


#TESTS
TEST1 = 0
if TEST1:
    L = []
    for i in range(10):
        for j in range(10):
            L.append(np.array([float(i), float(j)]))

    plt.scatter(*zip(*L), s=10, color='gray', alpha=0.5) #transparent
    M = []
    M.extend((np.array([i,3]), np.array([0, -1])) for i in range(3, 8))
    M.extend((np.array([3,j]), np.array([-1, 0])) for j in range(3, 8))
    M.extend((np.array([7,j]), np.array([1, 0])) for j in range(3, 8))
    M.extend([
        (np.array([4,7]), np.array([0, 1])),
        (np.array([5,7]), np.array([0, 1])),
        (np.array([6,7]), np.array([0, 1])),
    ])

    shift = np.array([0, 1])
    null = np.array([0.0, 0.0])
    transformation = M

    # avoid exact matches with control points (prevents duplicate appends in the current idw)
    def is_control(pt):
        return any(np.allclose(pt, cp) for cp, _ in transformation)

    L = [pt for pt in L if not is_control(pt)]

    # compute transformed points using IDW (power parameter p=2 chosen as example)
    new_points = idw(L, transformation, p=4)

    # plot control points (original) in red
    ctrl_orig = [cp for cp, _ in transformation]
    if ctrl_orig:
        plt.scatter(*zip(*ctrl_orig), color='red', label='controls')

    # plot transformed grid
    if new_points:
        plt.scatter(*zip(*new_points), s=10, color='blue', label='transformed')
    ctrl_new = [cp + disp for cp, disp in transformation]
    if ctrl_new:
        plt.scatter(*zip(*ctrl_new), color='orange', label='ctrl after')

    plt.legend()
    plt.axis('equal')
    plt.show()
    
TEST2 = 0
if TEST2:
    i_1 = 16681
    i_2 = 200
    L = extract_points('idw/domain.t')
    np.set_printoptions(precision=16, suppress=False)
    print(f"In domain.t, line {i_1} = {L[i_1]}, line {i_2} = {L[i_2]}")

    control_points = [L[i_1], L[i_2]]
    init_displacements = [np.array([0.0, 0.5]), np.array([0.0, 0])]

    
    new_points = idw(L, control_points, init_displacements, p=1)


    print(".\n.\n.\nTransformation done.")
    #print(new_points, len(new_points))
    print(f"{len(new_points)} points. In new_points, line {i_1} = {new_points[i_1]}, line {i_2} = {new_points[i_2]}")
    file = replace_points('idw/domain.t', new_points)
    print(f".\n.\n.\nFile written: {file}")
    check = extract_points('idw/domain_idw.t')
    print(f"In {file}, line {i_1} = {check[i_1]}, line {i_2} = {check[i_2]}")

TEST3 = 0
if TEST3:
    file = 'limace'
    L = extract_points('../idw/limace.t')
    print("Points extracted.")
    i_1 = 9529
    i_2 = 100
    print(f"In {file}.t, line {i_1} = {L[i_1]}, line {i_2} = {L[i_2]}")

    null = np.array([0.0, 0.0])
    shift = np.array([0.0, 0.06])
    control_points = []
    displ = []

    control_points.append(L[9529])

    print(f"Control points : {len(control_points)}")

    displ.append(shift)

    print(f"Displacements : {len(displ)}")

    control_points = np.array(control_points)
    displ = np.array(displ)
    new_points = idw(L, control_points, displ, p=3.5)

    # show full precision in console

    print(".\n.\n.\nTransformation done.")
    #print(new_points, len(new_points))
    print(f"{len(new_points)} points.") 

    print(f"In new_points, line {i_1} = {new_points[i_1]}, line {i_2} = {new_points[i_2]}")
    file = replace_points('../idw/limace.t', '../idw/limace_idw.t', new_points)
    
    print(f".\n.\n.\nFile written: {file}")
    check = extract_points('../idw/limace_idw.t')   
    print(f"In {file}, line {i_1} = {check[i_1]}, line {i_2} = {check[i_2]}")

    #move idw/limace.t into cfd_ex, and rename it test.t
    src = Path(file)  # 'file' was set earlier to 'idw/limace_idw.t'
    dest_dir = Path('../idw/cfd_ex/meshes')
    
    dest = dest_dir / 'test.t'

    try:
        shutil.move(str(src), str(dest))
        print(f"Moved '{src}' -> '{dest}'")
    except Exception as e:
        print(f"Failed to move '{src}' -> '{dest}': {e}")


TEST4=0
if TEST4:
    """
    Takes a meshed airfoil with control points, moves a point to a corresponding location 
    """
