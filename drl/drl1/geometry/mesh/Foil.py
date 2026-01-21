import os
from shutil import copy
import shutil
import sys
import gmsh
import subprocess
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# helper: orientation test
def _orient(a, b, c):
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

# helper: check if point c lies on segment ab
def _on_segment(a, b, c):
    return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

# check if two segments (a1-a2) and (b1-b2) intersect
def _segments_intersect(a1, a2, b1, b2):
    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    if o1 == 0 and _on_segment(a1, a2, b1):
        return True
    if o2 == 0 and _on_segment(a1, a2, b2):
        return True
    if o3 == 0 and _on_segment(b1, b2, a1):
        return True
    if o4 == 0 and _on_segment(b1, b2, a2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

# check if polygon defined by pts has any self-intersections
def _is_self_intersecting(pts):
    n = len(pts)
    if n < 4:
        return False
    edges = [(i, (i + 1) % n) for i in range(n)]
    for i, (a1_i, a2_i) in enumerate(edges):
        for j, (b1_i, b2_i) in enumerate(edges):
            # skip same edge or adjacent edges (they share a node)
            if abs(i - j) <= 1 or (i == 0 and j == n - 1) or (j == 0 and i == n - 1):
                continue
            if _segments_intersect(pts[a1_i], pts[a2_i], pts[b1_i], pts[b2_i]):
                return True
    return False




class Foil:

    def __init__(self, number_of_points, chord_length_multiplier, thickness_multiplier,
                 work_dir: str = "", suffix: str = ""):
        """
        work_dir: base directory for this instance (e.g., geometry/mesh/{ep})
                  Files will be written under work_dir/{txt,geo,msh,t}.
                  If None, falls back to original script_dir/{txt_files,geo_files,msh_files,t_files}.
        suffix:   string appended to file basenames (e.g., '_17') to ensure uniqueness.
        """
        self.msh_size = 0.01
        self.type = "spline"
        self.number_of_points = number_of_points
        self.chord_length = chord_length_multiplier
        self.thickness_multiplier = thickness_multiplier
        self.points = np.array(self.generate_airfoil_points())
        self.origin = self.points[:, :1].argmin()  # Furthest left point
        self.surface = self.compute_surface()
        self.name = "object"  # base name without suffix
        self.suffix = suffix
        # ---- new: episode-local work area
        self.work_dir = work_dir  # e.g., geometry/mesh/{ep}
        self._init_dirs()

    # ---- new: directory resolver
    def _init_dirs(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if self.work_dir is None:
            # backward compatible layout
            self.txt_dir = os.path.join(script_dir, "txt_files")
            self.geo_dir = os.path.join(script_dir, "geo_files")
            self.msh_dir = os.path.join(script_dir, "msh_files")
            self.t_dir   = os.path.join(script_dir, "t_files")
            self.tmp_msh_dir = os.path.join(script_dir, "tmp_msh_files")
        else:
            # episode-local, parallel-safe layout
            self.txt_dir = os.path.join(self.work_dir, "txt")
            self.geo_dir = os.path.join(self.work_dir, "geo")
            self.msh_dir = os.path.join(self.work_dir, "msh")
            self.t_dir   = os.path.join(self.work_dir, "t")
            self.tmp_msh_dir = os.path.join(self.work_dir, "tmp_msh")
        for d in (self.txt_dir, self.geo_dir, self.msh_dir, self.t_dir, self.tmp_msh_dir):
            os.makedirs(d, exist_ok=True)

    def _base(self):
        """Unified basename including optional suffix."""
        return f"{self.name}{self.suffix}"


    def generate_airfoil_points(self, random : bool = False):
        """
        Generates a list of number_of_points points randomly if random == true or a NACA 0010 profile if false.
        """
        points = []
        if random :
            for i in range(self.number_of_points-1):
                x = rd.uniform(0, self.chord_length)
                y = rd.uniform(-self.thickness_multiplier/2, self.thickness_multiplier/2)

                points.append([x, y])
            points.append([0, 0])
            points = np.ndarray(points)
            points = self.order_points()

            return points

        # If not random : NACA0010 profile (for splines), points already ordered
        points = np.array([
            [1.00000,  0.00105],
            [0.8, 0.02],
            [0.5, 0.043],
            [0.3, 0.05],
            [0.12, 0.042],
            [0.025, 0.022],
            [0.0,  0.0],
            [0.025, -0.022],
            [0.12, -0.042],
            [0.3, -0.05],
            [0.5, -0.043],
            [0.8, -0.02],
            [1.00000,  -0.00105],
            [1.00000,  0.00000]
            ])
        
        multiplier = np.full((len(points), 2), np.array([self.chord_length,self.thickness_multiplier]))
        points = np.array(np.multiply(points, multiplier))

        return points
    

    
    def compute_surface(self) :
        """ Computes the area of the polygon formed by the points stored in self.points"""
        points = np.array(self.points)
        points = points.reshape(-1,2)

        x = points[:,0]
        y = points[:,1]

        S1 = float(np.sum(x*np.roll(y,-1)))
        S2 = float(np.sum(y*np.roll(x,-1)))

        self.surface = 0.5*np.abs(S1-S2)

        return self.surface
        
    
    def order_points(self):
        """
        Orders the point of the airfoil, with increasing indices in the trigonometric direction, and starting from the rightmost point right above the x-axis.
        """
        # sort by polar angle (counter-clockwise)
        angles = [np.arctan2(p[1], p[0]) for p in self.points]

        order = np.argsort(angles)
        sorted_points = [self.points[i] for i in order]

        # find the "top-right" point: maximal x, break ties by maximal y
        start_idx = max(range(len(sorted_points)), key=lambda i: (sorted_points[i][0], sorted_points[i][1]))

        # rotate so the sequence starts at the top-right point
        self.points = sorted_points[start_idx:] + sorted_points[:start_idx]

    def plot(self):
        """
        Plots the airfoil using matplotlib.
        """
        x, y = zip(*self.points)
        plt.figure()
        plt.plot(x, y, 'o-')
        plt.title('Airfoil Shape')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def apply_symmetrical_y_actions(self, actions):
        """ 
        Changes self.points according to the actions, for a symmetrical foil profile
        and changing only the y coordinate of the points
        Args :
            actions : np.array([x,y], ....)
        Actions arguments are ordered as the points (trigonometric order starting at the top of 
        leading edge for NACA0010 basic profil of the class Foil)
        """
        current_points = np.array(self.points)

        origin = self.origin
        #print("Origine : ", self.origin)

        # Get current points and extract their y coordinates for points bewteen trailing edge and origin-1
        """ action x = np.array([thickness_te/2, y1, y2, y3, y4, y5]) """
        new_y_s = actions
        new_points = np.array(current_points)
        if np.any(actions <= 0.0):
            raise ValueError("Actions cannot be negative ! (y_coord of upper surface, symmetrical foil points)")
        if len(new_y_s) > origin :
            raise ValueError("Action is too large, larger than the number of control points")
        if len(new_y_s) < origin :
            raise ValueError("Action is too short, shorter than the number of control points")
        try:
            for i in range(len(current_points[:origin,1:])):
                new_points[i, 1] = new_y_s[i]
                new_points[-(i+2), 1] = -new_y_s[i]
        except ValueError as e:
            print(f"Error: {e}")

        self.points = new_points
        return
    
    def camber_thickness(self, actions):
        """
        Changes the airfoil points based actions structured as follows:
        - The first N actions are the y coordinates of the camber points 
        (except the leading and trailing edges which are fixed), whose x coordinates are know from the airfoil points
        - The N following actions are the thickness distribution, which is applied symmetrically in the horizontal direction at the camber points (except for the leading and trailing edges)
        - The last action is the angle of rotation

        Args :
            actions (np.array): The actions to apply to the airfoil.
        Returns :
            None
        """
        N = (len(actions)-1)//2
        print(f"N = {N}")
        #The new foil will have len(self.points) = 2*(N+1)
        camber = actions[:N] #camber is of length N
        print(f"Camber = {camber}, {np.shape(camber)}")
        thickness = actions[N:-1] #thickness is of length N-1
        print(f"Thickness = {thickness}, {np.shape(thickness)}")
        rotation = actions[-1]
        print(f"Rotation = {rotation}")
        origin = self.origin

        points = self.points
        new_points = np.array(points)
        if np.any(thickness < 0.0):
            raise ValueError("Thickness cannot be negative !")
        if len(camber) > origin-1 :
            raise ValueError("Action is too large, larger than the number of control points")
        if len(camber) < origin-1 :
            raise ValueError("Action is too short, shorter than the number of control points")
        try:
            for i in range(N):
                
                c = camber[i]
                print(f"Camber {i} : {c}")
       
                t = thickness[i]
                print(f"Thickness {i} : {t}")

                
                new_points[i+1, 1] = c + t/2
                new_points[-(i+3), 1] = c - t/2
                print(f'Initial point {i+1} of coordinates {points[i+1, 1]} moved to {new_points[i+1, 1]}')
                print(f'Initial point {-(i+3)%len(points)} of coordinates {points[-(i+3), 1]} moved to {new_points[-(i+3), 1]}')

        except ValueError as e:
            print(f"Error: {e}")

        self.points = new_points
        self.apply_rotation(rotation)

        return 

    def apply_transform_point(self, 
                  point_ind : int,
                  transformation_parameters,
                  transformation : str = "new_coord",
                  constraint_parameter = 1.0,
                  constraint : str = "max_thickness"):
        """
        Applies a transformation to a specific point of the airfoil.
        Since we want to avoid entanglement, we need to be careful with how we transform points.
        Args: 
            point_ind (int): The index of the point to transform.
            transformation_parameters (tuple[float, float]): The parameters for the transformation.
            transformation (str): The type of transformation to apply 
                (among 'new_coord' which changes the point's coordinates, 'translation' which moves the point, and 'scaling' which scales the point).
        """
        if 0 <= point_ind < len(self.points):
            candidate = list(self.points)
            constraint_satisfied = False 
            # Work on a copy and validate the transformation won't create intersections
            
            #COMPUTE NEW POINT CANDIDATE
            if transformation == "new_coord":
                new_x, new_y = transformation_parameters
                candidate[point_ind] = (new_x, new_y)

            elif transformation == "translation":
                dx, dy = transformation_parameters
                candidate[point_ind] = (candidate[point_ind][0] + dx,
                                        candidate[point_ind][1] + dy)
            elif transformation == "scaling":
                sx, sy = transformation_parameters
                candidate[point_ind] = (candidate[point_ind][0] * sx,
                                        candidate[point_ind][1] * sy)
            else:
                raise ValueError("Unknown transformation type. Please select from new_coord, translation and scaling")
            
            #CHECK WHETHER NEW POINT CANDIDATE RESPECTS CONSTRAINTS OR NOT
            if constraint == "max_thickness":
                #checks whether the delta between the max of the y coordinates of all points and the min is above constraint_parameter
                y_max = max(self.points[1])
                y_min = min(self.points[1])
                delta = y_max - y_min
                # print(delta)

                constraint_satisfied = (delta > constraint_parameter)

            if not _is_self_intersecting(candidate) and constraint_satisfied:
                self.points = candidate
            else:
                print("Transformation would create self-intersections or violate constraints")
                return -10
                
                
            
        else:
            raise IndexError("Point index out of range")
        
        return 10


    def apply_translation(self, x, y):
        """
        Translates the whole airfoil by a given (x, y) vector.
        """
        shape = np.array(self.points).shape
        translation = np.array(np.full(shape, [x, y]))
        self.points = self.points + translation


    def apply_rotation(self, angle):
        """
        Rotates the whole airfoil by a given angle in radians around its center of mass.
        """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        #Compute the coordinates of the center of mass
        x_center = np.mean([x for x, y in self.points])
        y_center = np.mean([y for x, y in self.points])
        rotated_points = []
        for x, y in self.points: #rotates around the center of mass
            x_rot = x_center + (x - x_center) * cos_angle - (y - y_center) * sin_angle
            y_rot = y_center + (x - x_center) * sin_angle + (y - y_center) * cos_angle
            rotated_points.append((x_rot, y_rot))
        self.points = rotated_points


    def get_txt(self):
        """
        Creates a .txt file from the airfoil points, with one point (x,y) per line, and returns its file path.
        """
        file_path = os.path.join(self.txt_dir, f"{self._base()}.txt")

        with open(file_path, "w") as f:
            for x, y in self.points:
                f.write("{}    {}\n".format(x, y))
        return file_path


    def get_geo(self, 
                h : float,
                type : str = "spline"
        ) -> str:
        """
        Creates a .geo file for GMSH from the airfoil points, and returns its file path.

        Args:
            input_file (str): The path to the input .txt file.
            h (float): The mesh size.
            type (str): The type of geometry to generate ("line" or "spline" or "bspline").

        Returns:
            str: The path to the generated .geo file.
        """
        
        input_file = self.get_txt()
        output_file = os.path.join(self.geo_dir, f"{self._base()}.geo")
        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:

            f_out.write("// Fichier .geo généré à partir de {}\n".format(input_file))
            f_out.write("h = {};\n".format(h))
            point_id = 1
            zero = 0

            for line in f_in:

                x_str, y_str = line.split()
                x, y = float(x_str), float(y_str)
                if (x, y) == (0.0, 0.0):
                    zero = point_id
                f_out.write("Point({}) = {{{}, {}, 0, h}};\n".format(point_id, x, y))
                point_id += 1

            if type == "line":
                # On relie les points par des lignes
                for i in range(1, point_id - 1):
                    f_out.write("Line({}) = {{{}, {}}};\n".format(i, i, i + 1))
                # On crée une ligne fermée
                f_out.write("Line({}) = {{{}, 1}};\n".format(point_id - 1, point_id - 1))

            if type == "bspline":
                L_attaque = [zero - 1, zero, zero + 1]
                L_up = [i for i in range(1, zero)]
                L_down = [i for i in range(zero + 1, point_id)] + [1]

                L = [i for i in range(1, point_id-1)]

                f_out.write("BSpline(1) = " + str((L)).replace("[", "{").replace("]", "}") + ";\n")
                f_out.write("Line(2) = {{1, {}, {}}};\n".format(point_id-1, point_id-2))

            if type == "spline":
                L_attaque = [zero - 1, zero, zero + 1]
                L_up = [i for i in range(1, zero)]
                L_down = [i for i in range(zero + 1, point_id)] + [1]

                L = [i for i in range(1, point_id-1)]

                f_out.write("Spline(1) = " + str((L)).replace("[", "{").replace("]", "}") + ";\n")
                f_out.write("Line(2) = {{1, {}, {}}};\n".format(point_id-1, point_id-2))

            f_out.write("Curve Loop(1) = {-1, 2};\n")
            f_out.write("Plane Surface(1) = {1};\n")

        # print("Fichier .geo généré : {}".format(output_file))
        return output_file


    def get_mesh_windows(self):
        import subprocess
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_msh_dir = os.path.join(script_dir, "tmp_msh_files")
        os.makedirs(tmp_msh_dir, exist_ok=True)

        geo_file = self.get_geo(self.msh_size, self.type)
        tmp_msh_file = os.path.join(tmp_msh_dir, f"{self.name}.msh")

        copy(geo_file, tmp_msh_file)
        
        msh_dir = os.path.join(script_dir, "msh_files")
        os.makedirs(msh_dir, exist_ok=True)
        
        msh_output = os.path.join(msh_dir, f"{self.name}.msh")
        
        # Write mesh commands to the file
        with open(tmp_msh_file, "a") as f_out:
            f_out.write(f"Mesh 2;\nSave \"{msh_output}\";\n")
        
        gmsh_exe = shutil.which("gmsh")
        if gmsh_exe is None:
            raise RuntimeError("gmsh not found in PATH")
        print(f"Using gmsh at: {gmsh_exe}")
        
        original_dir = os.getcwd()
        
        try:
            os.chdir(script_dir)
            rel_output = os.path.join("tmp_msh_files", f"{self.name}.msh")
            rel_msh_output = os.path.join("msh_files", f"{self.name}.msh")
            
            result = subprocess.run(
                [gmsh_exe, rel_output, "-2", "-o", rel_msh_output],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                #print(f".msh file generated: {msh_output}")
                pass
            else:
                print(f"GMSH FAILED with return code {result.returncode} for episode {self.suffix}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                raise RuntimeError(f"GMSH failed to generate mesh")
            
        finally:
            os.chdir(original_dir)

        return msh_output
    

    def get_mesh(self):
        geo_file_path = self.get_geo(self.msh_size, self.type)
        msh_output = os.path.join(self.msh_dir, f"{self._base()}.msh")
        try:
            gmsh.initialize(sys.argv)
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(geo_file_path)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)
            gmsh.model.mesh.generate(2)
            gmsh.write(msh_output)
        finally:
            # ensure finalize even on error
            try:
                gmsh.finalize()
            except Exception:
                pass
        return msh_output


    def get_t(self) -> None:
        
        """
        GMSH4MTC

        Utilitaire de conversion des maillages gmsh (.msh) de format 2 et 4
        2D et 3D en format mtc (.t).
        Fonctionne aussi desormais avec les fichiers stl.

        Fonctionne pour les maillages surfaciques (2D et 3D) et volumiques.
        Uniquement pour les éléments triangulaires, tetrahedriques.

        maxime.renault@minesparis.psl.eu
        06/2022

        Inspiré du travail de Tommy Carozzani en 2010.
       
        Convert a GMSH mesh file from version 2 to version 4.

        Args: 
            input (str): The path to the input .msh file in gmsh format.
            output (str): The path to the output .t file in mtc format.

        Returns:
            None
        """
        #print("Initialisation...\n")
        input = os.path.join(self.msh_dir, f"{self._base()}.msh")
        output = os.path.join(self.t_dir, f"{self._base()}.t")

        with open(input) as f:
            f.readline()
            """
            version = f.readline().split()[0]
            if len(version) > 1:
                version = version.split('.')[0]
            if version != '4' and version != '2':
                print("This version of gmsh isn't supported")
                sys.exit()
            """
            version = '2'

            flags = {"$Nodes": [],
                    "$EndNodes": [],
                    "$Elements": [],
                    "$EndElements": []}

            connect_3d = []
            connect_2d = []
            connect_1d = []

            ###########

            #print("Getting position flags...\n")

            t = f.readline()

            while t:
                t = t.strip("\t\n")
                if t.startswith("$"):
                    for i in range(len(list(flags.keys()))):
                        if(t == list(flags.keys())[i]):
                            flags[t].append(f.tell())
                            break
                t = f.readline()

            ###########

            #print("Treating connectivities...\n")

            if version == '4':
                for index in range(len(flags["$Elements"])):
                    f.seek(flags["$Elements"][index])

                    t = f.readline()    # line ignored (nb of elements)
                    t = f.readline()

                    while (t and f.tell() != flags["$EndElements"][index]):
                        t = t.strip("\t\n").split()

                        if(len(t) <= 1):
                            break
                        
                        if(t[2] != '2' and t[2] != '4'):
                            for i in range(int(t[-1])):
                                f.readline()
                        
                        if(t[2] == '2'):    # triangle
                            for i in range(int(t[-1])):
                                elem = f.readline().strip("\t\n").split()
                                lig = [int(elem[1]), int(elem[2]), int(elem[3])]
                                connect_2d.append(lig)

                        if(t[2] == '4'):    # tetrahedron
                            for i in range(int(t[-1])):
                                elem = f.readline().strip("\t\n").split()
                                lig = [int(elem[1]), int(elem[2]), int(elem[3]), int(elem[4])]
                                connect_3d.append(lig)

                        t = f.readline()

            if version == '2':
                for index in range(len(flags["$Elements"])):
                    f.seek(flags["$Elements"][index])

                    t = f.readline()    # line ignored (nb of elements)
                    t = f.readline()
                    
                    while (t and f.tell() != flags["$EndElements"][index]):
                        t = t.split()
                        
                        if(len(t) <= 1):
                            break

                        if(t[1] == '2'):    # triangle
                            lig = [int(t[-3]), int(t[-2]), int(t[-1])]
                            connect_2d.append(lig)

                        if(t[1] == '4'):    # tetrahedron
                            lig = [int(t[-4]), int(t[-3]), int(t[-2]), int(t[-1])]
                            connect_3d.append(lig)
                            
                        t = f.readline()

            connect_2d = np.array(connect_2d)
            connect_3d = np.array(connect_3d)

            ###########

            #print("Verifying nodes and edges...")

            # nodes

            nodes = []

            if version == '4':
                for index in range(len(flags["$Nodes"])):
                    f.seek(flags["$Nodes"][index])
                    f.readline()    # line ignored (nb of nodes)

                    t = f.readline()

                    while (t and f.tell() != flags["$EndNodes"][index]):
                        t = t.strip("\t\n").split()
                        
                        if(len(t) <= 1):
                            break
                        
                        for i in range(int(t[-1])):
                            f.readline()

                        for i in range(int(t[-1])):
                            node = f.readline().strip("\t\n").split()
                            nodes.append([float(node[0]), float(node[1]), float(node[2])])
                        
                        t = f.readline()

            if version == '2':
                for index in range(len(flags["$Nodes"])):
                    f.seek(flags["$Nodes"][index])
                    f.readline()    # line ignored (nb of nodes)

                    t = f.readline()

                    while (t and f.tell() != flags["$EndNodes"][index]):
                        t = t.strip("\t\n").split()
                        
                        if(len(t) <= 1):
                            break

                        nodes.append([float(t[1]), float(t[2]), float(t[3])])
                        
                        t = f.readline()

        nodes = np.array(nodes)
        
        dim = 3
        if(len(connect_3d) == 0):
            if(np.all(nodes[:, 0] == nodes[0, 0])):
                dim = 2
                nodes = nodes[:, 1:]
            elif(np.all(nodes[:, 1] == nodes[0, 1])):
                dim = 2
                nodes = nodes[:, -1:1]
            elif(np.all(nodes[:, 2] == nodes[0, 2])):
                dim = 2
                nodes = nodes[:, :2]
            else:
                dim = 2.5

        # Apparently Cimlib prefers normals looking down in 2D
        if(dim == 2):
            #print("   - Checking normals")  # Actually only checking the first normal
            normal = np.cross(nodes[connect_2d[0][1]] - nodes[connect_2d[0][0]], nodes[connect_2d[0][2]] - nodes[connect_2d[0][0]])
            if(normal > 0):
                connect_2d = connect_2d[:, [0, 2, 1]]

        #print("   - Detecting edges")

        if dim == 3:
            del connect_2d
            
            tris1 = connect_3d[:, [0, 2, 1]]  # Order is very important !
            tris2 = connect_3d[:, [0, 1, 3]]
            tris3 = connect_3d[:, [0, 3, 2]]
            tris4 = connect_3d[:, [1, 2, 3]]
            
            tris = np.concatenate((tris1, tris2, tris3, tris4), axis=0)
            tris_sorted = np.sort(tris, axis=1)  # creates a copy, may be source of memory error
            tris_sorted, uniq_idx, uniq_cnt = np.unique(tris_sorted, axis=0, return_index=True, return_counts=True)
            connect_2d = tris[uniq_idx][uniq_cnt == 1]

        if dim == 2:
            lin1 = connect_2d[:, [0, 1]]  # Once again, order is very important !
            lin2 = connect_2d[:, [2, 0]]
            lin3 = connect_2d[:, [1, 2]]
            
            lin = np.concatenate((lin1, lin2, lin3), axis=0)
            lin_sorted = np.sort(lin, axis=1)   # creates a copy, may be source of memory error
            lin_sorted, uniq_idx, uniq_cnt = np.unique(lin_sorted, axis=0, return_index=True, return_counts=True)
            connect_1d = lin[uniq_idx][uniq_cnt == 1]
            
        #print("   - Detecting unused nodes")

        to_delete = np.arange(1, len(nodes) + 1)   # Every node index
        used_elems = np.unique(np.concatenate((connect_3d.flat, connect_2d.flat)))   # Every index of USED nodes

        bools_keep = np.in1d(to_delete, used_elems)
        to_delete = to_delete[~bools_keep]
        del used_elems

        #print("   - Deleting unused nodes\n")

        nodes = nodes[bools_keep]
        del bools_keep

        if dim == 3 or dim == 2.5:
            connect_3d.flat -= np.searchsorted(to_delete, connect_3d.flat, side='left')
            connect_2d.flat -= np.searchsorted(to_delete, connect_2d.flat, side='left')

        if dim == 2:
            connect_2d.flat -= np.searchsorted(to_delete, connect_2d.flat, side='left')
            connect_1d.flat -= np.searchsorted(to_delete, connect_1d.flat, side='left')

        ##########

        nb_elems = len(connect_2d) + len(connect_3d)
        if dim == 2:
            nb_elems += len(connect_1d)
            #print("Nb elements 1d : " + str(len(connect_1d)))

        #print("Nb elements 2d : " + str(len(connect_2d)))
        #print("Nb elements 3d : " + str(len(connect_3d)))
        #print("Dimension : " + str(dim) + "\n")

        ###########

        # print("Writing .t file...")

        with open(output, "w") as fo:
            lig = str(len(nodes)) + " " + str(dim) + " " + str(nb_elems) + " " + str(dim + 1) + "\n"
            if(dim == 2.5):
                lig = str(len(nodes)) + " 3 " + str(nb_elems) + " 4\n"
            fo.write(lig)

            for node in nodes:
                fo.write("{0:.16f} {1:.16f}".format(node[0], node[1]))
                if(dim == 3 or dim == 2.5):
                    fo.write(" {0:.16f}".format(node[2]))
                fo.write(" \n")

            for e in connect_3d:
                fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " " + str(e[3]) + " \n")

            for e in connect_2d:
                if(dim == 3 or dim == 2.5):
                    fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " 0 \n")
                else:
                    fo.write(str(e[0]) + " " + str(e[1]) + " " + str(e[2]) + " \n")

            if dim == 2:
                for e in connect_1d:
                    fo.write(str(e[0]) + " " + str(e[1]) + " 0 \n")

        print(f".t file written at {output}.")
        return output

    def sync(self) -> str:
        self.get_mesh()
        return self.get_t()


#TESTS
TEST1 = 0
if TEST1:
    airfoil = Foil(20, 1.0, 1.0)
    print(airfoil.points)
    airfoil.plot()
    airfoil.apply_transform_point(0, (0.01, -0.01), "translation", 0.05, "max_thickness")
    airfoil.plot()

TEST2 = 0
if TEST2:
    airfoil = Foil(10, 1.0, 1.0)
    print(type(airfoil.points))
    print(airfoil.points)
    airfoil.plot()
    airfoil.sync()
    airfoil.apply_rotation(3.14)
    airfoil.plot()
    airfoil.sync()

TEST3 = 0
if TEST3:
    airfoil = Foil(10, 1.0, 1.0)
    n = 10
    airfoil.apply_translation(2.5,2)
    airfoil.plot()
    airfoil.number_of_points = len(airfoil.points)
    airfoil.name = f"object"
    
    airfoil.sync()
    count = 0
    while count < n:

        t = airfoil.apply_transform_point(rd.randint(0, airfoil.number_of_points - 1), 
                          (rd.uniform(-0.05, 0.05), rd.uniform(-0.05, 0.05)), 
                          transformation="translation", 
                          constraint_parameter=0.1)
        if t>0:
            #airfoil.plot()
            airfoil.name = f"object{count}"
            airfoil.sync()
            count += 1

TEST4 = 0
if TEST4:
    airfoil = Foil(10, 1.0, 1.0)
    
    airfoil.plot()
    airfoil.number_of_points = len(airfoil.points)
    airfoil.name = f"object"
    airfoil.apply_translation(2.5,2)
    airfoil.get_mesh_windows()
    airfoil.get_t()

TEST5 = 0
# Surface computation test
if TEST5:
    airfoil = Foil(10, 1.0, 1.0)

    airfoil.number_of_points = len(airfoil.points)
    airfoil.name = f"object"
    #airfoil.translate(2.5,2)
    #airfoil.transform(2, (0.0, 0.15), "translation", 0.1)
    #print(airfoil.compute_surface())
    action = np.array([0.02, 0.04, 0.03, 0.027, 0.02, 0.08])
    airfoil.apply_symmetrical_y_actions(action)
    airfoil.plot()
    airfoil.sync()

TEST6 = 0
if TEST6:
    airfoil = Foil(10, 1.0, 1.0)
    airfoil.plot()
    actions = np.array([0.01405, 0.095, 0.085, 0.061, 0.058, 0.042])
    airfoil.apply_symmetrical_y_actions(actions)
    airfoil.plot()

TEST7 = 0
if TEST7:
    airfoil = Foil(10, 1.0, 1.0)
    airfoil.name = "new_transfo"
    # points = airfoil.points
    # print(f"Points: {len(points)}")
    # print(f"points[0], points[1], points[13]: {points[0]}, {points[1]}, {points[13]}")
    actions = np.array([
            0.1, 0.12, 0.13, 0.07, 0.03, #camber
            0.05, 0.1, 0.1, 0.1, 0.05,    #thickness
            -0.3                               #rotation
    ])
    airfoil.camber_thickness(actions)
    
    airfoil.apply_translation(2.5,2)

    print(airfoil.sync())
