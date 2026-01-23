# Generic imports
import os
import math
import time
import numpy as np
import glob

from scipy.interpolate import splprep, splev
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# Import file for geometry creation
from geometry.mesh.Foil import *
# Import file for reward computation
from avg_liftdrag import *
from idw import *


###############################################
### Panels in wind environment
class airfoil():

    def __init__(self, path):

        # Fill structure
        self.name     = 'airfoil'
        self.act_size = 6                           # Actions change y-coord of 6 points (leading edge is fixed and only allowed to change thickness of trailing edge
        """ action x = np.array([thickness_te/2, y1, y2, y3, y4, y5])"""
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        #self.shape_h  = 1.0
        self.path     = path
        self.x_min    = -np.ones(self.act_size)     # Generates action-sized array filled with -1.0 
        self.x_max    = np.ones(self.act_size)      # Generates action-sized array filled with 1.0 
        self.x_0      = np.array([2*np.random.rand((1))-1.0 for i in range(self.act_size)])   # initial action

        self.foil_area = 0.0                          # Airfoil area initialization
        
        max_y = 0.06
        max_t_le = 0.04
        max_t_te = 0.01

        self.physical_scale = np.concatenate([max_y*np.ones(6),[max_t_le],[max_t_te]])                       # Remap to physical scale
        self.bad_rwrd = -100000
        self.cores    = '8'                         # num of cores per env 
        self.dim      = '2d'
        # self.dt       = 0.05                      # using 1 because 3D uses varying dt, dt at the end is 0.1
        # self.window   = [149.99,250.01]           # Time window for drag/lift calculus. Corresponds to [175,200], but increment numbers differ due to varying dt in cfd
        self.timeout  = '3600'                     # timeout limit in seconds (s) -> 24h timeout on 3D
        
        # Coefficients for Combined reward
        # self.MEAN_COEFF  = 1.0      # modify accordingly
        # self.COST_COEFF = 0.025      # modify accordingly

        # Set episode number
        self.episode  = 0
    
    def cfd_solve(self, x, ep):

        ## Create folders and copy cfd folder
        self.output_path       = self.path+'/'+str(ep)+'/'
        self.vtu_path          = self.output_path+'/vtu/'
        self.liftsensor_path   = self.output_path+'/LiftSensor/'
        self.efforts_path   = self.output_path+'/Efforts/'

        os.makedirs(self.liftsensor_path, exist_ok= True)
        os.makedirs(self.vtu_path, exist_ok= True)
        os.makedirs(self.efforts_path, exist_ok= True)
        os.system('cp -r cfd ' + self.output_path + '.')

        # (actions are already remapped to physical scale at this stage)
        self.write_actions(x,ep) # Saves action(s) to a file

        # Convert action to rotation velocities
        # rotation = np.array(x)

        # Convert action to angle of attack
        angle = np.array(x)

        name = "object"
        self.surface = self.create_geometry(x, name)

        """
        ## Change the rotation velocities in mtc
        # self.line_replace('Front_rotation_velocity', '{ Target= OmegaFront '+str(rotation[0])+' } // Front_rotation_velocity', self.output_path + 'cfd/IHM.mtc')
        # self.line_replace('Top_rotation_velocity', '{ Target= OmegaTop '+str(rotation[1])+' } // Top_rotation_velocity', self.output_path + 'cfd/IHM.mtc')
        # self.line_replace('Bottom_rotation_velocity', '{ Target= OmegaBottom '+str(rotation[2])+' } // Bottom_rotation_velocity', self.output_path + 'cfd/IHM.mtc')
        """

        ## Solve problem using cimlib and move vtu and drag folder
        os.system('cd '+self.output_path+'cfd/.; touch run.lock; mpirun -n '+self.cores+' --timeout '+self.timeout+' /home/mbailly/cimlib_CFD_driver lanceur/Principale.mtc > log.txt;') #trash.txt
        time.sleep(2)
        os.system('cp '+self.output_path+'cfd/resultats/capteurs/*.txt '+self.liftsensor_path+'.') 
        os.system('cp '+self.output_path+'cfd/resultats/*.txt '+self.efforts_path+'.') # Copy the efforts.txt
        os.system('mv '+self.output_path+'cfd/resultats/'+self.dim+'/* '+self.vtu_path+'.') # Move vtu.s
        os.system('rm -r '+self.output_path+'cfd') # Remove the copied cfd folder
        
        """
        # self.reward = 1 * ( self.reward_meandrag(window=self.window, dt=self.dt, rescale=self.MEAN_COEFF) 
        #                      + self.reward_cost(rescale=self.COST_COEFF) )
        # self.func = self.reward_meandrag(window=self.window, dt=self.dt, rescale=self.MEAN_COEFF)
        # self.cost = self.reward_cost(rescale=self.COST_COEFF)
        # self.reward = self.func + self.cost 
        # self.total = [-self.reward, -self.func, -self.cost]
        """

        # Reward
        self.reward = self.compute_reward()

        #self.write_rewards(self.total,ep)
        self.write_rewards([self.reward],ep)

        print(self.reward)

        ## Increment episode
        self.episode += 1

        return self.reward

    def step(self, actions, ep):

        self.write_actions_alt(actions,ep)

        conv_actions = self.convert_actions(actions)
        try:
            reward = self.cfd_solve(conv_actions, ep)
        except Exception as e:
            print("\n !!!!!! CFD failed !!!!!!", flush=True)
            return self.bad_rwrd, conv_actions

        return reward, conv_actions

    def convert_actions(self, actions):

        # Convert actions to physical scale
        conv_actions  = np.multiply(actions, self.physical_scale)
        return conv_actions

    def observe(self):

        # Always return the same observation
        return self.obs

    def close(self):
        pass

    def line_replace(self, string, line, target):

        command = "sed -i '/"+string+"/c\\"+line+"' "+target
        os.system(command)

    def order_points(self, points):
        """
        Orders the point of the airfoil, with increasing indices in the trigonometric direction, and starting from the rightmost point.
        """
        points = np.array(points)
        # sort by polar angle (counter-clockwise)
        angles = [np.arctan2(p[1], p[0]) for p in points]
        order = np.argsort(angles)
        sorted_points = [points[i] for i in order]

        # find the "top-right" point: maximal x, break ties by maximal y
        start_idx = max(range(len(sorted_points)), key=lambda i: (sorted_points[i][0], sorted_points[i][1]))

        # rotate so the sequence starts at the top-right point
        points = sorted_points[start_idx:] + sorted_points[:start_idx]

    def compute_init_mesh_displacements(self, init_cp, new_cp, init_foil_origin: int):
        origin = init_foil_origin                           # Locate the leading edge of the foil in the mesh
        init_displacements = np.zeros((len(init_cp),2))

        for i in range(len(init_cp[:origin,1:])):
            init_displacements[i] = new_cp[i] - init_cp[i]
            init_displacements[-(i+2)] = new_cp[-(i+2)] - init_cp[-(i+2)]
        init_displacements[origin] = new_cp[origin] - init_cp[origin]   # Add leading edge displacement
        init_displacements[-1] = new_cp[-1] - init_cp[-1]   # Add trailing edge displacement
        #print("compute_init_displacements_mesh : Displacements : ", len(init_displacements))
        return init_displacements
    
    def artificial_cp(self, init_foil, new_foil, n):
        
        init_acp = init_foil.create_acp(n)
        new_acp = new_foil.create_acp(n)
        
        acp_displacements = new_acp - init_acp

        # for i in range(len(acp_displacements)):
        #     print(f"Artificial control point {init_acp[i]} displaced by {acp_displacements[i]}")

        return init_acp, acp_displacements
    
    def compute_spline_length(self, tck, u_a, u_b):
        """
        Computes the arc length of a B-spline between two parameter values.
        
        spl: scipy BSpline object
        a and b: np.ndarray containing the points coordinates [x, y] between which spline length is computed
        Returns the length of the B-spline between points a and b
        """
        u_fine = np.linspace(u_a, u_b, 100)
        x_s, y_s = splev(u_fine, tck)
        pts = np.array(np.vstack((x_s, y_s)).T)

        length = np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))   
        return length
    
    def artificial_cp_spline(self, init_foil, new_foil, density, k=2):

        new_foil_points = new_foil.points
        init_foil_points = init_foil.points

        acp_displacements = []
        init_acp = []

        ### Create spline interpolating every point except last one (trailing edge not on same spline)
        points_init = np.array(init_foil_points[0:-1])
        points_new = np.array(new_foil_points[0:-1])
        x_init = points_init[:,0]
        y_init = points_init[:,1]
        x_new = points_new[:,0]
        y_new = points_new[:,1]

        (tck_init, u_init) = splprep([x_init, y_init], s=0, k=k)
        #print("u_init is : ",u_init)
        (tck_new, u_new) = splprep([x_new, y_new], s=0, k=k)

        print("Coords second spline point new : ", np.array(splev(u_new[1], tck_new)))
        print("Coords second foil point new : ", points_new[1])
        for i in range(len(points_init)-1):
            # Compute the length of the spline segment created between consecutive points
            new_point_i = u_new[i]
            new_point_ip1 = u_new[i+1]
            length_new = self.compute_spline_length(tck_new, new_point_i, new_point_ip1)
        
            # Create n artificial control points along the splines segment according to and new spline length density
            n = int(np.floor(2.0*length_new*density))

            init_point_i = u_init[i]
            init_point_ip1 = u_init[i+1]
                                    
            new_acp_u = np.linspace(new_point_i, new_point_ip1, (n+2))[1:-1]    
            init_acp_u = np.linspace(init_point_i, init_point_ip1, (n+2))[1:-1]
    
            # Get the coordinates of the init and new artificial control points
            x_acp_new, y_acp_new = splev(new_acp_u, tck_new)
            x_acp_init, y_acp_init = splev(init_acp_u, tck_init)
            # Compute points displacements
            new_acp_partial = np.array(np.vstack((x_acp_new, y_acp_new)).T)
            init_acp_partial = np.array(np.vstack((x_acp_init, y_acp_init)).T)

            acp_displacements_partial = new_acp_partial - init_acp_partial
            acp_displacements.append(acp_displacements_partial)
            init_acp.append(init_acp_partial)

            # print(f"Loop for {i+1}th spline running")
            # print("Longueur spline new : ", length_new)
            # print("Nombre de acp à ajouter spline new : ", n)
            # print(new_point_i, "   ", new_point_ip1)
            # print("Paramètres 0 à 1 de spline new : ", new_acp_u)

            # print("init_acp {i+1}th spline : ", init_acp_partial)
            # print("new_acp {i+1}th spline : ", new_acp_partial)

        #print("Artificial control points displacement :", acp_displacements)            
        return init_acp, acp_displacements

    def artificial_cp_bezier(self, init_foil, new_foil, density):
        new_foil_points = new_foil.points
        init_foil_points = init_foil.points

        ### Create spline interpolating every point except last one (trailing edge not on same spline)
        points_init = np.array(init_foil_points[0:-1])
        points_new = np.array(new_foil_points[0:-1])
        x_init = points_init[:,0]
        y_init = points_init[:,1]
        x_new = points_new[:,0]
        y_new = points_new[:,1]

        (tck_init, u_init) = splprep([x_init, y_init], s=0.002, k=3)
        #print("u_init is : ",u_init)
        (tck_new, u_new) = splprep([x_new, y_new], s=0.002, k=3)

        for i in range(len(points_init)-1):
            # Compute the length of the spline segment created between consecutive points
            new_point_i = u_new[i]
            new_point_ip1 = u_new[i+1]
            length_new = self.compute_spline_length(tck_new, new_point_i, new_point_ip1)
        
            # Create n artificial control points along the splines segment according to and new spline length density
            n = int(np.floor(2.0*length_new*density))

            init_point_i = u_init[i]
            init_point_ip1 = u_init[i+1]
                                    
            new_acp_u = np.linspace(new_point_i, new_point_ip1, (n+2))[1:]
            init_acp_u = np.linspace(init_point_i, init_point_ip1, (n+2))[1:]
    
            # Get the coordinates of the init and new artificial control points
            x_acp_new, y_acp_new = splev(new_acp_u, tck_new)
            x_acp_init, y_acp_init = splev(init_acp_u, tck_init)
            # Compute points displacements
            new_acp_partial = np.array(np.vstack((x_acp_new, y_acp_new)).T)
            init_acp_partial = np.array(np.vstack((x_acp_init, y_acp_init)).T)

            acp_displacements_partial = new_acp_partial - init_acp_partial

            if i == 0:
                acp_displacements = acp_displacements_partial
                init_acp = init_acp_partial
            else :
                acp_displacements = np.concatenate([acp_displacements,acp_displacements_partial])
                init_acp = np.concatenate([init_acp,init_acp_partial])

        # Add the last point (middle of trailling edge) at the end
        init_cp_te = init_foil_points[-1]
        new_cp_te = new_foil_points[-1]

        acp_displacements_partial = np.array(new_cp_te) - np.array(init_cp_te)
     
        acp_displacements = np.concatenate([acp_displacements,[acp_displacements_partial]])
        init_acp = np.concatenate([init_acp,[init_cp_te]])

        #print("Artificial control points displacement :", acp_displacements)            
        return init_acp, acp_displacements

    def stack(self, cp, acp):
        """
        Returns a np.ndarray control points in the same order as on the foil's spline
        
        cp : np.ndarray of control points
        acp : np.ndarray of artificial control points to insert between cps
        """
        stacked_cp = []

        for i in range(len(cp)-2):
            stacked_cp.append(cp[i])
            # Insert acp[i] between cp[i] and cp[i+1]
            for point in acp[i]:
                stacked_cp.append(point)
        stacked_cp.append(cp[-2]) 
        stacked_cp.append(cp[-1]) # Append the last control points

        stacked_cp = np.array(stacked_cp)
        return stacked_cp

    def compute_idw_mesh(self, actions, init_naca, end_naca, interp_type = "bezier", p = 2):
        """
        Returns the new control points of the foil
        Deforms the original NACA0010 mesh according to actions (specific to symetrical y-change actions)
        """

        # Extract original mesh domain data
        original_mesh = extract_points(r"domain\domain_naca0010_12_4.t")

        # Select control points, including artificial ones
        init_cp = init_naca.points
        init_cp = np.array(init_cp)
        self.order_points(init_cp)
        new_cp = end_naca.points

        # Compute init_displacements
        init_displacements = self.compute_init_mesh_displacements(init_cp, new_cp, init_naca.origin)
        if interp_type == "spline":
            init_acp, acp_displacements = self.artificial_cp_spline(init_naca, end_naca, density = 100, k=2)
            # Intercaler chaque acp entre les cp initiaux
            foil_cp = self.stack(init_cp, init_acp)
            control_points = get_closest_point(foil_cp, original_mesh)
            control_points = np.array(control_points) # No need to order here thanks to stack method

            displacements = self.stack(init_displacements, acp_displacements)
            print("All control points displacements with spline interpolation : ", displacements)

        if interp_type == "bezier":
            init_acp, acp_displacements = self.artificial_cp_bezier(init_naca, end_naca, density = 100)
            foil_cp = init_acp
            control_points = get_closest_point(foil_cp, original_mesh)
            control_points = np.array(control_points)

            displacements = acp_displacements
            # print("All control points displacements with Bézier interpolation : ", displacements)

        if interp_type == "linear":
            init_acp, acp_displacements = self.artificial_cp(init_naca, end_naca, n=2)

            foil_cp = np.vstack((init_cp, init_acp))
            control_points = get_closest_point(foil_cp, original_mesh)
            control_points = np.array(control_points)

            # Order control points by trigonometric angle
            displacements = np.vstack((init_displacements, acp_displacements))

            print(f"actions: {actions}")
            print(f"Initially, {len(init_cp)} control points and {len(init_displacements)} displacements.")
            print(f"{len(init_acp)} articifial cps and {len(acp_displacements)} corresponding displacements were added.")
            print(f"Total cps: {len(control_points)}. Total displacements: {len(displacements)}")

        # Move the points in mesh data
        new_mesh = idw(original_mesh, control_points, displacements, p)

        # Write new .t file
        input_t_file_path = "domain/domain_naca0010_12_4.t"
        output_t_file_path = "cfd/meshes/domain.t"
        replace_points(input_t_file_path, output_t_file_path, new_mesh)

        return foil_cp + displacements
    
    def create_geometry(self, actions : np.ndarray, name : str, plot : bool = False, naca0010 : bool = False):
        """
        Generates the mesh for the object at the given angle of attack and 
        copies the mesh in the right cfd directory
        
        Args: 
            actions: ArrayLike containing the actions to undertake
            name (str): name of the foil

        Returns: 
            float = foil's approximate area (polygon between points)
            """
        x_trans_domain = 2.5
        y_trans_domain = 2.0

        foil = Foil(10, 1.0, 1.0)
        foil.generate_airfoil_points(random = False) # Generate NACA 0010 profile
        foil.name = name
        # Save original foil points before applying actions
        naca0010_foil = Foil(10, 1.0, 1.0)
        naca0010_foil.generate_airfoil_points(random = False)
        naca0010_foil.apply_translation(x_trans_domain,y_trans_domain)

        if not naca0010 : 
            foil.camber_thickness(actions)

        foil.apply_translation(x_trans_domain,y_trans_domain)
        self.foil_area = foil.compute_surface() # Computes approximate area of the foil (polygon)
        print(self.foil_area)
        if plot : foil.plot()

        # Deform the original domain with IDW according to actions and control points position
        control_points = self.compute_idw_mesh(actions, naca0010_foil, foil, interp_type="bezier", p = 3)

        # Get every new control points & give it to foil.points()
        foil.points = control_points

        # Generate new .t file via sync()
        foil.sync()

        # Copy the newly created .t file in the cfd/meshes folder
        t_file_path = os.path.join(self.path, "t", name + ".t")
        t_file_output_path = os.path.join(self.path, "cfd", "meshes", name + ".t")
        shutil.copy2(t_file_path, t_file_output_path)

        return foil.surface

    def compute_reward(self) -> float:
        """Compute the reward for the episode (ep) based on the forces data."""

        file_path = os.path.join(f"{self.efforts_path}/Efforts.txt")
        data = read_lift_drag(file_path)
        print(data)
        cx0_value, cy0_value = avg_lift_drag(data, plot=False)
        sface_penalty = abs(1.0-self.surface) # Not too big, not too small foil surface
        reward = cy0_value/cx0_value - sface_penalty  # Maximise lift/drag with constraints

        return reward
    
    def reward_cost(self, rescale=1000):
        '''computes total actuation cost
        rescaled by a multiplication factor.'''
        OM     = self.retrieve_omega()
        cost   = self.compute_cost(OM)
        reward = - cost*rescale
        return reward
        
    def reward_meandrag(self, window=[100,300], dt=0.05, rescale=1000):
        '''computes mean of the time-averaged drag on each cylinder
        rescaled by a multiplication factor. time-average is performed 
        on the time window specified'''
        D      = self.retrieve_drag()
        mean   = self.compute_mdrag(D, window=window, dt=dt)
        reward = - mean*rescale
        return reward
    
    def reward_rmsdrag(self, window=[100,300], dt=0.05, rescale=1000):
        '''compute RMS drag on each cylinders and average on three cylinders'''
        D      = self.retrieve_drag()
        rms    = self.compute_rms(D, window=window, dt=dt)
        reward = - rms*rescale
        return reward
    
    ### Useful functions for reward computation
    def compute_rms(self, D, window=[50,100], dt=0.1):
        # D is an array of shape (3,timesteps)
        DCropped      = D[ : , int(window[0]//dt) : int(window[1]//dt) ]
        Avg           = np.mean( DCropped , axis=1 )
        DMinusAvg     = DCropped - Avg[:,None]
        DMASquared    = DMinusAvg**2
        RMSIndividual = np.sqrt( np.mean(DMASquared, axis=1) )
        RMS           = np.sum(RMSIndividual)
        return(RMS)

    def compute_mdrag(self, D, window=[50,100], dt=0.1):
        # D is an array of shape (3,timesteps)
        DCropped = D[ : , int(window[0]//dt) : int(window[1]//dt) ]
        Avg      = np.mean( DCropped , axis=1 )
        TotalDrag = np.sum(Avg, axis=0)
        return(TotalDrag)
        
    def compute_cost(self, OM):
        # OM is an array of shape (3,timesteps)
        cost = 0
        for i in range(3):
            cost += pow(abs(OM[i]),3.0)
        return cost

    ### Read Efforts files and get drag data
    def retrieve_drag(self):
        D = [ [], [], []]
        for i in range (3):
            with open(self.liftsensor_path + 'Efforts_Cyl' + str(i) + '.txt', 'r') as f:
                next(f) # Skip header
                for line in f:
                    D[i].append(float(line.split()[1])) # select drag column value
        D = np.array(D)
        return(D)

    def retrieve_omega(self):
        OM = [ [], [], []]
        with open(self.liftsensor_path + 'RotationFront1.txt', 'r') as f:
            next(f) # Skip header
            for line in f:
                OM[0] = float(line.split()[3])

        with open(self.liftsensor_path + 'RotationTop1.txt', 'r') as f:
            next(f) # Skip header
            for line in f:
                OM[1] = float(line.split()[3])

        with open(self.liftsensor_path + 'RotationBottom1.txt', 'r') as f:
            next(f) # Skip header
            for line in f:
                OM[2] = float(line.split()[3])

        OM = np.array(OM)
        return(OM)

    ### To write rotations at each environment
    def write_actions(self,actions,ep):
        with open(self.path+'/../actions.txt','a') as file:
            data_str = ' '.join( [str(ep)] + [str(rotation) for rotation in actions] )
            file.write(data_str + '\n')
        return('done writing actions of env '+str(ep)+' in file')

    ### To write rotations at each environment
    def write_actions_alt(self,actions,ep):
        with open('actions_alt.txt','a') as file:
            data_str = ' '.join( [str(ep)] + [str(rotation) for rotation in actions] )
            file.write(data_str + '\n')
        return('done writing actions of env '+str(ep)+' in file')

    ### To write rewards at each environment
    def write_rewards(self,rewards,ep):
        with open(self.path+'/../all_rewards.txt','a') as file:
            data_str = ' '.join( [str(ep)] + [str(val) for val in rewards] )
            file.write(data_str + '\n')
        return('done writing rewards of env '+str(ep)+' in file')



### Tests ###
filepath = os.path.dirname(os.path.abspath(__file__))
pbo = airfoil(filepath)
#actions = np.array([0.00105, 0.0105, 0.105, 0.0105, 0.105, 0.0105])
actions = np.array([
            0.05, 0.08, 0.08, 0.041, 0.08, #camber
            0.03, 0.07, 0.08, 0.08, 0.04,    #thickness
            -0.0                               #rotation
    ])

filepath = os.path.dirname(os.path.abspath(__file__))
pbo = airfoil(filepath)
actions = np.array([
        0.1, 0.12, 0.13, 0.07, 0.03, #camber
        0.05, 0.1, 0.1, 0.1, 0.05,   #thickness
        -0.3                         #rotation
])

action8 = np.array([0.12834850663760586, 0.02500000000000001, 0.1046007149217617, 0.08596053550893273, 0.03032417419335849, 
                    0.12834850663760586, 0.020000000000000007, 0.1046007149217617, 0.11461404734524366, 0.03032417419335849, 
                    -9.448621960899189])
pbo.create_geometry(action8,"object8", plot=True)
pbo.cfd_solve()


#pbo.create_geometry(action8, "object8", plot = True, naca0010 = False)



# TEST7 = 0
# if TEST7:
#     foil = Foil(10, 1.0, 1.0)
#     foil.name = "new_transfo"
#     # points = airfoil.points
#     # print(f"Points: {len(points)}")
#     # print(f"points[0], points[1], points[13]: {points[0]}, {points[1]}, {points[13]}")
#     actions = np.array([
#             0.1, 0.12, 0.13, 0.07, 0.03, #camber
#             0.05, 0.1, 0.1, 0.1, 0.05,    #thickness
#             -0.3                               #rotation
#     ])
#     foil.camber_thickness(actions)
    
#     foil.apply_translation(2.5,2)

#     print(foil.sync())