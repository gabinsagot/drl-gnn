import numpy as np
import geometry.mesh.Foil 
import gmsh 

def get_bodyfitted_domain(object : Foil, domain : list[np.ndarray]):

    