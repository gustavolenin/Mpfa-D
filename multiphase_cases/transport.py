import numpy as np

from preprocessor.mesh_preprocessor import MeshManager


K_1 = np.array(
    [1.0, 0.0, 0.0,
     0.0, 1.0, 0.0,
     0.0, 0.0, 1.0]
)
mesh = MeshManager("meshes/mesh_two_phase.msh", dim=3)
two_phase_props = {
    "Water_Sat": 0.20,
    "Water_Sat_i": 0.15,
    "Oil_Sat_i": 0.10,
    "Abs Permeability": K_1,
}
[
    set_media_property(prop, {1: value}, dim_target=3)
    for prop, value in two_phase_props.items()
]
bc_props = {"Neumann": {201: 0.0}, "SW_BC": {101: 1.0, 102: 0.0}}
[
    mesh.set_boundary_condition(prop, values, dim_target=2, set_nodes=True)
    for prop, values in bc_props.items()
]


def calculate_flux():
    pass

def update_delta_t():
    pass

def update_sat():
    pass
