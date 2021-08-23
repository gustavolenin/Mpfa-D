import os

import pdb

import numpy as np

from solvers.MpfaD import MpfaD3D
from solvers.interpolation.LPEW3 import LPEW3
from solvers.foum import Foum
from preprocessor.mesh_preprocessor import MeshManager


class FlowChannel:

    def __init__(self):
        K_1 = np.array(
            [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
        )
        # mesh_test_conservative
        self.mesh = MeshManager("meshes/mesh_test_conservative.msh", dim=3)
        #self.mesh = MeshManager("meshes/test_mesh_5_vols.h5m", dim=3)
        two_phase_props = {
            "Water_Sat": 0.5, "Water_Sat_i": 0.15, "Oil_Sat_i": 0.1, "Abs Permeability": K_1
        }
        [
            self.mesh.set_media_property(prop, {1: value}, dim_target=3)
            for prop, value in two_phase_props.items()
        ]
        bc_props = {
            "SW_BC": {102: 1.0, 101: 0.2, 201: 0.2},
            "Dirichlet": {102: 1.0, 101: 0.0},
            "Neumann": {201: 0.0}
        }
        [
            self.mesh.set_boundary_condition(
                prop, values, dim_target=2, set_nodes=True
            )
            for prop, values in bc_props.items()
        ]
        self.mesh.get_redefine_centre()
        self.mesh.set_global_id()
        self.mpfad = MpfaD3D(self.mesh)
        water_specific_mass = 1
        oil_SG = 0.8
        viscosity_w = 1
        viscosity_o = 0.8
        cfl = 0.5
        self.foum = Foum(
            self.mesh,
            water_specific_mass,
            oil_SG,
            viscosity_w,
            viscosity_o,
            cfl
        )

    def impes(self):
        # t = self.foum.delta_t
        # looper = 0
        # tmax = .1
        # while t < tmax:
        # print(looper)
        self.foum.set_relative_perms()
        self.foum.set_mobility()
        self.foum.set_mobility_in_perm()
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        self.foum.tag_velocity()
        self.foum.calculate_face_water_sat()
        self.foum.calculate_water_flux()

        # t += self.foum.delta_t
        # print("sim time elapsed: ", t)
        # self.foum.update_sat()
        # looper += 1
        # filename = os.path.basename(__file__).replace(".py", "")
        # self.mpfad.record_data(f"{filename}_{looper}.vtk")
            # if looper > 100:
            #     break
            # calcular velocidade
            # calcula cfl => calcula delta_t
            # variacao maxima de saturacao dado pelo user, delta t passar, volta pro passo minimo
            # atualiza saturacao da fase:
            # - computa os fluxos da fase agua
            # - realiza balaço de massa por volume
            # t += delta_t

            # na maioria dos casos, se usa a velocidade total e fluxo fracionario para o calculo da saturacao

            # calculo da saturacao nos volumes de contorno
