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
        # self.mesh = MeshManager("meshes/mesh_test_conservative.msh", dim=3)
        # self.mesh = MeshManager("meshes/test_mesh_5_vols.h5m", dim=3)
        self.mesh = MeshManager("backup_mesh_files/mytestmesh.msh", dim=3)
        two_phase_props = {
            "Water_Sat": 0.20, "Water_Sat_i": 0.15, "Oil_Sat_i": 0.10, "Abs Permeability": K_1
        }
        [
            mesh.set_media_property(prop, {1: value}, dim_target=3)
            for prop, value in two_phase_props.items()
        ]
        bc_props = {
            "Neumann": {201: 0.0}
        }
        [
            self.mesh.set_boundary_condition(
                prop, values, dim_target=2, set_nodes=True
            )
            for prop, values in bc_props.items()
        ]
        wells = {
            "Injection": {
                "SW_BC": {101: 1.0},
                "Source term": {101: 1.0}
            },
            "Production": {
                "Dirichlet": {102: 0.}
            }
        }
        self.elements = {}
        for key, values in wells.items():
            if key not in ["Injection", "Production"]:
                raise "You should either have a Production or Injection well here."
            physical_sets = self.mesh.physical_sets
            for _set in physical_sets:
                physical_group = self.mesh.mb.tag_get_data(
                    self.mesh.physical_tag, _set, flat=True
                )
                verts = self.mesh.mb.get_entities_by_dimension(
                    _set, 0
                )
                if verts:
                    volumes = self.mesh.mtu.get_bridge_adjacencies(verts[0], 0, 3)
                    well_vicininy = max([len(self.mesh.mtu.get_bridge_adjacencies(volume, 2, 3)) for volume in volumes])
                    well = [volume for volume in volumes if len(self.mesh.mtu.get_bridge_adjacencies(volume, 2, 3)) == well_vicininy][0]
                    for information_name, properties_map in values.items():
                        information_tag = self.mesh.mb.tag_get_handle(information_name)
                        for tag, prop in properties_map.items():
                            if tag == physical_group:
                                if key not in self.elements.keys():
                                    self.elements[key] = set()
                                self.elements[key].add(well)
                                # [self.elements[key].add(volume) for volume in np.asarray(volumes)]
                                self.mesh.mb.tag_set_data(information_tag, well, prop)
                                if information_name == "Source term":
                                    if information_name == "Dirichlet":
                                        self.mesh.dirichlet_volumes = self.mesh.dirichlet_volumes | set(
                                            well
                                        )

        #
        #
        #
        #
        #
        # for key, values in wells.items():
        #     for a_set in mesh.physical_sets:
        #         physical_group = mesh.mb.tag_get_data(
        #             mesh.physical_tag, a_set, flat = True
        #         )
        #         if key == "SW_BC" and physical_group == 101:
        #             elm = self.mesh.mb.get_entities_by_dimension(
        #                 a_set, 0
        #             )
        #             if elm:
        #                 vol = self.mesh.mtu.get_bridge_adjacencies(elm[0], 0, 3)
        #                 self.mesh.mb.tag_set_data(self.dirichlet_tag, vol, values[101])

        # self.mesh.get_redefine_centre()
        # self.mesh.set_global_id()
        # self.mpfad = MpfaD3D(self.mesh)
        # water_specific_mass = 1
        # oil_SG = 0.8
        # viscosity_w = 1
        # viscosity_o = 0.8
        # cfl = 0.2
        # self.foum = Foum(
        #     self.mesh,
        #     water_specific_mass,
        #     oil_SG,
        #     viscosity_w,
        #     viscosity_o,
        #     cfl
        # )

    def impes(self):
        # t = 0.0001  # self.foum.delta_t
        # looper = 0
        # tmax = .1
        # while t < tmax:
        self.foum.set_relative_perms()
        self.foum.set_mobility()
        self.foum.set_mobility_in_perm()
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        self.foum.tag_velocity()
        self.foum.calculate_face_water_sat()
        self.foum.calculate_water_flux()
        self.foum.update_sat()
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
