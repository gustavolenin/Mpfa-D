import numpy as np
# from solvers.MpfaD import MpfaD3D
from solvers.nMpfaD import MpfaD3D
# from mpfad.interpolation.LPEW3 import LPEW3
from preprocessor.mesh_preprocessor import MeshManager


class ObliqueDrain:
    def __init__(self, filename):
        self.mesh = MeshManager(filename, dim=3)
        self.mesh.set_boundary_condition(
            "Dirichlet", {101: None}, dim_target=2, set_nodes=True
        )
        self.mesh.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )

        bed_perm = [
            0.965384615384615,
            0.173076923076923,
            0.0,
            0.173076923076923,
            0.134615384615385,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        drain_perm = [
            96.538461538461530,
            17.307692307692307,
            0.0,
            17.307692307692307,
            13.461538461538462,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        self.mesh.set_media_property(
            "Permeability",
            {1: bed_perm, 2: drain_perm, 3: bed_perm},
            dim_target=3,
        )
        self.mesh.set_global_id()
        self.mesh.get_redefine_centre()
        self.mpfad = MpfaD3D(self.mesh)

    def norms_calculator(self, error_vector, volumes_vector, u_vector):
        error_vector = np.array(error_vector)
        volumes_vector = np.array(volumes_vector)
        u_vector = np.array(u_vector)
        l2_norm = np.dot(error_vector, error_vector) ** (1 / 2)
        l2_volume_norm = np.dot(error_vector ** 2, volumes_vector) ** (1 / 2)
        erl2 = (
            np.dot(error_vector ** 2, volumes_vector)
            / np.dot(u_vector ** 2, volumes_vector)
        ) ** (1 / 2)
        avr_error = l2_norm / len(volumes_vector)
        max_error = max(error_vector)
        min_error = min(error_vector)
        results = [
            l2_norm,
            l2_volume_norm,
            erl2,
            avr_error,
            max_error,
            min_error,
        ]
        return results

    def _obliqueDrain(self, x, y):
        return -x - 0.2 * y

    def runCase(self, interpolation_method, log_name):
        for node in self.mesh.get_boundary_nodes():
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self._obliqueDrain(x, y)
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        vols = []
        for volume in volumes:
            x, y, _ = self.mesh.mb.tag_get_data(
                self.mesh.volume_centre_tag, volume
            )[0]
            vol_nodes = self.mesh.mb.get_adjacencies(volume, 0)
            vol_nodes_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh.get_tetra_volume(vol_nodes_crds)
            vols.append(tetra_vol)
        self.mpfad.run_solver(interpolation_method(self.mesh).interpolate)