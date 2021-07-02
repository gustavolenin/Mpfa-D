import numpy as np


class Foum:
    def __init__(
        self,
        mesh_data,
        water_specific_mass,
        oil_SG,
        viscosity_w,
        viscosity_o,
        cfl,
    ):
        # self.mesh_data = mesh_data

        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu
        self.volumes = mesh_data.all_volumes
        self.faces = mesh_data.all_faces

        self.rho_w = water_specific_mass
        self.rho_o = oil_SG * water_specific_mass
        self.viscosity_w = viscosity_w
        self.viscosity_o = viscosity_o

        self.water_sat_i_tag = mesh_data.water_sat_i_tag
        self.oil_sat_i_tag = mesh_data.oil_sat_i_tag
        self.water_sat_bc_tag = mesh_data.water_sat_bc_tag
        self.rel_perm_w_tag = mesh_data.rel_perm_w_tag
        self.rel_perm_o_tag = mesh_data.rel_perm_o_tag
        self.water_sat_tag = mesh_data.water_sat_tag
        self.mobility_w_tag = mesh_data.mobility_w_tag
        self.mobility_o_tag = mesh_data.mobility_o_tag
        self.mobility_tag = mesh_data.mobility_tag
        self.source_tag = mesh_data.source_tag
        self.abs_perm_tag = mesh_data.abs_perm_tag
        self.perm_tag = mesh_data.perm_tag
        # self.face_velocity_tag = mesh_data.face_velocity_tag
        self.cfl = cfl

    def set_relative_perms(self, nw=2, no=2):
        sat_W = self.mb.tag_get_data(self.water_sat_tag, self.volumes)
        sat_W_i = self.mb.tag_get_data(self.water_sat_i_tag, self.volumes)
        sat_O_i = self.mb.tag_get_data(self.oil_sat_i_tag, self.volumes)
        krw = ((sat_W - sat_W_i) / (1 - sat_W - sat_W_i - sat_O_i)) ** nw
        kro = ((1 - sat_W - sat_O_i) / (1 - sat_W_i - sat_O_i)) ** no
        self.mb.tag_set_data(self.rel_perm_w_tag, self.volumes, krw)
        self.mb.tag_set_data(self.rel_perm_o_tag, self.volumes, kro)

    def set_mobility(self):
        krw = self.mb.tag_get_data(self.rel_perm_w_tag, self.volumes)
        kro = self.mb.tag_get_data(self.rel_perm_o_tag, self.volumes)
        mob_w = krw / self.viscosity_w
        mob_o = kro / self.viscosity_o
        mob = mob_w + mob_o
        self.mb.tag_set_data(self.mobility_w_tag, self.volumes, mob_w)
        self.mb.tag_set_data(self.mobility_o_tag, self.volumes, mob_o)
        self.mb.tag_set_data(self.mobility_tag, self.volumes, mob)

    def set_mobility_in_perm(self):
        mobility = self.mb.tag_get_data(self.mobility_tag, self.volumes)
        abs_perm = self.mb.tag_get_data(self.abs_perm_tag, self.volumes)
        full_perm = mobility * abs_perm
        self.mb.tag_set_data(self.perm_tag, self.volumes, full_perm)

    def calc_face_mobility(self, lambda_l, v_l, lambda_r, v_r):
        return (lambda_l * v_l + lambda_r * v_r) / (v_l + v_r)

    def get_delta_t(self):
        pass

    def calc_fractional_flux(self, lambda_w, lambda_o):
        return lambda_w / (lambda_w + lambda_o)

    def get_dirichlet_face_verts(self):
        face_verts = np.asarray(
            [
                self.mtu.get_bridge_adjacencies(face, 2, 0)
                for face in self.dirichlet_faces
            ],
            dtype='uint64'
        )
        I, J, K = [face_verts[:, idx] for idx in face_verts.shape[1]]
        return I, J, K

    def get_face_aux_vectors(self, I, J, K):
        JI = self.mb.get_coords(I) - self.mb.get_coords(J)
        JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
        JI, JK = [vec.reshape([len(self.dirichlet_faces), 3]) for vec in [JI, JK]]
        return JI, JK

    def get_bc_volumes(self):
        left_volumes = np.asarray(
            [
                self.mtu.get_bridge_adjacencies(face, 2, 3)
                for face in self.dirichlet_faces
             ],
            dtype='uint64'
        )
        left_ids = self.mb.tag_get_data(fc.mesh.global_id_tag, l_vols)
        return left_volumes, left_ids

    def set_normal_vector(self, JI, JK, left_volumes):
        volumes_centres = self.mesh_data.mb.tag_get_data(
            self.volume_centre_tag, left_volumes
        ).flatten()
        N_IJK = np.cross(JI, JK) / 2.0
        LJ = (
            self.mb.get_coords(J) - volumes_centres) \
            .reshape([len(self.dirichlet_faces), 3]
        )


    # def init(self):
    #     s_w = np.asarray(
    #         self.mb.tag_get_data(
    #             self.water_sat_tag, self.volumes
    #         )
    #     ).flatten()
    #     sat_W_i = np.asarray(
    #         self.mb.tag_get_data(
    #             self.water_sat_i_tag, self.volumes
    #         )
    #     ).flatten()
    #     sat_O_i = np.asarray(
    #         self.mb.tag_get_data(
    #             self.oil_sat_i_tag, self.volumes
    #         )
    #     ).flatten()
    #
    #     krw, kro = self.get_relative_perms(s_w, sat_W_i, sat_O_i)
    #     self.mb.tag_set_data(
    #         self.rel_perm_w_tag, self.volumes, krw
    #     )
    #     self.mb.tag_set_data(
    #         self.rel_perm_o_tag, self.volumes, kro
    #     )
    #     lambda_face = []
    #     for face in self.faces:
    #         try:
    #             vol_l, vol_r = self.mtu.get_bridge_adjacencies(face, 2, 3)
    #
    #             nodes_l = self.mtu.get_bridge_adjacencies(vol_l, 3, 0)
    #             nodes_l_crds = self.mesh_data.mb.get_coords(nodes_l).reshape(
    #                 [4, 3]
    #             )
    #             vol_l_volume = self.mesh_data.get_tetra_volume(nodes_l_crds)
    #
    #             krw_l = self.mb.tag_get_data(self.rel_perm_w_tag, vol_l)
    #             lambda_W_l = self.calc_mobility(krw_l, self.viscosity_w)
    #             kro_l = self.mb.tag_get_data(self.rel_perm_o_tag, vol_l)
    #             lambda_O_l = self.calc_mobility(kro_l, self.viscosity_o)
    #
    #             nodes_r = self.mtu.get_bridge_adjacencies(vol_r, 3, 0)
    #
    #             nodes_r_crds = self.mesh_data.mb.get_coords(nodes_r).reshape(
    #                 [4, 3]
    #             )
    #             vol_r_volume = self.mesh_data.get_tetra_volume(nodes_r_crds)
    #
    #             krw_r = self.mb.tag_get_data(self.rel_perm_w_tag, vol_r)
    #             lambda_W_r = self.calc_mobility(krw_r, self.viscosity_w)
    #             kro_r = self.mb.tag_get_data(self.rel_perm_o_tag, vol_r)
    #             lambda_O_r = self.calc_mobility(kro_r, self.viscosity_o)
    #
    #             lambda_face_W = self.calc_face_mobility(
    #                 lambda_W_l, vol_l_volume, lambda_W_r, vol_r_volume
    #             )
    #             lambda_face_O = self.calc_face_mobility(
    #                 lambda_O_l, vol_l_volume, lambda_O_r, vol_r_volume
    #             )
    #             lambda_face.append(lambda_face_W + lambda_face_O)
    #
    #         except ValueError:
    #             vol_l = self.mtu.get_bridge_adjacencies(face, 2, 3)
    #             krw_l = self.mb.tag_get_data(self.rel_perm_w_tag, vol_l)[0]
    #             lambda_face_W = self.calc_mobility(krw_l, self.viscosity_w)
    #             kro_l = self.mb.tag_get_data(self.rel_perm_o_tag, vol_l)
    #             lambda_face_O = self.calc_mobility(kro_l, self.viscosity_o)
    #             lambda_face.append(lambda_face_W + lambda_face_O)
    #     lambda_face = np.asarray(lambda_face).flatten()
    #     self.mb.tag_set_data(
    #         self.face_mobility_tag, self.faces, lambda_face,
    #     )
    #
    # def run(self):
    #     raise NotImplementedError
