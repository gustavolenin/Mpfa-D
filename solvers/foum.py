import numpy as np

from pymoab import types

import solvers.helpers.geometric as geo


def get_tetra_volume(tet_nodes):
    vect_1 = tet_nodes[1] - tet_nodes[0]
    vect_2 = tet_nodes[2] - tet_nodes[0]
    vect_3 = tet_nodes[3] - tet_nodes[0]
    vol_eval = abs(np.dot(np.cross(vect_1, vect_2), vect_3)) / 6.0
    return vol_eval


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
        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces
        self.global_id_tag = mesh_data.global_id_tag
        self.volume_centre_tag = mesh_data.volume_centre_tag
        self.dirichlet_tag = mesh_data.dirichlet_tag
        self.neumann_tag = mesh_data.neumann_tag
        self.all_nodes = mesh_data.all_nodes
        self.volume_tag = mesh_data.volume_tag
        self.face_area_tag = mesh_data.face_area_tag

        self.rho_w = water_specific_mass
        self.rho_o = oil_SG * water_specific_mass
        self.viscosity_w = viscosity_w
        self.viscosity_o = viscosity_o

        self.water_sat_i_tag = mesh_data.water_sat_i_tag
        self.oil_sat_i_tag = mesh_data.oil_sat_i_tag
        self.water_sat_bc_tag = mesh_data.water_sat_bc_tag
        self.face_water_sat_tag = mesh_data.face_water_sat_tag
        self.water_volume_flux = mesh_data.water_volume_flux
        self.water_volume_flux2 = mesh_data.water_volume_flux2
        self.rel_perm_w_tag = mesh_data.rel_perm_w_tag
        self.rel_perm_o_tag = mesh_data.rel_perm_o_tag
        self.water_sat_tag = mesh_data.water_sat_tag
        self.mobility_w_tag = mesh_data.mobility_w_tag
        self.mobility_o_tag = mesh_data.mobility_o_tag
        self.mobility_tag = mesh_data.mobility_tag
        self.source_tag = mesh_data.source_tag
        self.abs_perm_tag = mesh_data.abs_perm_tag
        self.perm_tag = mesh_data.perm_tag
        self.pressure_tag = mesh_data.pressure_tag
        self.velocity_tag = mesh_data.velocity_tag
        self.cfl = cfl

        self.nodes_ws = mesh_data.nodes_ws
        self.nodes_nts = mesh_data.nodes_nts
        self.all_volumes = mesh_data.all_volumes
        self.all_faces = mesh_data.all_faces
        mesh_data.calculate_volume()
        mesh_data.calculate_face_areas()

        self.delta_t = 0
        self.delta_sat_max = 0.4

    def set_relative_perms(self, nw=2, no=2):
        sat_W = self.mb.tag_get_data(self.water_sat_tag, self.volumes)
        sat_W_i = self.mb.tag_get_data(self.water_sat_i_tag, self.volumes)
        sat_O_i = self.mb.tag_get_data(self.oil_sat_i_tag, self.volumes)
        krw = ((sat_W - sat_W_i) / (1 - sat_W_i - sat_O_i)) ** nw
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
        phis = np.repeat(1, len(self.all_volumes))
        volumes = self.mb.tag_get_data(self.volume_tag, self.all_volumes)
        velocities = abs(
            self.mb.tag_get_data(self.water_volume_flux, self.volumes)
        )
        delta_t = np.asarray(
            [
                (self.cfl * (volume * phi)) / velocity
                for volume, phi, velocity in zip(volumes, phis, velocities) if
                velocity > 0
            ]
        ).min()
        return delta_t

    def calculate_face_water_sat(self):
        face_sats = []
        for face in self.faces:
            if face in self.dirichlet_faces:
                sat = self.mb.tag_get_data(self.water_sat_bc_tag, face)
            elif face in self.neumann_faces:
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                sat = self.mb.tag_get_data(self.water_sat_tag, volume)
            else:
                adj_vols = self.mtu.get_bridge_adjacencies(face, 2, 3)
                adj_vols_p = self.mb.tag_get_data(self.pressure_tag, adj_vols)
                mapper = {
                    adj_vol: adj_vol_p for adj_vol, adj_vol_p in zip(adj_vols, adj_vols_p)
                }
                p_max = max(adj_vols_p)
                volume = adj_vols[0]
                for key, value in mapper.items():
                    if value >= p_max:
                        volume = key
                sat = self.mb.tag_get_data(self.water_sat_tag, volume)
            face_sats.append(sat)
        face_sats = np.asarray(face_sats).flatten()
        self.mb.tag_set_data(self.face_water_sat_tag, self.all_faces, face_sats)

    def calculate_water_flux(self):
        volume_fluxes = []
        _volume_fluxes = []
        for volume in self.volumes:
            adj_faces = self.mtu.get_bridge_adjacencies(volume, 2, 2)
            face_areas = self.mb.tag_get_data(self.face_area_tag, adj_faces)
            face_velocities = self.mb.tag_get_data(self.velocity_tag, adj_faces)
            face_sats = self.mb.tag_get_data(self.face_water_sat_tag, adj_faces)
            volume_flux = sum(face_sats * face_velocities * face_areas)
            # print(volume_flux, len(adj_faces))
            if abs(volume_flux) > 1e-10:
                _volume_flux = 1
            else:
                _volume_flux = 0
            volume_fluxes.append(volume_flux)
            _volume_fluxes.append(_volume_flux)

        print(volume_fluxes)
        volume_fluxes = np.asarray(volume_fluxes).flatten()
        # self.mb.tag_set_data(self.water_volume_flux2, self.volumes, _volume_fluxes)
        self.mb.tag_set_data(self.water_volume_flux, self.volumes, volume_fluxes)

    def get_delta_t_for_delta_sat_max(self):
        phis = np.repeat(1, len(self.all_volumes))
        volumes = self.mb.tag_get_data(self.volume_tag, self.all_volumes)
        water_fluxes = abs(
            self.mb.tag_get_data(self.water_volume_flux, self.volumes)
        )
        delta_t = self.delta_sat_max / \
            (water_fluxes / (volumes * phis)).max()
        return delta_t

    def use_delta_t_min(self):
        delta_t1 = self.get_delta_t()
        delta_t2 = self.get_delta_t_for_delta_sat_max()
        self.delta_t = min(delta_t1, delta_t2)

    def calculate_sat(self):
        phis = np.repeat(1, len(self.all_volumes))
        volumes = self.mb.tag_get_data(self.volume_tag, self.all_volumes).flatten()
        water_fluxes = self.mb.tag_get_data(
            self.water_volume_flux, self.volumes
        ).flatten()
        sats = self.mb.tag_get_data(self.water_sat_tag, self.volumes).flatten()
        sats += water_fluxes / (volumes * phis) * self.delta_t
        return sats

    def update_sat(self):
        self.use_delta_t_min()
        sats = self.mb.tag_get_data(self.water_sat_tag, self.volumes)
        water_sat_min = self.mb.tag_get_data(self.water_sat_i_tag, self.volumes).max()
        water_sat_max = 1 - self.mb.tag_get_data(self.oil_sat_i_tag, self.volumes).min()
        new_sats = self.calculate_sat()
        max_delta = abs(new_sats - sats).max() > self.delta_sat_max
        irr_boundary_sat = any(new_sats > water_sat_max) | any(new_sats < water_sat_min)
        counts = 0
        while any([max_delta, irr_boundary_sat]):
            self.delta_t /= 2
            new_sats = self.calculate_sat()
            max_delta = abs(new_sats - sats).max() > self.delta_sat_max
            irr_boundary_sat = any(new_sats > water_sat_max) | any(new_sats < water_sat_min)
            counts += 1
            print(self.delta_t)
            if counts > 100:
                break
        print(new_sats.flatten())
        self.mb.tag_set_data(self.water_sat_tag, self.volumes, new_sats)


        # Comparar novas saturacoes com as anteriores e garantir qu o delta de
        # sat no tempo nao viola o max delta sat
        # se violar, reduz o passo de tempo
        # garantir que saturação residual não está sendo extrapolada
        # caso extrapole, reduz o passo de tempo.


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
        I, J, K = [face_verts[:, idx] for idx in range(face_verts.shape[1])]
        return I, J, K

    def get_face_aux_vectors(self, I, J, K):
        JI = self.mb.get_coords(I) - self.mb.get_coords(J)
        JK = self.mb.get_coords(K) - self.mb.get_coords(J)
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
        left_ids = self.mb.tag_get_data(self.global_id_tag, left_volumes)
        return left_volumes, left_ids

    def rearrange_face_entities(self, I, J, K, left_volumes):
        JI, JK = self.get_face_aux_vectors(I, J, K)
        volumes_centres = self.mb.tag_get_data(
            self.volume_centre_tag, left_volumes
        ).flatten()
        N_IJK = np.cross(JI, JK) / 2.0
        LJ = (
            self.mb.get_coords(J) - volumes_centres) \
            .reshape([len(self.dirichlet_faces), 3]
        )
        _test = np.sum(LJ * N_IJK, axis=1)
        is_reverse = np.where(_test < 0)[0]
        I[is_reverse], K[is_reverse] = K[is_reverse], I[is_reverse]
        JI, JK = self.get_face_aux_vectors(I, J, K)
        N_IJK = np.cross(JI, JK) / 2.0
        return JI, JK, N_IJK, I, J, K

    def calc_geometric_props(self, N_IJK, vol_centre):
        face_area = np.sqrt(sum(N_IJK * N_IJK), axis=1)
        height = geo.get_height(N_IJK, vol_centre)
        return face_area, height

    def tag_velocity(self):
        self.node_pressure_tag = self.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )
        p_verts = []
        for node in self.all_nodes:
            try:
                p_vert = self.mb.tag_get_data(
                    self.dirichlet_tag, node
                )
                p_verts.append(p_vert[0])
            except Exception:
                p_vert = 0.0
                p_tag = self.pressure_tag
                nd_weights = self.nodes_ws[node]
                for volume, wt in nd_weights.items():
                    p_vol = self.mb.tag_get_data(p_tag, volume)
                    p_vert += p_vol * wt
                p_verts.append(p_vert[0])
        self.mb.tag_set_data(
            self.node_pressure_tag, self.all_nodes, p_verts
        )
        vols = []
        for a_volume in self.all_volumes:
            vol_faces = self.mtu.get_bridge_adjacencies(a_volume, 2, 2)
            vol_nodes = self.mtu.get_bridge_adjacencies(a_volume, 0, 0)
            vol_crds = self.mb.get_coords(vol_nodes)
            vol_crds = np.reshape(vol_crds, ([4, 3]))
            vol_volume = get_tetra_volume(vol_crds)
            vols.append(vol_volume)
            I, J, K = self.mtu.get_bridge_adjacencies(vol_faces[0], 2, 0)
            L = list(
                set(vol_nodes).difference(
                    set(
                        self.mtu.get_bridge_adjacencies(
                            vol_faces[0], 2, 0
                        )
                    )
                )
            )
            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
            LJ = self.mb.get_coords([J]) - self.mb.get_coords(L)
            N_IJK = np.cross(JI, JK) / 2.0

            test = np.dot(LJ, N_IJK)
            if test < 0.0:
                I, K = K, I
                JI = self.mb.get_coords([I]) - self.mb.get_coords(
                    [J]
                )
                JK = self.mb.get_coords([K]) - self.mb.get_coords(
                    [J]
                )
                N_IJK = np.cross(JI, JK) / 2.0

            tan_JI = np.cross(N_IJK, JI)
            tan_JK = np.cross(N_IJK, JK)
            face_area = np.sqrt(np.dot(N_IJK, N_IJK))

            h_L = geo.get_height(N_IJK, LJ)

            p_I = self.mb.tag_get_data(self.node_pressure_tag, I)
            p_J = self.mb.tag_get_data(self.node_pressure_tag, J)
            p_K = self.mb.tag_get_data(self.node_pressure_tag, K)
            p_L = self.mb.tag_get_data(self.node_pressure_tag, L)
            grad_normal = -2 * (p_J - p_L) * N_IJK

            grad_cross_I = (p_J - p_I) * (
                (np.dot(tan_JK, LJ) / face_area ** 2) * N_IJK
                - (h_L / (face_area)) * tan_JK
            )
            grad_cross_K = (p_K - p_J) * (
                (np.dot(tan_JI, LJ) / face_area ** 2) * N_IJK
                - (h_L / (face_area)) * tan_JI
            )
            grad_p = -(1 / (6 * vol_volume)) * (
                grad_normal + grad_cross_I + grad_cross_K
            )
            vol_centroid = np.asarray(
                self.mb.tag_get_data(
                    self.volume_centre_tag, a_volume
                )[0]
            )
            vol_perm = self.mb.tag_get_data(
                self.perm_tag, a_volume
            ).reshape([3, 3])
            x, y, z = vol_centroid
            vels = []
            for face in vol_faces:
                face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
                face_nodes_crds = self.mb.get_coords(face_nodes)
                area_vect = geo._area_vector(
                    face_nodes_crds.reshape([3, 3]), vol_centroid
                )[0]
                unit_area_vec = area_vect / np.sqrt(
                    np.dot(area_vect, area_vect)
                )
                k_grad_p = np.dot(vol_perm, grad_p[0])
                vel = -np.dot(k_grad_p, unit_area_vec)
                vels.append(vel)
            self.mb.tag_set_data(self.velocity_tag, vol_faces, vels)





# %load_ext autoreload
# %autoreload 2
# import numpy as np
# from multiphase_cases.flow_channel import FlowChannel
# fc = FlowChannel()
# fc.impes()
# self = fc.foum
# self.update_sat()
# left_volumes, left_volumes_ids = fc.foum.get_bc_volumes()
# I, J, K = fc.foum.get_dirichlet_face_verts()
# JI, JK = fc.foum.get_face_aux_vectors(I, J, K)
