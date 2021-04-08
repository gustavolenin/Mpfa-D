"""This is the begin."""
import pdb
import matplotlib.pyplot as plt
from pymoab import types
from PyTrilinos import Epetra, AztecOO, Amesos
import solvers.helpers.geometric as geo
import numpy as np
import time

from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve


class MpfaD3D:
    """Implement the MPFAD method."""

    def __init__(self, mesh_data, x=None, mobility=None):
        """Init class."""
        self.mesh_data = mesh_data
        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu

        self.comm = Epetra.PyComm()

        self.dirichlet_tag = mesh_data.dirichlet_tag
        self.neumann_tag = mesh_data.neumann_tag
        self.perm_tag = mesh_data.perm_tag
        self.source_tag = mesh_data.source_tag
        self.global_id_tag = mesh_data.global_id_tag
        self.volume_centre_tag = mesh_data.volume_centre_tag
        self.pressure_tag = mesh_data.pressure_tag
        self.node_pressure_tag = mesh_data.node_pressure_tag

        self.flux_info_tag = self.mb.tag_get_handle(
            "flux info", 7, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.normal_tag = self.mb.tag_get_handle(
            "Normal", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.dirichlet_nodes = set(
            self.mb.get_entities_by_type_and_tag(
                0, types.MBVERTEX, self.dirichlet_tag, np.array((None,))
            )
        )

        self.neumann_nodes = set(
            self.mb.get_entities_by_type_and_tag(
                0, types.MBVERTEX, self.neumann_tag, np.array((None,))
            )
        )
        self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes

        boundary_nodes = self.dirichlet_nodes | self.neumann_nodes
        self.intern_nodes = set(self.mesh_data.all_nodes) - boundary_nodes

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces
        self.intern_faces = mesh_data.intern_faces()
        # self.intern_faces = set(mesh_data.all_faces).difference(
        #     self.dirichlet_faces | self.neumann_faces
        # )
        self.volumes = self.mesh_data.all_volumes

        std_map = Epetra.Map(len(self.volumes), 0, self.comm)
        self.T = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)
        self.Q = Epetra.Vector(std_map)
        if x is None:
            self.x = Epetra.Vector(std_map)
        else:
            self.x = x
        self.aux_q = {vol: [] for vol in range(len(self.volumes))}
        # self.T = lil_matrix((len(self.volumes), len(self.volumes)),
        #                     dtype=np.float)
        # self.Q = lil_matrix((len(self.volumes), 1), dtype=np.float)

    def record_data(self, file_name):
        """Record data to file."""
        volumes = self.mb.get_entities_by_dimension(0, 3)
        # faces = self.mb.get_entities_by_dimension(0, 2)
        ms = self.mb.create_meshset()
        self.mb.add_entities(ms, volumes)
        # self.mb.add_entities(ms, faces)
        self.mb.write_file(file_name, [ms])

    def get_boundary_node_pressure(self, node):
        """Return pressure at the boundary nodes of the mesh."""
        pressure = self.mesh_data.mb.tag_get_data(self.dirichlet_tag, node)[0]
        return pressure

    def vmv_multiply(self, normal_vector, tensor, CD):
        """Return a vector-matrix-vector multiplication."""
        vmv = np.dot(np.dot(normal_vector, tensor), CD) / np.dot(
            normal_vector, normal_vector
        )
        return vmv

    def get_cross_diffusion_term(
        self, tan, vec, S, h1, Kn1, Kt1, h2=0, Kt2=0, Kn2=0, boundary=False
    ):
        """Return a cross diffusion multiplication term."""
        if not boundary:
            mesh_anisotropy_term = np.dot(tan, vec) / (S ** 2)
            physical_anisotropy_term = -(
                (1 / S) * (h1 * (Kt1 / Kn1) + h2 * (Kt2 / Kn2))
            )
            cross_diffusion_term = (
                mesh_anisotropy_term + physical_anisotropy_term
            )
            return cross_diffusion_term
        if boundary:
            dot_term = np.dot(-tan, vec) * Kn1
            cdf_term = h1 * S * Kt1
            b_cross_difusion_term = (dot_term + cdf_term) / (2 * h1 * S)
            return b_cross_difusion_term

    # @celery.task
    def get_nodes_weights(self, method):
        """Return the node weights."""
        self.nodes_ws = {}
        self.nodes_nts = {}
        # This is the limiting part of the interpoation method. The Dict
        for node in self.intern_nodes:
            self.nodes_ws[node] = method(node)

        for node in self.neumann_nodes:
            self.nodes_ws[node] = method(node, neumann=True)
            self.nodes_nts[node] = self.nodes_ws[node].pop(node)

    def _node_treatment(self, node, id_left, id_right, K_eq, D_JK=0, D_JI=0.0):
        """Add flux term from nodes RHS."""
        RHS = 0.5 * K_eq * (D_JK + D_JI)
        if node in self.dirichlet_nodes:
            pressure = self.get_boundary_node_pressure(node)
            self.Q[id_left] += RHS * pressure
            self.Q[id_right] += -RHS * pressure
            self.aux_q[int(id_left)].append({node: RHS * pressure})
            self.aux_q[int(id_right)].append({node: -RHS * pressure})
            # self.Q[id_left[0], 0] += RHS * pressure
            # self.Q[id_right[0], 0] += - RHS * pressure

        if node in self.intern_nodes:
            for volume, weight in self.nodes_ws[node].items():
                self.ids.append([id_left, id_right])
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                self.v_ids.append([v_id, v_id])
                self.ivalues.append([-RHS * weight, RHS * weight])

        if node in self.neumann_nodes:
            neu_term = self.nodes_nts[node]
            self.Q[id_right] += -RHS * neu_term
            self.Q[id_left] += RHS * neu_term
            self.aux_q[int(id_left)].append({node: RHS * neu_term})
            self.aux_q[int(id_right)].append({node: -RHS * neu_term})
            # self.Q[id_right, 0] += - RHS * neu_term
            # self.Q[id_left, 0] += RHS * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                self.ids.append([id_left, id_right])
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                self.v_ids.append([v_id, v_id])
                self.ivalues.append([-RHS * weight_N, RHS * weight_N])

    def get_global_rows(self):
        aux_list = []
        epetra_global_rows = np.asarray(
            [
                self.T.ExtractGlobalRowCopy(row)[1]
                for row in range(len(self.volumes))]
        )
        for row, arrs in enumerate(epetra_global_rows):
            for val in arrs:
                aux_list.append([row, val])
        rows = np.asarray(aux_list)[:, 0]
        cols = np.asarray(aux_list)[:, 1]
        return rows, cols

    @staticmethod
    def create_sp_matrix(shape, dtype=np.float64):
        return lil_matrix((shape, shape), dtype=dtype)

    def copy_mat(self, rows, cols, shape=None, dtype=np.float64):
        A = self.create_sp_matrix(shape, dtype)
        A[rows, cols] = [self.T[row][col] for row, col in zip(rows, cols)]
        return A

    def solve_original_problem(self, mat: lil_matrix, q: np.array):

        if any([mat.shape[0] < len(self.volumes), len(q) < len(self.volumes)]):
            raise (
                "not solvable. matrix length must be at leat the size of the original problem"
            )

        x = spsolve(
            mat[:len(self.volumes), :len(self.volumes)].tocsc(), q[:len(self.volumes)]
        )
        return x

    def tag_verts_pressure(self):
        print("Will tag vertices pressure")
        p_verts = []
        for node in self.mesh_data.all_nodes:
            try:
                p_vert = self.mb.tag_get_data(self.dirichlet_tag, node)
                p_verts.append(p_vert[0])
            except Exception:
                p_vert = 0.0
                p_tag = self.pressure_tag
                nd_weights = self.nodes_ws[node]
                for volume, wt in nd_weights.items():
                    p_vol = self.mb.tag_get_data(p_tag, volume)
                    p_vert += p_vol * wt
                p_verts.append(p_vert)
        self.mb.tag_set_data(
            self.node_pressure_tag, self.mesh_data.all_nodes, p_verts
        )
        print("Done tagging vertices pressure!!")

    def compute_mi(self, a_volume):
        vol_faces = self.mtu.get_bridge_adjacencies(a_volume, 2, 2)
        vol_nodes = self.mtu.get_bridge_adjacencies(a_volume, 0, 0)
        vol_crds = self.mb.get_coords(vol_nodes)
        vol_crds = np.reshape(vol_crds, ([4, 3]))
        vol_volume = self.mesh_data.get_tetra_volume(vol_crds)
        vol_centre = self.mb.get_coords(a_volume)
        mi = []
        for face in vol_faces:
            I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)
            L = list(set(vol_nodes).difference(
                set(self.mtu.get_bridge_adjacencies(face, 2, 0)))
            )
            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
            LJ = self.mb.get_coords([J]) - self.mb.get_coords(L)
            N_IJK = np.cross(JI, JK) / 2.0
            test = np.dot(LJ, N_IJK)
            if test < 0.0:
                I, K = K, I
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords(
                    [J]
                )
                N_IJK = np.cross(JI, JK) / 2.0
            adj_vol = set(self.mtu.get_bridge_adjacencies(face, 2, 3))\
                .difference(set([a_volume, ]))
            if not adj_vol:
                continue
            adj_vol_centre = self.mb.get_coords(adj_vol)
            r = vol_centre - adj_vol_centre
            dot_product = np.dot(N_IJK, r)
            mi.append(abs(dot_product))
        mi = sum(mi) / (vol_volume)
        return mi

    def compute_all_mi(self):
        self.mis = {volume: self.compute_mi(volume) for volume in self.volumes}

    def get_local_min_max(self, volume):
         volume_pressure = self.mb.tag_get_data(self.pressure_tag, volume).flatten()
         adj_volumes = self.mtu.get_bridge_adjacencies(volume, 2, 3)
         volume_vertices = self.mtu.get_bridge_adjacencies(volume, 0, 0)
         adj_volumes_pressure = self.mb.tag_get_data(self.pressure_tag, adj_volumes)
         adj_volumes_verts = [
             self.mtu.get_bridge_adjacencies(adj_vol, 0, 0)
             for adj_vol in adj_volumes
         ]
         all_volumes_verts = set(
             np.asarray(adj_volumes_verts).flatten()
         ).union(set(np.asarray(volume_vertices)))
         all_vertices_in_patch_pressure = self.mb.tag_get_data(
             self.mesh_data.node_pressure_tag,
             np.asarray(list(all_volumes_verts), dtype=np.uint64)
         ).flatten()
         surrounding_pressure_patch = all_vertices_in_patch_pressure\
                                          .flatten().tolist()\
                                      + adj_volumes_pressure\
                                          .flatten().tolist()
         p_max_local = max(surrounding_pressure_patch)
         p_min_local = min(surrounding_pressure_patch)
#         p_max_local = max(
#             0, volume_pressure + max(
#                 [adj_p - volume_pressure for adj_p in surrounding_pressure_patch]
#             )
#         )
#         p_min_local = min(
#             0, volume_pressure + min(
#                 [adj_p - volume_pressure for adj_p in surrounding_pressure_patch]
#             )
#         )
# =============================================================================
#         try:
#             volumes_in_patch = [
#                 *self.mtu.get_bridge_adjacencies(volume, 0, 3), *volume
#             ]
#         except:
#             volumes_in_patch = [
#                 *self.mtu.get_bridge_adjacencies(volume, 0, 3), volume
#             ]
#         vols_pressure = self.mb.tag_get_data(
#             self.pressure_tag, volumes_in_patch
#         )
#         p_max_local = max(vols_pressure)
#         p_min_local = min(vols_pressure)
# =============================================================================
         return p_max_local, p_min_local

    def compute_slip_fact(self, face):
        if face in self.dirichlet_faces | self.neumann_faces:
            volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
            _u = self.mb.tag_get_data(self.pressure_tag, volume)
            max_v, min_v = self.get_local_min_max(volume)
            verts = self.mtu.get_bridge_adjacencies(face, 2, 0)
            avg_verts_pressure = self.mb.tag_get_data(
                self.mesh_data.node_pressure_tag, verts
            ).mean()
            lp = _u - avg_verts_pressure
            volume = np.asarray(volume)[0]
            if _u > avg_verts_pressure:
                # lfs = (max_v - _u)
                lfs = 2 * self.mis[volume] * (max_v - _u)
                _s = min(lfs, lp)
            else:
                # rts = (min_v - _u)
                rts = 2 * self.mis[volume] * (min_v - _u)
                _s = max(lp, rts)
                
        else:
            volume, adj = self.mtu.get_bridge_adjacencies(face, 2, 3)
            _u = self.mb.tag_get_data(self.pressure_tag, volume)
            max_v, min_v = self.get_local_min_max(volume)
            max_v_adj, min_v_adj = self.get_local_min_max(adj)
            u_adj = self.mb.tag_get_data(self.pressure_tag, adj)
            lp = _u - u_adj
            if _u > u_adj:
                lfs = 2 * self.mis[volume] * (max_v - _u)
                rts = 2 * self.mis[adj] * (u_adj - min_v_adj)
                # lfs = (max_v - _u)
                # rts = (u_adj - min_v_adj)
                _s = min(lfs, lp, rts)
                slip_factor = (_s + 1e-20) / (lp + 1e-20)
            else:
                lfs = 2 * self.mis[volume] * (min_v - _u)
                rts = 2 * self.mis[adj] * (u_adj - max_v)
                # lfs = (min_v - _u)
                # rts = (u_adj - max_v)
                lp = -lp
                _s = max(lfs, lp, rts)
                slip_factor = (_s + 1e-20) / (lp + 1e-20)
        slip_factor = (_s + 1e-20) / (lp + 1e-20)
        if abs(slip_factor) < 1e-15:
            slip_factor = 0
        return slip_factor

    def run_solver(self, interpolation_method):
        """Run solver."""
        self.interpolation_method = interpolation_method
        t0 = time.time()
        n_vertex = len(set(self.mesh_data.all_nodes) - self.dirichlet_nodes)
        print("interpolation runing...")
        self.get_nodes_weights(interpolation_method)
        print(
            "done interpolation...",
            "took {0} seconds to interpolate over {1} verts".format(
                time.time() - t0, n_vertex
            ),
        )
        print("filling the transmissibility matrix...")
        begin = time.time()

        try:
            for volume in self.volumes:
                volume_id = self.mb.tag_get_data(self.global_id_tag, volume)[
                    0
                ][0]
                RHS = self.mb.tag_get_data(self.source_tag, volume)[0][0]
                self.Q[volume_id] += RHS
                # self.aux_q[volume_id].append({node: -RHS * neu_term})
                # self.Q[volume_id, 0] += RHS
        except Exception:
            pass

        for face in self.neumann_faces:
            face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
            volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
            volume = np.asarray(volume, dtype="uint64")
            id_volume = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
            face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
            node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
            face_area = geo._area_vector(node_crds, norma=True)
            RHS = face_flow * face_area
            self.Q[id_volume] += -RHS
            self.aux_q[id_volume].append({node: RHS for node in face_nodes})
            # self.Q[id_volume, 0] += - RHS

        id_volumes = []
        all_LHS = []
        for face in self.dirichlet_faces:
            # '2' argument was initially '0' but it's incorrect
            I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)

            left_volume = np.asarray(
                self.mtu.get_bridge_adjacencies(face, 2, 3), dtype="uint64"
            )
            id_volume = self.mb.tag_get_data(self.global_id_tag, left_volume)[
                0
            ][0]
            id_volumes.append(id_volume)

            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
            LJ = (
                self.mb.get_coords([J])
                - self.mesh_data.mb.tag_get_data(
                    self.volume_centre_tag, left_volume
                )[0]
            )
            N_IJK = np.cross(JI, JK) / 2.0
            _test = np.dot(LJ, N_IJK)
            if _test < 0.0:
                I, K = K, I
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                N_IJK = np.cross(JI, JK) / 2.0
            tan_JI = np.cross(N_IJK, JI)
            tan_JK = np.cross(N_IJK, JK)
            self.mb.tag_set_data(self.normal_tag, face, N_IJK)

            face_area = np.sqrt(np.dot(N_IJK, N_IJK))
            h_L = geo.get_height(N_IJK, LJ)

            g_I = self.get_boundary_node_pressure(I)
            g_J = self.get_boundary_node_pressure(J)
            g_K = self.get_boundary_node_pressure(K)

            K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
                [3, 3]
            )
            K_n_L = self.vmv_multiply(N_IJK, K_L, N_IJK)
            K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
            K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

            D_JK = self.get_cross_diffusion_term(
                tan_JK, LJ, face_area, h_L, K_n_L, K_L_JK, boundary=True
            )
            D_JI = self.get_cross_diffusion_term(
                tan_JI, LJ, face_area, h_L, K_n_L, K_L_JI, boundary=True
            )
            K_eq = (1 / h_L) * (face_area * K_n_L)

            RHS = D_JK * (g_I - g_J) - K_eq * g_J + D_JI * (g_J - g_K)
            LHS = K_eq
            all_LHS.append(LHS)

            self.Q[id_volume] += -RHS
            self.aux_q[id_volume].append({node: -RHS for node in [I, J, K]})
            # self.Q[id_volume, 0] += - RHS
            # self.mb.tag_set_data(self.flux_info_tag, face,
            #                      [D_JK, D_JI, K_eq, I, J, K, face_area])

        all_cols = []
        all_rows = []
        all_values = []
        self.ids = []
        self.v_ids = []
        self.ivalues = []
        for face in self.intern_faces:
            left_volume, right_volume = self.mtu.get_bridge_adjacencies(
                face, 2, 3
            )
            L = self.mesh_data.mb.tag_get_data(
                self.volume_centre_tag, left_volume
            )[0]
            R = self.mesh_data.mb.tag_get_data(
                self.volume_centre_tag, right_volume
            )[0]
            dist_LR = R - L
            I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])

            N_IJK = np.cross(JI, JK) / 2.0
            test = np.dot(N_IJK, dist_LR)

            if test < 0:
                left_volume, right_volume = right_volume, left_volume
                L = self.mesh_data.mb.tag_get_data(
                    self.volume_centre_tag, left_volume
                )[0]
                R = self.mesh_data.mb.tag_get_data(
                    self.volume_centre_tag, right_volume
                )[0]
                dist_LR = R - L

            face_area = np.sqrt(np.dot(N_IJK, N_IJK))
            tan_JI = np.cross(N_IJK, JI)
            tan_JK = np.cross(N_IJK, JK)

            K_R = self.mb.tag_get_data(self.perm_tag, right_volume).reshape(
                [3, 3]
            )
            RJ = R - self.mb.get_coords([J])
            h_R = geo.get_height(N_IJK, RJ)

            K_R_n = self.vmv_multiply(N_IJK, K_R, N_IJK)
            K_R_JI = self.vmv_multiply(N_IJK, K_R, tan_JI)
            K_R_JK = self.vmv_multiply(N_IJK, K_R, tan_JK)

            K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
                [3, 3]
            )

            LJ = L - self.mb.get_coords([J])
            h_L = geo.get_height(N_IJK, LJ)

            K_L_n = self.vmv_multiply(N_IJK, K_L, N_IJK)
            K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
            K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

            D_JI = self.get_cross_diffusion_term(
                tan_JI,
                dist_LR,
                face_area,
                h_L,
                K_L_n,
                K_L_JI,
                h_R,
                K_R_JI,
                K_R_n,
            )
            D_JK = self.get_cross_diffusion_term(
                tan_JK,
                dist_LR,
                face_area,
                h_L,
                K_L_n,
                K_L_JK,
                h_R,
                K_R_JK,
                K_R_n,
            )

            K_eq = (K_R_n * K_L_n) / (K_R_n * h_L + K_L_n * h_R) * face_area

            id_right = self.mb.tag_get_data(self.global_id_tag, right_volume)
            id_left = self.mb.tag_get_data(self.global_id_tag, left_volume)

            col_ids = [id_right, id_right, id_left, id_left]
            row_ids = [id_right, id_left, id_left, id_right]
            values = [K_eq, -K_eq, K_eq, -K_eq]
            all_cols.append(col_ids)
            all_rows.append(row_ids)
            all_values.append(values)
            # wait for interpolation to be done
            self._node_treatment(I, id_left, id_right, K_eq, D_JK=D_JK)
            self._node_treatment(
                J, id_left, id_right, K_eq, D_JI=D_JI, D_JK=-D_JK
            )
            self._node_treatment(K, id_left, id_right, K_eq, D_JI=-D_JI)
            # self.mb.tag_set_data(self.flux_info_tag, face,
            #                      [D_JK, D_JI, K_eq, I, J, K, face_area])

        self.T.InsertGlobalValues(self.ids, self.v_ids, self.ivalues)
        # self.T[
        #     np.asarray(self.ids)[:, :, 0, 0], np.asarray(self.v_ids)
        # ] = np.asarray(self.ivalues)
        self.T.InsertGlobalValues(id_volumes, id_volumes, all_LHS)
        # self.T[
        #     np.asarray(id_volumes), np.asarray(id_volumes)
        # ] = np.asarray(all_LHS)
        self.T.InsertGlobalValues(all_cols, all_rows, all_values)
        # self.T[
        #     np.asarray(all_cols)[:, 0, 0, 0],
        #     np.asarray(all_rows)[:, 0, 0, 0]
        # ] = np.asarray(all_values)[:, 0]
        self.T.FillComplete()

    def get_only_positive_off_diagonal_values(self, rows, cols):
        positive_values = np.asarray(
            [
                [row, col] for row, col
                in zip(rows, cols) if self.T[row, col] > 0
                and row != col
            ]
        )
        positive_rows = positive_values[:, 0]
        positive_cols = positive_values[:, 1]
        return positive_rows, positive_cols

    def sum_into_diagonal(self, mat: lil_matrix):
        size = mat.shape[0]
        diagonals = [-np.sum(mat[row].toarray()) for row in range(mat.shape[0])]
        mat[range(size), range(size)] = diagonals
        mat.tocsc()

    def defect_correction(self):
        self.compute_all_mi()
        rows, cols = self.get_global_rows()
        A = self.copy_mat(rows, cols, shape=len(self.volumes))
        q = np.asarray(self.Q)
        p_rows, p_cols = self.get_only_positive_off_diagonal_values(
            rows, cols
        )
        A_plus = self.copy_mat(p_rows, p_cols, shape=len(self.volumes))
        self.sum_into_diagonal(A_plus)
        A_minus = A - A_plus
        x_minus = self.solve_original_problem(A_minus, q)
        self.mb.tag_set_data(self.pressure_tag, self.volumes, x_minus)
        self.tag_verts_pressure()
        all_faces = self.dirichlet_faces.union(
            self.neumann_faces.union(self.intern_faces)
        )
        for face in all_faces:
            alpha = self.compute_slip_fact(face)
            try:
                row, col = self.mb.tag_get_data(
                    self.global_id_tag, self.mtu.get_bridge_adjacencies(face, 2, 3)
                )
                A_plus[row, col] *= alpha
            except ValueError:
                # self.aux_q[id_volume].append({node: -RHS for node in [I, J, K]})
                pass

        residual = q -  (A_minus + A_plus)*x_minus# Resíduo - Equação 70 (Kuzmin)
        number = 0
        while np.max(np.abs(residual)) > 1E-10:    
            dx_minus = self.solve_original_problem(A_minus, residual)
            x_minus += dx_minus
            residual = q -  (A_minus + A_plus)*x_minus# Resíduo - Equação 70 (Kuzmin)
            number += 1
            
            if number == 100:
                break

        # Precisa verificar como chamar o volume e não aparecer números gigantes

    def savefig(self, df, column_name, figname):

        hist = df[column_name].hist(bins=15)
        fig = hist.get_figure()
        fig.savefig(figname)
        plt.close()
        



# sol = []
# for face in all_faces:
#     sl = self.compute_slip_fact(face)
#     print(sl)
#     try:
#         sol.append([face, sl[0][0]])
#     except TypeError:
#         sol.append([face, sl])
# print('done')
# df = pd.DataFrame(sol, columns=['face_id', 'alpha'])
# self.savefig(df, 'alpha', 'distribuicao_alpha_sol_A.png')
