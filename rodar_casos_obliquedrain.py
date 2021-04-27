%load_ext autoreload
%autoreload 2
import numpy as np 
from single_phase_cases.n_oblique_drain import ObliqueDrain
from solvers.interpolation.LPEW3 import LPEW3
mesh = 'meshes/oblique-drain.msh'
ob = ObliqueDrain(mesh)
ob.runCase(LPEW3,'teste')
self = ob.mpfad
self.compute_all_mi()
rows, cols = self.get_global_rows()
A = self.copy_mat(rows, cols, shape=len(self.volumes))
q = np.asarray(self.Q)
p_rows, p_cols = self.get_only_positive_off_diagonal_values(rows, cols)
A_plus = self.copy_mat(p_rows, p_cols, shape=len(self.volumes))
self.sum_into_diagonal(A_plus)
A_minus = A - A_plus
x_minus = self.solve_original_problem(A, q)
#x = self.solve_original_problem(A, q)
self.mb.tag_set_data(self.pressure_tag, self.volumes, x_minus)
self.tag_verts_pressure()
all_faces = self.dirichlet_faces.union(self.neumann_faces.union(self.intern_faces))
for face in all_faces:
    sl = self.compute_slip_fact(face)
    print(sl)