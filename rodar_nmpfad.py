#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:23:10 2021

@author: gustavo
"""

import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve
from single_phase_cases.flow_channel import FlowChannel

fl = FlowChannel()
self = fl.mpfad
self.compute_all_mi()
rows, cols = self.get_global_rows()
A = self.copy_mat(rows, cols, shape=len(self.volumes))
q = np.asarray(self.Q)
#p_rows, p_cols = self.get_only_positive_off_diagonal_values(
#    rows, cols
#)
#A_plus = self.copy_mat(p_rows, p_cols, shape=len(self.volumes))
#self.sum_into_diagonal(A_plus)
#A_minus = A - A_plus
x_minus = self.solve_original_problem(A_minus, q)
self.mb.tag_set_data(self.pressure_tag, self.volumes, x_minus)
self.tag_verts_pressure()
all_faces = self.dirichlet_faces.union(
    self.neumann_faces.union(self.intern_faces)
)
for face in all_faces:
    sl = self.compute_slip_fact(face)