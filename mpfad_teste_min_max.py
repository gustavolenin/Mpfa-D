#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:28:00 2021

@author: gustavo
"""


import numpy as np
from single_phase_cases.flow_channel import FlowChannel

fl = FlowChannel()
self = fl.mpfad
rows, cols = self.get_global_rows()
A = self.copy_mat(rows, cols, shape=len(self.volumes))
q = np.asarray(self.Q)
x = self.solve_original_problem(A, q)
self.mb.tag_set_data(self.pressure_tag, self.volumes, x)
adj_volumes = self.mtu.get_bridge_adjacencies(self.volumes[-1], 2,3)
u_i = self.volumes[-1]; u_j = adj_volumes[0]; neumann_adj = self.mtu.get_bridge_adjacencies(self.volumes[-1], 0,3)
neumann_adj = np.asarray(neumann_adj)
neumann_adj = np.append(neumann_adj,np.asarray(u_i))
max_i = np.max(self.mb.tag_get_data(self.pressure_tag,neumann_adj.tolist()))
min_i = np.min(self.mb.tag_get_data(self.pressure_tag,neumann_adj.tolist()))