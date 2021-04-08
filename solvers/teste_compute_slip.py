#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:41:53 2021

@author: gustavo
"""
import numpy as np
from single_phase_cases.flow_channel import FlowChannel

fl = FlowChannel()
self = fl.mpfad
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
    sl = self.compute_slip_fact(face)

def compute_slip_fact(volume):
    
    adj_volumes = self.mtu.get_bridge_adjacencies(volume, 2, 3)
    if len(adj_volumes) == 4:
        print('O volume é interno')
        for adj in adj_volumes:
            ui = self.mb.tag_get_data(self.pressure_tag, volume)
            max_v, min_v = self.get_local_min_max(volume)
            max_v_adj, min_v_adj = self.get_local_min_max(adj)
            uj = self.mb.tag_get_data(self.pressure_tag, adj)
            lp = ui - uj
            if ui > uj:
                lfs = 2 * self.mis[volume] * (max_v - ui)
                rts = 2 * self.mis[adj] * (uj - min_v_adj)
                # lfs = (max_v - ui)
                # rts = (uj - min_v_adj)
                _s = min(lfs, lp, rts)
                if _s < 0:
                    print('Passei no IF')
                    print('Volume avaliado: ',volume,'\n Slope limit = ',_s)
            else:
                lfs = 2 * self.mis[volume] * (min_v - ui)
                rts = 2 * self.mis[adj] * (uj - max_v)
                # lfs = (min_v - ui)
                # rts = (uj - max_v)
                _s = max(lfs, lp, rts)
                if _s < 0:
                    print('Passei no ELSE')
                    print('Volume avaliado: ',volume,'\n Slope limit = ',_s)
            return _s
    else:
        print('O volume pertence ao contorno')
        return 
        
[compute_slip_fact(volume) for volume in self.volumes-1]

        
        
        
        
        
        
        