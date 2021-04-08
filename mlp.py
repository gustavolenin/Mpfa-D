#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 00:06:29 2021

@author: gustavo
"""

x_minus = self.solve_original_problem(A, q) # Solução direta do problema, matriz A = matriz original
self.mb.tag_set_data(self.pressure_tag, self.volumes, x_minus)
self.tag_verts_pressure()
b_param_lpc = {}
for node in self.mesh_data.all_nodes:
     try:
         adj_volumes = [volume for volume in self.nodes_ws.get(node).keys()]
         adj_volumes_weigths = [volume for volume in self.nodes_ws.get(node).values()]
         adj_volumes_pressure = [pressure for pressure in self.mb.tag_get_data(self.pressure_tag, adj_volumes)]
         vert_pressure = np.dot(adj_volumes_weigths, adj_volumes_pressure)
     except AttributeError:
         adj_volumes = np.asarray(self.mtu.get_bridge_adjacencies(node, 0, 3)).tolist()
         vert_pressure = self.mb.tag_get_data(self.dirichlet_tag, node)
         adj_volumes_pressure = self.mb.tag_get_data(self.pressure_tag, adj_volumes)
     min_pressure, max_pressure = min(adj_volumes_pressure), max(adj_volumes_pressure)
     for adj_volume, adj_volume_pressure in zip(adj_volumes, adj_volumes_pressure):
         if adj_volume not in b_param_lpc.keys():
             b_param_lpc[adj_volume] = []
         if vert_pressure - adj_volume_pressure > 0:
             b_factor = min(1, (max_pressure - adj_volume_pressure) / (vert_pressure - adj_volume_pressure))
         elif vert_pressure - adj_volume_pressure < 0:
             b_factor = min(1, (min_pressure - adj_volume_pressure) / (vert_pressure - adj_volume_pressure))
         else:
             b_factor = 1
         b_param_lpc[adj_volume].append(b_factor)