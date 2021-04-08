#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 00:16:00 2021

@author: gustavo
"""
''' Rodar MPFAD Não Linear '''
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
    
''' Método MLP - Tese de Márcio '''

# A partir de um nó j avaliado são considerados os volumes de controle na vizinhança e são realizadas avaliações quanto aos valores de saturação 
# (Nesse caso pressão) a partir da extrapolação com base nos respectivos volumes de controle. Para mais informações consultar tese.

# Realização de MLP original, pensar na possibilidade de utilizar mlp-vk
    
x_minus = self.solve_original_problem(A, q) # Solução direta do problema, matriz A = matriz original
self.mb.tag_set_data(self.pressure_tag, self.volumes, x_minus)
self.tag_verts_pressure()
b_param_lpc = {}
for node in self.mesh_data.all_nodes:
     try:
         # Cálculo da pressão no vértice realizada a partir de uma ponderação baseada nos pesos de cada volume de controle vizinho ao nó
         adj_volumes = [volume for volume in self.nodes_ws.get(node).keys()]
         adj_volumes_weigths = [volume for volume in self.nodes_ws.get(node).values()]
         adj_volumes_pressure = [pressure for pressure in self.mb.tag_get_data(self.pressure_tag, adj_volumes)]
         vert_pressure = np.dot(adj_volumes_weigths, adj_volumes_pressure)
     except AttributeError:
         adj_volumes = np.asarray(self.mtu.get_bridge_adjacencies(node, 0, 3)).tolist()
         vert_pressure = self.mb.tag_get_data(self.dirichlet_tag, node)
         adj_volumes_pressure = self.mb.tag_get_data(self.pressure_tag, adj_volumes)
     min_pressure, max_pressure = min(adj_volumes_pressure), max(adj_volumes_pressure)
     # Necessidade de estudar melhor esse recurso zip
     for adj_volume, adj_volume_pressure in zip(adj_volumes, adj_volumes_pressure): # Equaçao (190) - Tese de Márcio - Página 104 pdf
         if adj_volume not in b_param_lpc.keys():
             b_param_lpc[adj_volume] = []
         if vert_pressure - adj_volume_pressure > 0:
             b_factor = min(1, (max_pressure - adj_volume_pressure) / (vert_pressure - adj_volume_pressure))
         elif vert_pressure - adj_volume_pressure < 0:
             b_factor = min(1, (min_pressure - adj_volume_pressure) / (vert_pressure - adj_volume_pressure))
         else:
             b_factor = 1
         b_param_lpc[adj_volume].append(b_factor) # Armazena os ID's e os valores da limitação para cada vértice
    
    # Verificar a equação (192) para ter noção do próximo passo.
    # Necessidade de uma equação para aplicar a limitação calculada por b_param_lpc
