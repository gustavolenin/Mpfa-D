#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:46:48 2021

@author: gustavo
"""

''' Nessa seção do Pre Saturation será desenvolvida a parte de Gustavo,
    ou seja, a partir da linha 344'''
    
def getsurnode(inode):    
    
    a_node = mesh.nodes.all[ni]
    # Ordenamento para obtenção do nsurn
    order = MPFAD2DOrdering(mesh.nodes, "faces")
    nsurn = mesh.nodes.bridge_adjacencies(a_node, "edges", "nodes", ordering_inst=order)
    # Ordenamento para obtenção do esurn
    order = MPFAD2DOrdering(mesh.faces, "edges")
    esurn = mesh.nodes.bridge_adjacencies(a_node, "edges", "faces", ordering_inst=order)
    
    return esurn,nsurn

def getothervertex(elemeval,inode,countnsurn1):
    
    return othervertex