# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:46:36 2024

@author: dphy-dupontferrielab
"""
import scipy.constants as c


ub = c.physical_constants['Bohr magneton'][0]
h = c.h

def f_res(g=2, g_dev=0.6, B=0.6, B_dev=0):

    g_min = g * (1 - g_dev / 100)
    g_max = g * (1 + g_dev / 100)
    
    B_min = B * (1 - B_dev / 100)
    B_max = B * (1 + B_dev / 100)
    
    
    f = g * ub * B / h
    f_min = g_min * ub * B_min / h
    f_max = g_max * ub * B_max / h
    
    print(f"f pour g = 2: {f*1e-9:6f} GHz")
    print(f"f pour g = {g_min} (-{g_dev}%): {f_min*1e-9:.6f} Ghz")
    print(f"f pour g = {g_max} (+{g_dev}%): {f_max*1e-9:.6f} Ghz")
    
    return f, f_min, f_max
    
def g_eff(f_res, B):
    
    g = h * f_res / (ub * B)
    g_dev = 50 * (2-g)
    print(f"g pour f_res = {f_res*1e-9} Ghz: g* = {g:.4f} ({g_dev:.4f}%) ")
    
    return g
    