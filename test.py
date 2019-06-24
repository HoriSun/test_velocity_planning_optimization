import numpy as np
from matplotlib import pyplot as plt

v0 = 0.0
a0 = 0.0
j0 = 0.0

p0 = 0
p1 = v0
p2 = a0/2.0
p3 = j0/6.0

def get_powers(t, order):
    ret = [0]*(order+1)
    ret[0] = 1
    for i in range(1, order+1):
        ret[i] = ret[i-1] * t

def f_s_t(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return (  2/3.0 * p5 * Ts[5] + 
                      p4 * Ts[4] +
                      p3 * Ts[3] + 
                      p2 * Ts[2] +
                      p1 * Ts[1] +
                      p0           )   
                     
def f_L_dT_dp5(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return ( 10/3.0      * Ts[4] +
                 12 * lv * Ts[3] +
                 30 * la * Ts[2] +
                 40 * lj * Ts[1] +
                 30 * l1         +
             10/3.0 * l4 * Ts[4] +
             45/8.0 * l5 * Ts[2]   )

def f_L_dT_dp4(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return (      4      * Ts[3] +
                 12 * lv * Ts[2] +
                 24 * la * Ts[1] +
                 24 * lj         +
                  4 * l4 * Ts[3] +
                  6 * l5 * Ts[1]   )
                  
def f_L_dT_dT(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return (      ( 40/3.0 * p5 * Ts[3] +
                        12 * p4 * Ts[2] +
                         6 * p3 * Ts[1] + 
                         2 * p2           ) +
             lv * (     36 * p5 * Ts[2] +
                        24 * p4 * Ts[1] + 
                         6 * p3           ) +
             la * (     60 * p5         +
                        24 * p4           ) +
             lj * (     40 * p5           ) +
             l4 * ( 40/3.0 * p5 * Ts[3] +
                        12 * p4 * Ts[2] +
                         6 * p3 * Ts[1] +
                         2 * p2           ) +
             l5 * ( 45/4.0 * p5 * Ts[1] +
                         6 * p4           )   )
                         

             