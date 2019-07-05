from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
import mpmath
import sys
from hl import Color
import matplotlib.animation as ani
    
v0 = 0.0
a0 = 0.0
j0 = 0.0

v1 = 2.0
s1 = 10.0
T1 = 10.0
a_max = 1.0

p0 = 0
p1 = v0
p2 = a0/2.0
p3 = j0/6.0

threshold = 1e-2

fva_list = []
t_list = []
p5_list = []
p4_list = []

clr = Color()

result_found = False

def plot_fva():
    if(not fva_list):
        return
    fva_list_a = np.array(fva_list).T
    print(fva_list_a)
    #plt.cla()
    #plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #plt.autoscale()
    ax.plot(fva_list_a[0], "o-", color="green", label="fv")
    ax.plot(fva_list_a[1], "o-", color="blue",  label="fa")
    
    #ax = plt.gca()
    #ax.plot(fva_list_a[0], color="green", label="fv")
    #ax.plot(fva_list_a[1], color="blue",  label="fa")
    ##ax.axis("square")
    #ax.axis("scaled")
    #ax.relim()
    #ax.autoscale()
    ax.grid(True)
    #ax.legend()
    #plt.grid(False)
    #plt.axis("scaled")
    ax.legend()
    #plt.show()


def plot_t():
    if(not t_list):
        return
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #plt.autoscale()
    ax.plot(t_list, "o-", color="black",  label="T")
    ax.plot(p5_list, "o-", color="red",  label="p5")
    ax.plot(p4_list, "o-", color="blue",  label="p4")
    
    #ax = plt.gca()
    #ax.plot(fva_list_a[0], color="green", label="fv")
    #ax.plot(fva_list_a[1], color="blue",  label="fa")
    ##ax.axis("square")
    #ax.axis("scaled")
    #ax.relim()
    #ax.autoscale()
    ax.grid(True)
    #ax.legend()
    #plt.grid(False)
    #plt.axis("scaled")
    ax.legend()
    #plt.show()

    
    
def plot_all_final(p5, p4, T):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    def f_s_temp(param):
        p5, p4, p3, p2, p1, p0, Ts, T_max = param
        p6 = -p5/(3.0*T_max) if Ts[1] else 0
        res =  (          p6 * Ts[6] + 
                          p5 * Ts[5] + 
                          p4 * Ts[4] +
                          p3 * Ts[3] + 
                          p2 * Ts[2] +
                          p1 * Ts[1] +
                          p0           )
        return res

    def f_fv_temp(param):
        p5, p4, p3, p2, p1, p0, Ts, T_max = param
        p6 = -p5/(3.0*T_max) if Ts[1] else 0
        res =  (  6 * p6 * Ts[5] +
                  5 * p5 * Ts[4] +
                  4 * p4 * Ts[3] +
                  3 * p3 * Ts[2] +
                  2 * p2 * Ts[1] +
                      p1           )
        return res

    def f_fa_temp(param):
        p5, p4, p3, p2, p1, p0, Ts, T_max = param
        p6 = -p5/(3.0*T_max) if Ts[1] else 0
        res =  ( 30 * p6 * Ts[4] +
                 20 * p5 * Ts[3] +
                 12 * p4 * Ts[2] +
                  6 * p3 * Ts[1] +
                  2 * p2          )
        return res

    def f_fj_temp(param):
        p5, p4, p3, p2, p1, p0, Ts, T_max = param
        p6 = -p5/(3.0*T_max) if Ts[1] else 0
        res =  ( 120 * p6 * Ts[3] +
                  60 * p5 * Ts[2] +
                  24 * p4 * Ts[1] +
                   6 * p3          )
        return res
    
    def f_fdj_temp(param):
        p5, p4, p3, p2, p1, p0, Ts, T_max = param
        p6 = -p5/(3.0*T_max) if Ts[1] else 0
        res =  ( 360 * p6 * Ts[2] +
                 120 * p5 * Ts[1] +
                  24 * p4           )
        return res
            
    t = np.linspace(0,T,100)
    param = p5, p4, p3, p2, p1, p0
    s  = np.array(list(map(lambda x:f_s_temp(  param+(get_powers(x,6),T)),t)))
    v  = np.array(list(map(lambda x:f_fv_temp( param+(get_powers(x,6),T)),t)))
    a  = np.array(list(map(lambda x:f_fa_temp( param+(get_powers(x,6),T)),t)))
    j  = np.array(list(map(lambda x:f_fj_temp( param+(get_powers(x,6),T)),t)))
    #dj = np.array(list(map(lambda x:f_fdj_temp(param+(get_powers(x,6),T)),t)))
    
    s_max = s.max()
    v_max = v.max()
    a_max = a.max()
    j_max = j.max()
    #dj_max = dj.max()
    
    s_min = s.min()
    v_min = v.min()
    a_min = a.min()
    j_min = j.min()
    #dj_min = dj.min()
    
    y_max = np.array([s_max, v_max, a_max, j_max, 
                      #dj_max
                      ]).max()
    y_min = np.array([s_min, v_min, a_min, j_min, 
                      #dj_min
                      ]).min()
    
    #y_max = dj_max
    #y_min = dj_min
    
    
    #ax.cla()
    #self.ax.axis("equal")
    margin = 1
    
    #ax.set_xlim(0 - margin, T + margin    )
    #ax.set_ylim(y_min - margin, y_max + margin)

    s_line, = ax.plot(t, s, "o-", color="red",   label="s")
    v_line, = ax.plot(t, v, "o-", color="blue",  label="v")
    a_line, = ax.plot(t, a, "o-", color="green", label="a")
    j_line, = ax.plot(t, j, "o-", color="orange", label="j")
        
    #self.dj_line.set_data(t,dj)
    
    
    n_s_max = np.argmax(s)
    n_v_max = np.argmax(v)
    n_a_max = np.argmax(a)
    n_j_max = np.argmax(j)
    #n_dj_max = np.argmax(dj)
    
    n_s_min = np.argmin(s)
    n_v_min = np.argmin(v)
    n_a_min = np.argmin(a)
    n_j_min = np.argmin(j)
    #n_dj_min = np.argmin(dj)
    
    s_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
    s_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
    v_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
    v_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
    a_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
    a_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
    j_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
    j_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
    
    
    annos = {"s":[(n_s_max, a_max_anno),
                  (n_s_min, a_min_anno)],
             "v":[(n_v_max, v_max_anno),
                  (n_v_min, v_min_anno)],
             "a":[(n_a_max, s_max_anno),
                  (n_a_min, a_min_anno)],
             "j":[(n_j_max, j_max_anno),
                  (n_j_min, j_min_anno)],
             #"dj":[(n_dj_max, self.dj_max_anno),
             #     (n_dj_min, self.dj_min_anno)]
             }
     
    lines = {"s":s,
             "v":v,
             "a":a,
             "j":j,
             #"dj":dj
             }
     
    #plt.plot([0,1.0],[1.5,1.5], "-.", label="target speed = %3.1fm/s"%(v_max))
    
    
    anno_artists = []
    
    for key in annos:
        line = lines[key]
        for i in annos[key]:
            #print(i)
            xy = (t[i[0]], line[i[0]])
            i[1].set_text(" %4.2lf"%xy[1])
            #i[1].set_position(xytext=xy, xy=xy)
            i[1].set_position(xy)
    
    #print_param(self.param)
    

    ##ax.axis("square")
    #ax.axis("scaled")
    #ax.axis("equal")
    #ax.relim()
    #ax.autoscale()
    ax.grid(True)
    #ax.legend()
    #plt.grid(False)
    #plt.axis("scaled")
    ax.legend()
    plt.show()

    
    
    
def err_handle():
    plt.close()
    plot_fva()
    plot_t()
    plt.show()

def get_ni(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    ni = len(list(filter(lambda x:x, [m1,m2,m3,m4,m5])))
    param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
    return ni, param

def print_ps(param, msg=""):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    p6 = -p5/(3.0*Ts[1]) if Ts[1] else np.inf
    print("%s"
          "p6=%lf, p5=%lf, p4=%lf, p3=%lf, p2=%lf, p1=%lf\n"
          ""%(msg+"\n" if msg else "", 
              p6, p5, p4, p3, p2, p1))

    
def print_param(param, msg=""):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    print("%s"
          "p5=%lf, p4=%lf, lv=%lf, la=%lf, lj=%lf, \n"
          "l1=%lf, l2=%lf, l3=%lf, l4=%lf, l5=%lf, \n"
          "T=%lf, ni=%d\n"
          "m1=%d, m2=%d, m3=%d, m4=%d, m5=%d\n"
          ""%(msg+"\n" if msg else "", 
              p5, p4, lv, la, lj, 
              l1, l2, l3, l4, l5, 
              Ts[1], ni,
              m1, m2, m3, m4, m5))

def get_powers(t, order):
    ret = [0]*(order+1)
    ret[0] = 1
    for i in range(1, order+1):
        ret[i] = ret[i-1] * t
    return ret

def f_s(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  (  2/3.0 * p5 * Ts[5] + 
                      p4 * Ts[4] +
                      p3 * Ts[3] + 
                      p2 * Ts[2] +
                      p1 * Ts[1] +
                      p0           )
    return res, param
                      
def f_s_dx(param):

    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  np.array( [ 2/3.0 * Ts[5],
                       Ts[4],
                       ( 10/3.0 * p5 * Ts[4] +
                              4 * p4 * Ts[3] +
                              3 * p3 * Ts[2] +
                              2 * p2 * Ts[1] +
                                  p1           ) ])
    #print("res=%s"%(res))
    return res, param

def f_s_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    s_dTdp5 =   10/3.0      * Ts[4]
    s_dTdp4 =        4      * Ts[3]
    s_d2T   = ( 40/3.0 * p5 * Ts[3] + 
                    12 * p4 * Ts[2] +
                     6 * p3 * Ts[1] +
                     2 * p2          )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [       0,       0, s_dTdp5 ] ,
                       [       0,       0, s_dTdp4 ] ,
                       [ s_dTdp5, s_dTdp4, s_d2T   ] ] )
    return res, param
    
def f_fv(param, asrt=True):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  (  3 * p5 * Ts[4] +
              4 * p4 * Ts[3] +
              3 * p3 * Ts[2] +
              2 * p2 * Ts[1] +
                  p1         - 
                  v1          )
    if np.abs(res)>=threshold and asrt:
        #lv = 0
        err_handle()
        raise AssertionError("fv()=%20.18lf, should be 0."%(res))
    return res, param
                                  
def f_fv_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  np.array( [ 3 * Ts[4],
                       4 * Ts[3],
                       (     12 * p5 * Ts[3] +
                             12 * p4 * Ts[2] +
                              6 * p3 * Ts[1] +
                              2 * p2           ) ])
    return res, param
    
def f_fv_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    fv_dTdp5 =       12      * Ts[3]
    fv_dTdp4 =       12      * Ts[2]
    fv_d2T   = (     36 * p5 * Ts[2] +
                     24 * p4 * Ts[1] + 
                      6 * p3           )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, fv_dTdp5 ] ,
                       [        0,        0, fv_dTdp4 ] ,
                       [ fv_dTdp5, fv_dTdp4, fv_d2T   ] ] )
    return res, param
    
def f_fa(param, asrt=True):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  ( 10 * p5 * Ts[3] +
             12 * p4 * Ts[2] +
              6 * p3 * Ts[1] +
              2 * p2          )
    
    if np.abs(res)>=threshold and asrt:
        #la = 0
        err_handle()
        raise AssertionError("fa()=%20.18lf, should be 0."%(res))
    return res, param

def f_fa_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  np.array( [ 10 * Ts[3],
                       12 * Ts[2],
                       (     30 * p5 * Ts[2] +
                             24 * p4 * Ts[1] +
                              6 * p3           ) ])
    return res, param

def f_fa_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    fa_dTdp5 =       30      * Ts[2]
    fa_dTdp4 =       24      * Ts[1]
    fa_d2T   = (     60 * p5 * Ts[1] +
                     24 * p4           )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, fa_dTdp5 ] ,
                       [        0,        0, fa_dTdp4 ] ,
                       [ fa_dTdp5, fa_dTdp4, fa_d2T   ] ] )
    return res, param

def f_fj(param, asrt=True):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  ( 20 * p5 * Ts[2] +
             24 * p4 * Ts[1] +
              6 * p3          )
    if res!=0 and asrt:
        #lj = 0
        err_handle()
        raise AssertionError("fj()=%lf, should be 0."%(res))
    return res, param

def f_fj_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  np.array( [ 20 * Ts[2],
                       24 * Ts[1],
                       (     40 * p5 * Ts[1] +
                             24 * p4           ) ])
    return res, param

def f_fj_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    fj_dTdp5 =       40      * Ts[1]
    fj_dTdp4 =       24
    fj_d2T   = (     40 * p5           )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, fj_dTdp5 ] ,
                       [        0,        0, fj_dTdp4 ] ,
                       [ fj_dTdp5, fj_dTdp4, fj_d2T   ] ] )
    return res, param

def f_f1(param, asrt=True):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    m1 = 1
    res =  ( 30 * p5 * Ts[1] +
             24 * p4          )
    if res!=0:
        if res > 0 and asrt:
            raise AssertionError("f1=%32.30lf > 0"%res)
        l1 = 0
        m1 = 0
    param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
    return res, param

def f_f1_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  np.array( [ 30 * Ts[1],
                       24,
                       (     30 * p5           ) ])
    return res, param

def f_f1_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    f1_dTdp5 =       30
    f1_dTdp4 =        0
    f1_d2T   = (      0                )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, f1_dTdp5 ] ,
                       [        0,        0, f1_dTdp4 ] ,
                       [ f1_dTdp5, f1_dTdp4, f1_d2T   ] ] )
    return res, param

    
############ f2 is ignored ####################
def f_f2(param, asrt=True):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    m2 = 1
    res =  ( Ts[1] - T1 )
    if res!=0:
        if res > 0 and asrt:
            raise AssertionError("f2=%32.30lf > 0"%res)
        l2 = 0
        m2 = 0
    param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
    return res, param

def f_f2_dx(param):
    res =  np.array( [ 0,
                       0,
                       1 ])
    return res, param

def f_f2_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    f2_dTdp5 =        0
    f2_dTdp4 =        0
    f2_d2T   = (      0                )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, f2_dTdp5 ] ,
                       [        0,        0, f2_dTdp4 ] ,
                       [ f2_dTdp5, f2_dTdp4, f2_d2T   ] ] )
    return res, param
############ f2 is ignored ####################

              
def f_f3(param, asrt=True):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    m3 = 1
    res =  ( Ts[2] - T1*Ts[1] )
    if res!=0:
        if res > 0 and asrt:
            raise AssertionError("f3=%32.30lf > 0"%res)
        l3 = 0
        m3 = 0
    param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
    return res, param

def f_f3_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  np.array( [  0,
                        0,
                        2*Ts[1] - T1 ])
    return res, param

def f_f3_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    f3_dTdp5 =        0
    f3_dTdp4 =        0
    f3_d2T   = (      2                )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, f3_dTdp5 ] ,
                       [        0,        0, f3_dTdp4 ] ,
                       [ f3_dTdp5, f3_dTdp4, f3_d2T   ] ] )
    return res, param
              
def f_f4(param, asrt=True):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    m4 = 1
    s_res, param = f_s(param)
    res =  ( s_res -
             1/2.0 * s1     )
    if res!=0:
        if res > 0 and asrt:
            raise AssertionError("f4=%32.30lf > 0"%res)
        l4 = 0
        m4 = 0
    param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
    return res, param
                       
def f_f4_dx(param):
    res, param =  f_s_dx(param)
    return res, param

def f_f4_d2x(param):
    res, param = f_s_d2x(param)
    return res, param
                           
def f_f5(param, asrt=True):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    m5 = 1
    def f5(p5):
        return ( 15/8.0 * p5 * Ts[3] +
                      3 * p4 * Ts[2] +
                      3 * p3 * Ts[1] +
                      2 * p2         - 
                          a_max        )
    res = f5(p5)
    print("-----------0 f_f5()  p5=%32.30lf   res=%32.30lf\n"%(p5, res))
                          
    if res > 0:
    
        #p5 = (a_max - 2*p2 - 3*p3*Ts[1] - 3*p4*Ts[2]) / (15/8.0*Ts[3])
    
        #raise AssertionError("f5=%32.30lf > 0"%res)
        #counter = 0
        while np.abs(res) > 1e-13 and res > 0 and False:
            dp5 = -res / (15/8.0*Ts[3])
            p5 += dp5
            #counter += 1
            res = f5()
            print("************** f_f5()  p5=%32.30lf   res=%32.30lf\n"%(p5, res))
            #if(counter >= 3):
            #    sys.exit(0)
        else:
            pass
            #sys.exit(0)
        res = f5(p5)
        print("-----------1 f_f5()  p5=%32.30lf   res=%32.30lf\n"%(p5, res))
            
                      
    if np.abs(res) > 1e-7:
        if res > 0 and asrt:
            raise AssertionError("f5=%32.30lf > 0"%res)
        l5 = 0
        m5 = 0
        #raise AssertionError
    print("f_f5()  m5=%s"%(repr(m5)))
    param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
    return res, param

def f_f5_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    res =  np.array( [ 15/8.0 * Ts[3],
                       3 * Ts[2],
                       ( 45/8.0 * p5 * Ts[2] +
                              6 * p4 * Ts[1] +
                              3 * p3           ) ])
    return res, param

def f_f5_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    f5_dTdp5 =   45/8.0 * Ts[2]
    f5_dTdp4 =        6 * Ts[1]
    f5_d2T   = ( 45/4.0 * p5 * Ts[1] +
                      6 * p4           )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, f5_dTdp5 ] ,
                       [        0,        0, f5_dTdp4 ] ,
                       [ f5_dTdp5, f5_dTdp4, f5_d2T   ] ] )
    return res, param
                  
def f_ce(param):

    # fv, fa, fj must be 0, as they are all equality constraints
    fv, param = f_fv(param)
    fa, param = f_fa(param)
    #fj, param = f_fj(param)
    
    print("fv=%lf fa=%lf\n"
          ""%(fv, fa))
    fva_list.append([fv,fa])
    
    if(any(map(lambda x:np.abs(x)>=threshold,
               [fv,
                fa,
                #fj
                   ]))):
        err_handle()
        raise AssertionError("fv, fa=(%lf, %lf)"
                             ", should be all 0."%(fv, fa))
    
    res =  np.array( [ 0, 0 ] )
    return res, param
              
def f_ce_dx(param):

    fv_dx, param = f_fv_dx(param)
    fa_dx, param = f_fa_dx(param)
    #fj_dx, param = f_fj_dx(param)#
    res =  np.concatenate( ( [fv_dx] ,
                             [fa_dx] ,
                             #[fj_dx] 
                                     ) , axis = 0 )
    return res, param

    
def f_ci(param):
    
    f1, param = f_f1(param)
    #f2, param = f_f2(param)
    f2, _     = f_f2(param) # not using l2
    f3, param = f_f3(param)
    f4, param = f_f4(param)
    f5, param = f_f5(param)
    
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    
    print("f_ci()  f1=%lf f2=%lf f3=%lf f4=%lf f5=%lf"
          ""%(f1, f2, f3, f4, f5))
    print("f_ci()  m1=%d m2=%d m3=%d m4=%d m5=%d\n"
          ""%(m1, m2, m3, m4, m5))
    
    #lf = [(l1, f1),
    #      (l2, f2),
    #      (l3, f3),
    #      (l4, f4),
    #      (l5, f5)]
    
    # still, they should all be zero, normally.
    
    ni,param = get_ni(param)
    
    print("f_ci()  f1=%lf f2=%lf f3=%lf f4=%lf f5=%lf"
          ""%(f1, f2, f3, f4, f5))
    print("f_ci()  m1=%d m2=%d m3=%d m4=%d m5=%d\n"
          ""%(m1, m2, m3, m4, m5))
    
    if(ni):
        fs = [0] * ni
    else:
        fs = []
    
    #print_param(param)
    
    res =  np.array( fs )
    
    param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
    return res, param
    
    
def f_ci_dx(param):

    df1, param = f_f1_dx(param)
    df2, param = f_f2_dx(param)
    df3, param = f_f3_dx(param)
    df4, param = f_f4_dx(param)
    df5, param = f_f5_dx(param)
    
    ni,param = get_ni(param)
    
    if(ni):
        p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
        
        lf = [(m1, df1),
              (m2, df2),
              (m3, df3),
              (m4, df4),
              (m5, df5)]
        
        ci_dx = list(map(lambda y:[y[1]],filter(lambda x:x[0],lf)))
        
        res =  np.concatenate( ( ci_dx ) , axis = 0 )
    else:
        res =  np.array([[]])
                             
    return res, param
                                  
def f_c(param):
    ce, param = f_ce(param)
    ci, param = f_ci(param)
    
    ni,param = get_ni(param)
    #print(ce, ci, ni)
    
    if(ni):
        res =  np.concatenate( ( ce ,
                                 ci ), axis = 0 )
    else:
        res =  ce
        
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
    print("f_c()  m1=%d m2=%d m3=%d m4=%d m5=%d\n"
          ""%(m1, m2, m3, m4, m5))
        
        
    return res, param

def f_c_dx(param):
    ce_dx, param = f_ce_dx(param)
    ci_dx, param = f_ci_dx(param)
    
    ni,param = get_ni(param)
    print("f_c_dx() ", ce_dx, ci_dx)
    if(ni):
        res =  np.concatenate( ( ce_dx , 
                                 ci_dx ), axis = 0 )
    else:
        res = ce_dx
        
    return res, param
                                      
             
def f_L_dx(param):



    s_dx, param  = f_s_dx(param)
    ce_dx, param  = f_ce_dx(param)
    ci_dx, param  = f_ci_dx(param)

    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param

    #print_param(param)
    
    le = np.array([lv,
                   la,
                   #lj
                     ]).reshape((1,2))
    
    lf = [(m1, l1),
          (m2, l2),
          (m3, l3),
          (m4, l4),
          (m5, l5)]
          
    lilist = list(map(lambda y:y[1], filter(lambda x:x[0], lf)))
    
    li = np.array(lilist).reshape((1,len(lilist)))
    
    if(0):
        print("s_dx=%s"%s_dx)
    
        print("le=%s"%le)
        print("li=%s"%li)
    
        print("ce_dx=%s"%ce_dx)
        print("ci_dx=%s"%ci_dx)
    
    res = s_dx + le.dot(ce_dx)
    ni,param = get_ni(param)
    
    if(ni):
        res += li.dot(ci_dx)
        
    res = res[0]
    
    return res, param
             


def f_L_d2x(param):

    res = np.zeros((3,3))
    
    fv_d2x, param = f_fv_d2x(param)
    fa_d2x, param = f_fa_d2x(param)
    #fj_d2x, param = f_fj_d2x(param)
    
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
        
    res += ( fv_d2x +
             fa_d2x
             #fj_d2x 
                     )
    
    lf = ((m1, l1, f_f1_d2x),
          (m2, l2, f_f2_d2x),
          (m3, l3, f_f3_d2x),
          (m4, l4, f_f4_d2x),
          (m5, l5, f_f5_d2x))

    ni,param = get_ni(param)
    
    if(ni):
        res += reduce(lambda y0,y1:y0+y1, 
                      map(lambda y:y[1]*y[2](param)[0], 
                          filter(lambda x:x[0], lf)))
    
    return res, param
    

# ( 3 * p5 * T^4 ) + ( 4 * p4 * T^3 ) = v1 - v0



def loop_gen():    
    
    init_param = (1e-1,) * 17
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, T, m1, m2, m3, m4, m5, ni = init_param
    ni = 5
    m1, m2, m3, m4, m5 = (1,) * 5

    T = 1
    dv = v1 - v0

    def get_p5_p4(T, dv):
        A = np.array( [ [ 3*(T**4), 4*(T**3) ] ,
                        [ 5*(T**4), 6*(T**3) ] ] )
        b = np.array( [ dv, 0 ] )
        p5, p4 = np.linalg.solve( A, b )
        
        return p5, p4
        
    p5, p4 = get_p5_p4(T, dv)


    def loop():
        Ts = get_powers(T, 5)

        if(0):
            v = (  3 * p5 * Ts[4] +
                   4 * p4 * Ts[3] +
                   3 * p3 * Ts[2] +
                   2 * p2 * Ts[1] +
                       p1           )
            a = ( 10 * p5 * Ts[3] +
                  12 * p4 * Ts[2] +
                   6 * p3 * Ts[1] +
                   2 * p2           )
            print("T=%lf  v1=%lf  v0=%lf  dv=%lf  p4=%lf  p5=%lf  v=%lf  a=%lf\n"
                  ""%(T, v1, v0, dv, p4, p5, v, a))
         
        #print(Ts)
        param = (p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni)

        # shutdown one condition
        p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
        m2 = 0 
        fs = [0] * len(list(filter(lambda x:x, [m1,m2,m3,m4,m5])))
        ni = len(fs)    
        param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni

        c, param = f_c(param)

        L_dx, param = f_L_dx(param)
        print("L_dx=",L_dx)
        print("c=", c)
        b = np.concatenate( ( -L_dx,
                              -c ), axis = 0 )
        #print("b=",b)
                      
        L_d2x, param = f_L_d2x(param)
        c_dx, param = f_c_dx(param)
        #print(c_dx)
        #print(L_d2x.shape, c_dx.shape)
        A_0 = np.concatenate( ( L_d2x, c_dx.T         ), axis = 1 )

        ni,param = get_ni(param)
            
        nc = 2 + ni


        A_1 = np.concatenate( ( c_dx , np.zeros((nc,nc))), axis = 1 )
        A = np.concatenate( ( A_0, A_1 ), axis = 0 )

        print(A.shape,b.shape)
        #print(A,"="*10,b)

        nd = nc + 3

        if(1):
            f = open("test.csv","w")
            for i in range(nd):
                for j in range(nd):
                    f.write("%lf,"%A[i][j])
                f.write(",%lf"%b[i])
                f.write("\n")
            f.close()

        #x = np.linalg.solve(A,b)
        
        x = mpmath.lu_solve(mpmath.matrix(A.tolist()),
                            mpmath.matrix(b.tolist()))
        x = list(map(lambda y:float(str(y)),x))
        
        #x = np.dot(np.linalg.inverse(A), b)

        if(0):
            f = open("test.csv","w")
            for i in range(nd):
                for j in range(nd):
                    f.write("%lf,"%A[i][j])
                f.write(",%lf,,"%b[i])
                f.write("%lf,"%x[i])
                f.write("\n")
            f.close()

        print(x)

        B,d,D = (np.linalg.svd(A))
        if(0):
            f = open("test.csv","w")
            for i in range(nd):
                for j in range(nd):
                    f.write("%lf,"%B[i][j])
                f.write(",%lf,,"%d[i])
                for j in range(nd):
                    f.write("%lf,"%D[i][j])
                f.write("\n")
                
            f.close()

    

            
            
class iter(object):
    def __init__(self, ax):
        self.inited = False
        self.counter = 0
       
        self.ax = ax
        self.ax.grid(True)
        #self.ax.axis("equal")
        self.ax.set_xlabel("time (s)", fontsize=12)
        #self.ax.set_xlim(0, 10)
        #self.ax.set_ylim(0, 100)
        
        init_param = (1e-1,) * 17
        p5, p4, lv, la, lj, l1, l2, l3, l4, l5, T, m1, m2, m3, m4, m5, ni = init_param
        ni = 5
        m1, m2, m3, m4, m5 = (1,) * 5

        T = 2
        
        dv = v1 - v0

        def get_p5_p4(T, dv):
            A = np.array( [ [ 3*(T**4), 4*(T**3) ] ,
                            [ 5*(T**4), 6*(T**3) ] ] )
            b = np.array( [ dv, 0 ] )
            p5, p4 = np.linalg.solve( A, b )
            
            return p5, p4
            
        p5, p4 = get_p5_p4(T, dv)

        p5_list.append(p5)
        p4_list.append(p4)
        t_list.append(T)
        
        Ts = get_powers(T, 5)

        self.param = (p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni)
        
        self.is_1st_loop = True
        
    def loop(self):
        # shutdown one condition
        p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = self.param
        
        m2 = 0 
        fs = [0] * len(list(filter(lambda x:x, [m1,m2,m3,m4,m5])))
        ni = len(fs)    
        param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni

        mf = [m1,m2,m3,m4,m5]
        print("loop() -5 mf=%s"%(mf))
        
        if(self.is_1st_loop):
            self.is_1st_loop = False
        else:
            
            c, param = f_c(param)

            p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = param
            
            #print_param(param,"="*10+"[1]")
            mf = [m1,m2,m3,m4,m5]
            print("loop() -4 mf=%s"%(mf))
            
            
            L_dx, param = f_L_dx(param)
            print("loop() L_dx=%s"%(L_dx))
            #print("c=", c)
            b = np.concatenate( ( -L_dx,
                                  -c ), axis = 0 )
            #print("b=",b)
          
            mf = [m1,m2,m3,m4,m5]
            print("loop() -3 mf=%s"%(mf))
            
          
            L_d2x, param = f_L_d2x(param)
            
            mf = [m1,m2,m3,m4,m5]
            print("loop() -2 mf=%s"%(mf))
            
            c_dx, param = f_c_dx(param)
            
            mf = [m1,m2,m3,m4,m5]
            print("loop() -1 mf=%s"%(mf))
            
            
            print("loop() c_dx=%s"%(c_dx))
            print_param(param, "[ loop()  1 param ]")
            #print(L_d2x.shape, c_dx.shape)
            A_0 = np.concatenate( ( L_d2x, c_dx.T         ), axis = 1 )

            
            mf = [m1,m2,m3,m4,m5]
            print("loop() 0 mf=%s"%(mf))
            
            
            ni,param = get_ni(param)
            
            mf = [m1,m2,m3,m4,m5]
            print("loop() 1 mf=%s"%(mf))
            
            
            nc = 2 + ni

            #print_param(param,"="*10+"[2]")
            

            A_1 = np.concatenate( ( c_dx , np.zeros((nc,nc))), axis = 1 )
            A = np.concatenate( ( A_0, A_1 ), axis = 0 )

            #print(A.shape,b.shape)
            #print(A,"="*10,b)

            nd = nc + 3

            if(1):
                f = open("test.csv","w")
                for i in range(nd):
                    for j in range(nd):
                        f.write("%lf,"%A[i][j])
                    f.write(",%lf"%b[i])
                    f.write("\n")
                f.close()

            x = np.linalg.solve(A,b)

            #print_param(param,"="*10+"[3]")
            mf = [m1,m2,m3,m4,m5]
            print("loop() 2 mf=%s"%(mf))
            
            
            if(0):
                prec_n = 30
                prec = "%d.%d"%(prec_n+2, prec_n)
                f = open("test.csv","w")
                for i in range(nd):
                    for j in range(nd):
                        f.write("%%%slf,"%prec%A[i][j])
                    f.write(",%%%slf,,"%prec%b[i])
                    f.write("%%%slf,"%prec%x[i])
                    f.write("\n")
                f.close()

            print(x)
            
            T = Ts[1]
            
            scale = 1
            
            p5 += x[0] * scale
            p4 += x[1] * scale
            T  += x[2] * scale
            lv += x[3] * scale
            la += x[4] * scale
            
            p5_list.append(p5)
            p4_list.append(p4)
            t_list.append(T)
            
            
            Ts = get_powers(T, 5)
            
            
            
            mf = [m1,m2,m3,m4,m5]
            lf = [0,0,0,0,0]
            
            
            j = 0
            for i in range(5,5+ni):
                while(mf[j]==0):
                    j+=1
                else:
                    lf[j] = x[i]
            
            print("loop() 3 mf=%s"%(mf))
            print("loop()  lf=%s"%(lf))
            
            
            l1 += lf[0] * scale
            l2 += lf[1] * scale
            l3 += lf[2] * scale
            l4 += lf[3] * scale
            l5 += lf[4] * scale
            #B,d,D = (np.linalg.svd(A))
            
        self.param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni

        
        da, param = f_fa(self.param, asrt=False)
        dv, param = f_fv(self.param, asrt=False)
        
        counter = 0
        
        temp_param = self.param
        
        print("[%d] +++: da=%32.30lf  dv=%32.30lf"%(counter, da,dv))
        while (np.abs(da)>1e-13 or np.abs(dv)>1e-13) and 1:
            print("[%d] before: da=%32.30lf  dv=%32.30lf"%(counter, da,dv))
            a_dx, param = f_fa_dx(temp_param)
            v_dx, param = f_fv_dx(temp_param)
            
            dM = np.array( [ [  3 * Ts[4],  4 * Ts[3] ] ,
                             [ 10 * Ts[3], 12 * Ts[2] ] ] )
                            
            dva = np.array( [ dv ,
                              da ] )
                              
            print(dM, dva)
                              
            dp5, dp4 = np.linalg.solve(dM, -dva)
           
            print(dp5, dp4)
            
            p5 = p5 + dp5
            p4 = p4 + dp4
            
            temp_param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni

            da_before = da
            dv_before = dv
            
            da, param = f_fa(temp_param, asrt=False)
            dv, param = f_fv(temp_param, asrt=False)
            
            dda = np.abs(da) - np.abs(da_before)
            ddv = np.abs(dv) - np.abs(dv_before)
            
            #print("[%d] after: da=%32.30lf  dv=%32.30lf"%(counter, da,dv))
            print("---- [%d] change: dda=%32.30lf  ddv=%32.30lf"%(counter, dda,ddv))
            counter += 1
            
            #if(counter >= 10):
            #    sys.exit(0)
            
        if (da or dv):
            M_ = np.array( [ [  3 * Ts[4],  4 * Ts[3] ] ,
                             [ 10 * Ts[3], 12 * Ts[2] ] ] )
                            
            va = np.array( [ v1 - p1 ,
                             0 ] )
            
        #print_param(param,"="*10+"[4]")
        print_param(self.param, "loop()  after da dv mod")
        
        def f_c_calc(param):
            f1, param = f_f1(param, asrt=False)
            f2, param = f_f2(param, asrt=False)
            f3, param = f_f3(param, asrt=False)
            f4, param = f_f4(param, asrt=False)
            f5, param = f_f5(param, asrt=False)
            res = [f1,f2,f3,f4,f5]

            return res, param
        
        fs, param = f_c_calc(temp_param)
        clr.print_red_text("loop()  fs=%s"%(fs))
        
        counter = 0
        f1, f2, f3, f4, f5 = fs
            
        while ( any(map(lambda x:x>0 and np.abs(x)>1e-13, fs)) or 
                any(map(lambda x:np.abs(x)>1e-13, [dv, da])) ):
            counter+=1
            clr.print_green_text("[%d] loop()  fs=%s"%(counter, repr(fs)))
            if(f5>0 and np.abs(f5)>1e-13):
                print("[%d] [f5] before: da=%32.30lf  dv=%32.30lf  f5=%32.30lf"%(counter, da,dv,f5))
            
                p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = temp_param
                
                T = Ts[1]
                
                fv_dx, param = f_fv_dx(temp_param)
                fa_dx, param = f_fa_dx(temp_param)
                f5_dx, param = f_f5_dx(temp_param)
                
                dv, param = f_fv(temp_param, asrt=False)
                da, param = f_fa(temp_param, asrt=False)
                f5, param = f_f5(temp_param, asrt=False)
                    
                fs = (f1, f2, f3, f4, f5)
                    
                A_ = np.concatenate( ( [fv_dx] ,
                                       [fa_dx] ,
                                       [f5_dx] ), axis=0 )
                
                b_ = np.array( [ dv ,
                                 da ,
                                 f5 ] )
                                  
                print(A_, b_)
                                  
                dp5, dp4, dT = np.linalg.solve(A_, -b_)
                p5 += dp5
                p4 += dp4
                T += dT
                Ts = get_powers(T, 6)
                print("[%d] [f5] dp5=%32.30lf  dp4=%32.30lf  dT=%32.30lf"%(counter, dp5, dp4, dT))
                print("[%d] [f5] after: da=%32.30lf  dv=%32.30lf  f5=%32.30lf"%(counter, da,dv,f5))
                
            elif(np.abs(dv) > 1e-13 or np.abs(da) > 1e-13 ):
                print("[%d] [dva] before: da=%32.30lf  dv=%32.30lf"%(counter, da,dv))
                a_dx, param = f_fa_dx(temp_param)
                v_dx, param = f_fv_dx(temp_param)
                
                dM = np.array( [ [  3 * Ts[4],  4 * Ts[3] ] ,
                                 [ 10 * Ts[3], 12 * Ts[2] ] ] )
                                
                dva = np.array( [ dv ,
                                  da ] )
                                  
                print(dM, dva)
                                  
                dp5, dp4 = np.linalg.solve(dM, -dva)
               
                print(dp5, dp4)
                
                p5 = p5 + dp5
                p4 = p4 + dp4
                
                temp_param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni

                da_before = da
                dv_before = dv
                
                da, param = f_fa(temp_param, asrt=False)
                dv, param = f_fv(temp_param, asrt=False)
                f1, param = f_f1(temp_param, asrt=False)
                f2, param = f_f2(temp_param, asrt=False)
                f3, param = f_f3(temp_param, asrt=False)
                f4, param = f_f4(temp_param, asrt=False)
                f5, param = f_f5(temp_param, asrt=False)
                
                dda = np.abs(da) - np.abs(da_before)
                ddv = np.abs(dv) - np.abs(dv_before)
                
                
                print("[%d] [f5] dp5=%32.30lf  dp4=%32.30lf  dT=%32.30lf"%(counter, dp5, dp4, dT))
                print("---- [%d] change: dda=%32.30lf  ddv=%32.30lf"%(counter, dda,ddv))
                print("[%d] [dva] after: da=%32.30lf  dv=%32.30lf"%(counter, da,dv))
                
                
            temp_param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
        else:
            clr.print_blue_text("[%d] loop()  fs=%s"%(counter, repr(fs)))
            
        self.param = temp_param
        
        c, param = f_c(self.param)

        #print_param(param,"="*10+"[1]")
        
        
        L_dx, param = f_L_dx(self.param)
        print("loop() c=%s"%(c))
        clr.print_green_text("loop() L_dx=%s"%(L_dx))
        if(np.linalg.norm(L_dx) < 1e-10):
            p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = self.param
            T = Ts[1]
            clr.print_green_text("Calc success. \n"
                                 "p5=%32.30lf, p4=%32.30lf, T=%32.30lf\n"
                                 ""%(p5, p4, T))
            #anim.event_source.stop()
            result_found = True
            plot_all_final(p5, p4, T)
        
        
        if(0):
            f = open("test.csv","w")
            for i in range(nd):
                for j in range(nd):
                    f.write("%lf,"%B[i][j])
                f.write(",%lf,,"%d[i])
                for j in range(nd):
                    f.write("%lf,"%D[i][j])
                f.write("\n")
                
            f.close()

    def plot(self):
        p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni = self.param
        
        T = Ts[1]
    
        def f_s_temp(param):
            p5, p4, p3, p2, p1, p0, Ts, T_max = param
            p6 = -p5/(3.0*T_max) if Ts[1] else 0
            res =  (          p6 * Ts[6] + 
                              p5 * Ts[5] + 
                              p4 * Ts[4] +
                              p3 * Ts[3] + 
                              p2 * Ts[2] +
                              p1 * Ts[1] +
                              p0           )
            return res

        def f_fv_temp(param):
            p5, p4, p3, p2, p1, p0, Ts, T_max = param
            p6 = -p5/(3.0*T_max) if Ts[1] else 0
            res =  (  6 * p6 * Ts[5] +
                      5 * p5 * Ts[4] +
                      4 * p4 * Ts[3] +
                      3 * p3 * Ts[2] +
                      2 * p2 * Ts[1] +
                          p1           )
            return res

        def f_fa_temp(param):
            p5, p4, p3, p2, p1, p0, Ts, T_max = param
            p6 = -p5/(3.0*T_max) if Ts[1] else 0
            res =  ( 30 * p6 * Ts[4] +
                     20 * p5 * Ts[3] +
                     12 * p4 * Ts[2] +
                      6 * p3 * Ts[1] +
                      2 * p2          )
            return res

        def f_fj_temp(param):
            p5, p4, p3, p2, p1, p0, Ts, T_max = param
            p6 = -p5/(3.0*T_max) if Ts[1] else 0
            res =  ( 120 * p6 * Ts[3] +
                      60 * p5 * Ts[2] +
                      24 * p4 * Ts[1] +
                       6 * p3          )
            return res
        
        def f_fdj_temp(param):
            p5, p4, p3, p2, p1, p0, Ts, T_max = param
            p6 = -p5/(3.0*T_max) if Ts[1] else 0
            res =  ( 360 * p6 * Ts[2] +
                     120 * p5 * Ts[1] +
                      24 * p4           )
            return res
                
        t = np.linspace(0,T,100)
        param = p5, p4, p3, p2, p1, p0
        s  = np.array(list(map(lambda x:f_s_temp(  param+(get_powers(x,6),T)),t)))
        v  = np.array(list(map(lambda x:f_fv_temp( param+(get_powers(x,6),T)),t)))
        a  = np.array(list(map(lambda x:f_fa_temp( param+(get_powers(x,6),T)),t)))
        j  = np.array(list(map(lambda x:f_fj_temp( param+(get_powers(x,6),T)),t)))
        dj = np.array(list(map(lambda x:f_fdj_temp(param+(get_powers(x,6),T)),t)))
        
        s_max = s.max()
        v_max = v.max()
        a_max = a.max()
        j_max = j.max()
        dj_max = dj.max()
        
        s_min = s.min()
        v_min = v.min()
        a_min = a.min()
        j_min = j.min()
        dj_min = dj.min()
        
        y_max = np.array([s_max, v_max, a_max, j_max, 
                          #dj_max
                          ]).max()
        y_min = np.array([s_min, v_min, a_min, j_min, 
                          #dj_min
                          ]).min()
        
        #y_max = dj_max
        #y_min = dj_min
        
        
        #ax.cla()
        #self.ax.axis("equal")
        margin = 1
        
        if(not result_found):
            self.ax.set_xlim(0 - margin, T + margin    )
            self.ax.set_ylim(y_min - margin, y_max + margin)

        self.s_line.set_data(t,s)
        self.v_line.set_data(t,v)
        self.a_line.set_data(t,a)
        self.j_line.set_data(t,j)
        #self.dj_line.set_data(t,dj)
        
        
        n_s_max = np.argmax(s)
        n_v_max = np.argmax(v)
        n_a_max = np.argmax(a)
        n_j_max = np.argmax(j)
        n_dj_max = np.argmax(dj)
        
        n_s_min = np.argmin(s)
        n_v_min = np.argmin(v)
        n_a_min = np.argmin(a)
        n_j_min = np.argmin(j)
        n_dj_min = np.argmin(dj)
        
        annos = {"s":[(n_s_max, self.a_max_anno),
                      (n_s_min, self.a_min_anno)],
                 "v":[(n_v_max, self.v_max_anno),
                      (n_v_min, self.v_min_anno)],
                 "a":[(n_a_max, self.s_max_anno),
                      (n_a_min, self.a_min_anno)],
                 "j":[(n_j_max, self.j_max_anno),
                      (n_j_min, self.j_min_anno)],
                 #"dj":[(n_dj_max, self.dj_max_anno),
                 #     (n_dj_min, self.dj_min_anno)]
                 }
         
        lines = {"s":s,
                 "v":v,
                 "a":a,
                 "j":j,
                 #"dj":dj
                 }
         
        #plt.plot([0,1.0],[1.5,1.5], "-.", label="target speed = %3.1fm/s"%(v_max))
        
        anno_artists = []
        
        for key in annos:
            line = lines[key]
            for i in annos[key]:
                #print(i)
                xy = (t[i[0]], line[i[0]])
                i[1].set_text(" %4.2lf"%xy[1])
                #i[1].set_position(xytext=xy, xy=xy)
                i[1].set_position(xy)
        
        print_param(self.param)
        #plt.plot(t,s,color="red",label="s")
        #plt.plot(t,v,color="blue",label="v")
        #plt.plot(t,a,color="green",label="a")
        
        #plt.legend()
        #plt.axis("equal")
        #plt.show()
        if(1):
            return ( self.s_line, 
                     self.v_line, 
                     self.a_line, 
                     self.j_line,
                     #self.dj_line,
                     self.s_max_anno ,
                     self.s_min_anno ,
                     self.v_max_anno ,
                     self.v_min_anno ,
                     self.a_max_anno ,
                     self.a_min_anno ,
                     self.j_max_anno ,
                     self.j_min_anno ,
                     #self.dj_max_anno ,
                     #self.dj_min_anno ,
                     )
        else:
            return self.dj_line,
       

        #return self.s_line,
       
    def __call__(self, i):
        #print("wtf %d"%i)
        self.loop()
        self.counter += 1
        print("-----------iter %d  counter %d"%(i, self.counter))
        print_ps(self.param)
        
        res = self.plot()
        
        return res
        
    def init(self):
        self.s_line, = ax.plot([], [], "o-", color="red",   label="s")
        self.v_line, = ax.plot([], [], "o-", color="blue",  label="v")
        self.a_line, = ax.plot([], [], "o-", color="green", label="a")
        self.j_line, = ax.plot([], [], "o-", color="orange", label="j")
        #self.dj_line, = ax.plot([], [], "o-", color="purple", label="dj")
        self.s_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        self.s_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        self.v_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        self.v_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        self.a_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        self.a_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        self.j_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        self.j_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        #self.dj_max_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        #self.dj_min_anno = ax.annotate("", xytext=(0,0), xy=(0,0))
        
        self.x = np.linspace(0, 1, 200)
        
        if not self.inited:
            self.ax.legend()
            self.inited = True
        
        if(1):
            return ( self.s_line, 
                     self.v_line, 
                     self.a_line, 
                     self.j_line,
                     #self.dj_line,
                     self.s_max_anno ,
                     self.s_min_anno ,
                     self.v_max_anno ,
                     self.v_min_anno ,
                     self.a_max_anno ,
                     self.a_min_anno ,
                     self.j_max_anno ,
                     self.j_min_anno ,
                     #self.dj_max_anno ,
                     #self.dj_min_anno ,
                     )
        else:
            return self.dj_line,
       

        

from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
it = iter(ax)
anim = FuncAnimation(fig, it, frames=np.arange(1), init_func=it.init,
                     interval=5, 
                     blit=True)

#for i in range(1):
#    it.loop()
#it.plot()

plt.show()
