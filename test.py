import numpy as np
from matplotlib import pyplot as plt

v0 = 0.0
a0 = 0.0
j0 = 0.0

v1 = 1.0
s1 = 1.0
T1 = 1.0
a_max = 1.0

p0 = 0
p1 = v0
p2 = a0/2.0
p3 = j0/6.0

def get_powers(t, order):
    ret = [0]*(order+1)
    ret[0] = 1
    for i in range(1, order+1):
        ret[i] = ret[i-1] * t
    return ret

def f_s(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  (  2/3.0 * p5 * Ts[5] + 
                      p4 * Ts[4] +
                      p3 * Ts[3] + 
                      p2 * Ts[2] +
                      p1 * Ts[1] +
                      p0           )
    return res, param
                      
def f_s_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  np.array( [ 2/3.0 * Ts[5],
                       Ts[4],
                       ( 10/3.0 * p5 * Ts[4] +
                              4 * p4 * Ts[3] +
                              3 * p3 * Ts[2] +
                              2 * p2 * Ts[1] +
                                  p1           ) ])
    return res, param

def f_s_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
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
    
def f_fv(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  (  3 * p5 * Ts[4] +
              4 * p4 * Ts[3] +
              3 * p3 * Ts[2] +
              2 * p2 * Ts[1] +
                  p1         - 
                  v1          )
    if res:
        #lv = 0
        raise AssertionError("fv()=%lf, should be 0."%(res))
    return res, param
                                  
def f_fv_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  np.array( [ 3 * Ts[4],
                       4 * Ts[3],
                       (     12 * p5 * Ts[3] +
                             12 * p4 * Ts[2] +
                              6 * p3 * Ts[1] +
                              2 * p2           ) ])
    return res, param
    
def f_fv_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
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
    
def f_fa(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  ( 10 * p5 * Ts[3] +
             12 * p4 * Ts[2] +
              6 * p3 * Ts[1] +
              2 * p2          )
    if res:
        #la = 0
        raise AssertionError("fa()=%lf, should be 0."%(res))
    return res, param

def f_fa_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  np.array( [ 10 * Ts[3],
                       12 * Ts[2],
                       (     30 * p5 * Ts[2] +
                             24 * p4 * Ts[1] +
                              6 * p3           ) ])
    return res, param

def f_fa_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    fa_dTdp5 =       30      * Ts[2]
    fa_dTdp4 =       24      * Ts[1]
    fa_d2T   = (     60 * p5 * Ts[1] +
                     24 * p4           )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, fa_dTdp5 ] ,
                       [        0,        0, fa_dTdp4 ] ,
                       [ fa_dTdp5, fa_dTdp4, fa_d2T   ] ] )
    return res, param

def f_fj(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  ( 20 * p5 * Ts[2] +
             24 * p4 * Ts[1] +
              6 * p3          )
    if res:
        #lj = 0
        raise AssertionError("fj()=%lf, should be 0."%(res))
    return res, param

def f_fj_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  np.array( [ 20 * Ts[2],
                       24 * Ts[1],
                       (     40 * p5 * Ts[1] +
                             24 * p4           ) ])
    return res, param

def f_fj_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    fj_dTdp5 =       40      * Ts[1]
    fj_dTdp4 =       24
    fj_d2T   = (     40 * p5           )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, fj_dTdp5 ] ,
                       [        0,        0, fj_dTdp4 ] ,
                       [ fj_dTdp5, fj_dTdp4, fj_d2T   ] ] )
    return res, param

def f_f1(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  ( 30 * p5 * Ts[1] +
             24 * p4          )
    if res:
        l1 = 0
        m1 = 0
    return res, param

def f_f1_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  np.array( [ 30 * Ts[1],
                       24,
                       (     30 * p5           ) ])
    return res, param

def f_f1_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    f1_dTdp5 =       30
    f1_dTdp4 =        0
    f1_d2T   = (      0                )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, f1_dTdp5 ] ,
                       [        0,        0, f1_dTdp4 ] ,
                       [ f1_dTdp5, f1_dTdp4, f1_d2T   ] ] )
    return res, param

def f_f2(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  ( Ts[1] - T1 )
    if res:
        l2 = 0
        m2 = 0
    return res, param

def f_f2_dx(param):
    res =  np.array( [ 0,
                       0,
                       1 ])
    return res, param

def f_f2_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    f2_dTdp5 =        0
    f2_dTdp4 =        0
    f2_d2T   = (      0                )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, f2_dTdp5 ] ,
                       [        0,        0, f2_dTdp4 ] ,
                       [ f2_dTdp5, f2_dTdp4, f2_d2T   ] ] )
    return res, param
              
def f_f3(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  ( -Ts[1] )
    if res:
        l3 = 0
        m3 = 0
    return res, param

def f_f3_dx(param):
    res =  np.array( [  0,
                        0,
                       -1 ])
    return res, param

def f_f3_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    f3_dTdp5 =        0
    f3_dTdp4 =        0
    f3_d2T   = (      0                )
    
    # the Hessian matrix is symmetric
    res =  np.array( [ [        0,        0, f3_dTdp5 ] ,
                       [        0,        0, f3_dTdp4 ] ,
                       [ f3_dTdp5, f3_dTdp4, f3_d2T   ] ] )
    return res, param
              
def f_f4(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    s_res, param = f_s(param)
    res =  ( s_res -
             1/2.0 * s1     )
    if res:
        l4 = 0
        m4 = 0
    return res, param
                       
def f_f4_dx(param):
    res, param =  f_s_dx(param)
    return res, param

def f_f4_d2x(param):
    res, param = f_s_d2x(param)
    return res, param
                           
def f_f5(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  ( 15/8.0 * p5 * Ts[3] +
                  3 * p4 * Ts[2] +
                  3 * p3 * Ts[1] +
                  2 * p2         - 
                      a_max        )
    if res:
        l5 = 0
        m5 = 0
    return res, param

def f_f5_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    res =  np.array( [ 15/8.0 * Ts[3],
                       3 * Ts[2],
                       ( 45/8.0 * p5 * Ts[2] +
                              6 * p4 * Ts[1] +
                              3 * p3           ) ])
    return res, param

def f_f5_d2x(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
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
    fj, param = f_fj(param)
    
    if(fv or fa or fj):
        raise AssertionError("fv, fa, fj=(%lf, %lf, %lf)"
                             ", should be all 0."%(fv, fa, fj))
    
    res =  np.array( [ 0, 0, 0 ] )
    return res, param
              
def f_ce_dx(param):
    fv_dx = f_fv_dx(param)
    fa_dx = f_fa_dx(param)
    fj_dx = f_fj_dx(param)
    res =  np.concatenate( ( [fv_dx] ,
                             [fa_dx] ,
                             [fj_dx] ) , axis = 0 )
    return res, param
             
def f_ci(param):
    f1, param = f_f1(param)
    f2, param = f_f2(param)
    f3, param = f_f3(param)
    f4, param = f_f4(param)
    f5, param = f_f5(param)
    
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    #lf = [(l1, f1),
    #      (l2, f2),
    #      (l3, f3),
    #      (l4, f4),
    #      (l5, f5)]
    
    # still, they should all be zero, normally.
    fs = [0] * len(filter(lambda x:x, [m1,m2,m3,m4,m5]))
    
    res =  np.array( fs )
    return res, param
                       
def f_ci_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    
    df1, param = f_f1_dx(param)
    df2, param = f_f2_dx(param)
    df3, param = f_f3_dx(param)
    df4, param = f_f4_dx(param)
    df5, param = f_f5_dx(param)
    
    lf = [(m1, df1),
          (m2, df2),
          (m3, df3),
          (m4, df4),
          (m5, df5)]
    
    ci_dx = list(map(lambda y:[y[1]], filter(lambda x:x[0], lf)))
    
    res =  np.concatenate( ( ci_dx ) , axis = 0 )
                             
    return res, param
                                  
def f_c(param):
    ce, param = f_ce(param)
    ci, param = f_ci(param)
    res =  np.concatenate( ( ce ,
                             ci ), axis = 0 )
    return res, param

def f_c_dx(param):
    ce_dx = f_ce_dx(param)
    ci_dx = f_ci_dx(param)
    res =  np.concatenate( ( ce_dx , 
                             ci_dx ), axis = 0 )
    return res, param
                                      
             
def f_L_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    
    s_dx, param  = f_s_dx(param)
    ce_dx, param  = f_ce_dx(param)
    ci_dx, param  = f_ci_dx(param)
    
    le = np.array([lv,la,lj]).reshape((1,3))
    
    lf = [(m1, l1),
          (m2, l2),
          (m3, l3),
          (m4, l4),
          (m5, l5)]
          
    lilist = list(map(lambda y:y[1], filter(lambda x:x[0], lf)))
    
    li = np.array(li).reshape((1,len(lilist)))
    res =  ( s_dx  +
             le.dot(ce_dx) +
             li.dot(ci_dx)   )[0]
    return res, param
             


def f_L_d2x(param):

    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5 = param
    
    res = np.zeros((3,3))
    
    fv_d2x, param = f_fv_d2x(param)
    fa_d2x, param = f_fa_d2x(param)
    fj_d2x, param = f_fj_d2x(param)
    
    res += ( fv_d2x +
             fa_d2x +
             fj_d2x )
    
    lf = ((m1, l1, f_f1_d2x),
          (m2, l2, f_f2_d2x),
          (m3, l3, f_f3_d2x),
          (m4, l4, f_f4_d2x),
          (m5, l5, f_f5_d2x))

    res += reduce(lambda y0,y1:y0+y1, 
                  map(lambda y:y[1]*y[2](param)[0], 
                      filter(lambda x:x[0], lf)))
    
    return res, param
    
init_param = (1e-1,) * 16
p5, p4, lv, la, lj, l1, l2, l3, l4, l5, T, m1, m2, m3, m4, m5 = init_param
m1, m2, m3, m4, m5 = (1,) * 5
T = 0.01
p4 = 1/24.0
p5 = -1/30.0/T-1-0.1
Ts = get_powers(T, 5)
 
#print(Ts)
param = (p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5)

c, param = f_c(param)
L_dx, param = f_L_dx(param)
#print("L_dx=",L_dx)
b = np.concatenate( ( -L_dx,
                      -c ), axis = 0 )
#print("b=",b)
              
L_d2x, param = f_L_d2x(param)
c_dx, param = f_c_dx(param)
#print(c_dx)
#print(L_d2x.shape, c_dx.shape)
A_0 = np.concatenate( ( L_d2x, c_dx.T         ), axis = 1 )
A_1 = np.concatenate( ( c_dx , np.zeros((8,8))), axis = 1 )
A = np.concatenate( ( A_0, A_1 ), axis = 0 )

#print(A.shape,b.shape)
#print(A,"="*10,b)

if(1):
    f = open("test.csv","w")
    for i in range(11):
        for j in range(11):
            f.write("%lf,"%A[i][j])
        f.write(",%lf"%b[i])
        f.write("\n")
    f.close()

x = np.linalg.solve(A,b)

if(0):
    f = open("test.csv","w")
    for i in range(11):
        for j in range(11):
            f.write("%lf,"%A[i][j])
        f.write(",%lf,,"%b[i])
        f.write("%lf,"%x[i])
        f.write("\n")
    f.close()

print(x)

B,d,D = (np.linalg.svd(A))
if(0):
    f = open("test.csv","w")
    for i in range(11):
        for j in range(11):
            f.write("%lf,"%B[i][j])
        f.write(",%lf,,"%d[i])
        for j in range(11):
            f.write("%lf,"%D[i][j])
        f.write("\n")
        
    f.close()
