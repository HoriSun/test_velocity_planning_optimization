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
                         
def f_s_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return np.array( [ 2/3.0 * Ts[5],
                       Ts[4],
                       ( 10/3.0 * p5 * Ts[4] +
                              4 * p4 * Ts[3] +
                              3 * p3 * Ts[2] +
                              2 * p2 * Ts[1] +
                                  p1           ) ])
             
def f_fv_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return np.array( [ 3 * Ts[3],
                       Ts[3],
                       (     12 * p5 * Ts[3] +
                             12 * p4 * Ts[2] +
                              6 * p3 * Ts[1] +
                              2 * p2           ) ])

def f_fa_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return np.array( [ 10 * Ts[3],
                       12 * Ts[2],
                       (     30 * p5 * Ts[2] +
                             24 * p4 * Ts[1] +
                              6 * p3           ) ])

def f_fj_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return np.array( [ 20 * Ts[2],
                       24 * Ts[1],
                       (     40 * p5 * Ts[1] +
                             24 * p4           ) ])

def f_f1_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return np.array( [ 30 * Ts[1],
                       24,
                       (     30 * p5           ) ])

def f_f2_dx(param):
    return np.array( [ 0,
                       0,
                       1 ])

def f_f3_dx(param):
    return np.array( [ 0,
                       0,
                       1 ])
                       
def f_f4_dx(param):
    return f_s_dx(param)

def f_f5_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return np.array( [ 15/8.0 * Ts[3],
                       3 * Ts[2],
                       ( 45/8.0 * p5 * Ts[2] +
                              6 * p4 * Ts[1] +
                              3 * p3           ) ])

def f_ce_dx(param):
    return np.concatenate( ( [f_fv_dx(param)] ,
                             [f_fa_dx(param)] ,
                             [f_fj_dx(param)] ) , axis = 0 )
             
def f_ci_dx(param):
    return np.concatenate( ( [f_f1_dx(param)] ,
                             [f_f2_dx(param)] ,
                             [f_f3_dx(param)] ,
                             [f_f4_dx(param)] ,
                             [f_f5_dx(param)] ) , axis = 0 )
             
def f_L_dx(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    le = np.array([lv,la,lj]).reshape((1,3))
    li = np.array([l1,l2,l3,l4,l5]).reshape((1,5))
    return ( f_s_dx(param)  +
             le.dot(f_ce_dx(param)) +
             li.dot(f_ci_dx(param))   )[0]
             

def f_fv(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return (  3 * p5 * Ts[4] +
              4 * p4 * Ts[3] +
              3 * p3 * Ts[2] +
              2 * p2 * Ts[1] +
                  p1        - 
                  v1          )

def f_fa(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return ( 10 * p5 * Ts[3] +
             12 * p4 * Ts[2] +
              6 * p3 * Ts[1] +
              2 * p2          )

def f_fj(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return ( 20 * p5 * Ts[2] +
             24 * p4 * Ts[1] +
              6 * p3          )

def f_f1(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return ( 30 * p5 * Ts[1] +
             24 * p4          )

def f_f2(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return ( Ts[1] - T1 )
              
def f_f3(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return ( -Ts[1] )
              
def f_f4(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return ( f_s_t(param) -
             1/2.0 * s1     )
             
def f_f5(param):
    p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts = param
    return ( 15/8.0 * p5 * Ts[3] +
                  3 * p4 * Ts[2] +
                  3 * p3 * Ts[1] +
                  2 * p2         - 
                      a_max        )
    
              
def f_ce(param):
    return np.array( [ f_fv(param) , 
                       f_fa(param) , 
                       f_fj(param) ] )
              
def f_ci(param):
    return np.array( [ f_f1(param) , 
                       f_f2(param) , 
                       f_f3(param) , 
                       f_f4(param) , 
                       f_f5(param) ] )
                       
def f_c(param):
    return np.concatenate( ( f_ce(param) ,
                             f_ci(param) ), axis = 0 )

def f_L_dx_dx(param):
    LdT_5 = f_L_dT_dp5(param)
    LdT_4 = f_L_dT_dp4(param)
    LdT_T = f_L_dT_dT(param)
    
    return np.array( [ [     0,     0, LdT_5 ] , 
                       [     0,     0, LdT_4 ] , 
                       [ LdT_5, LdT_4, LdT_T ] ] )
              
init_param = (1e-1,) * 11
p5, p4, lv, la, lj, l1, l2, l3, l4, l5, T = init_param
T = 0.01
p4 = 1/24.0
p5 = -1/30.0/T-1-0.1
Ts = get_powers(T, 5)

#print(Ts)
param = (p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts)

L_dx = f_L_dx(param)
c = f_c(param)
#print("L_dx=",L_dx)
b = np.concatenate( ( -L_dx,
                      -c ), axis = 0 )
#print("b=",b)

def f_c_dx(param):
    return np.concatenate( (f_ce_dx(param), f_ci_dx(param)), axis = 0 )
              
L_d2x = f_L_dx_dx(param)
c_dx = f_c_dx(param)
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
