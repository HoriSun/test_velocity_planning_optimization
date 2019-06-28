from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
import mpmath
import sys
from hl import Color
import matplotlib.animation as ani

clr = Color()

def get_powers(t, order):
    ret = [0]*(order+1)
    ret[0] = 1
    for i in range(1, order+1):
        ret[i] = ret[i-1] * t
    return ret

class StaticParam(object):
    def __init__(self):
        self.v0 = 0.0
        self.a0 = 0.0
        self.j0 = 0.0
        
        self.v1 = 2.0
        self.s1 = 10.0
        self.T1 = 10.0
        self.a_max = 1.0
        
        self.p0 = 0
        self.p1 = self.v0
        self.p2 = self.a0/2.0
        self.p3 = j0/6.0
        
        self.threshold = 1e-2
        

class Param(object):
    
    # hard-coded, must be correct
    ice = {"v":0,"a":1}
    ici = {"1":0,"2":1,"3":2,"4":3}
    nce = ["v","a"]
    nci = ["1","2","3","4"]
    
    def __init__(self, *args):
        #print("construct: %s"%(repr(args)))
        self.init_empty()
        if(not args):
            return
        elif(isinstance(args[0], Param)):
            self.init_param(args[0])
            return
        else:
            self.init_each(args)
            return
            
    def init_empty(self):
        self.p5 = 0.0
        self.p4 = 0.0
        self.set_T(0.0)
        
        self.lce = [0.0]*2
        self.lci = [0.0]*4
        
            
    def init_param(self, p):
        self.copy_from(p)
        
    def copy_from(self,p):
        self.p5 = p.p5
        self.p4 = p.p4
        self.set_T(p.get_T())
        
        self.lce = p.lce[::]
        self.lci = p.lci[::]
        
    def init_each(self, args):
        p5, p4, T, lv, la, l1, l2, l3, l4 = args
        
        self.p5 = p5
        self.p4 = p4
        self.set_T(T)
        
        self.lce = [lv, la]
        self.lci = [l1, l2, l3, l4]
        
    def set_T(self, T):
        self.__T = T
        self.__Ts = get_powers(self.__T, 6)
        
    def get_T(self):
        return self.__T
        
    def get_Ts(self):
        return self.__Ts
        
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return ("( "
                "p5=%lf, p4=%lf, "
                "lv=%lf, la=%lf, "
                "l1=%lf, l2=%lf, l3=%lf, l4=%lf )"
                ""%(self.p5, self.p4,
                    self.lce[0], self.lce[1],
                    self.lci[0], self.lci[1], self.lci[2], self.lci[3]))
          
    def copy(self):
        return Param(self)
        
class poly_vel_profile(object):
    def __init__(self):
        sp = StaticParam()
        sp.v0 = 0.0
        sp.a0 = 0.0
        sp.j0 = 0.0
        
        sp.v1 = 2.0
        sp.s1 = 10.0
        sp.T1 = 10.0
        sp.a_max = 1.0
        
        sp.p0 = 0
        sp.p1 = sp.v0
        sp.p2 = sp.a0/2.0
        sp.p3 = j0/6.0
        
        sp.threshold = 1e-2
        
        self.static_param = sp
        
        self.var = Param()
        
        self.ci_mask = [1] * 4  # 1 for activated, 0 for deactivated
        
    def get_n_ci_activated(self):
        return len(list(filter(lambda x:x, self.ci_mask)))
        
        
    def f_s(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  (  2/3.0 * v.p5 * Ts[5] + 
                          v.p4 * Ts[4] +
                          s.p3 * Ts[3] + 
                          s.p2 * Ts[2] +
                          s.p1 * Ts[1] +
                          s.p0           )
        
        return res


        
    def f_s_dx(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  np.array( [ 2/3.0 * Ts[5],
                           Ts[4],
                           ( 10/3.0 * v.p5 * Ts[4] +
                                  4 * v.p4 * Ts[3] +
                                  3 * s.p3 * Ts[2] +
                                  2 * s.p2 * Ts[1] +
                                      s.p1           ) ])
                                      
        return res

        
        
    def f_s_d2x(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        s_dTdp5 =   10/3.0        * Ts[4]
        s_dTdp4 =        4        * Ts[3]
        s_d2T   = ( 40/3.0 * v.p5 * Ts[3] + 
                        12 * v.p4 * Ts[2] +
                         6 * s.p3 * Ts[1] +
                         2 * s.p2           )
        
        # the Hessian matrix is symmetric
        res =  np.array( [ [       0,       0, s_dTdp5 ] ,
                           [       0,       0, s_dTdp4 ] ,
                           [ s_dTdp5, s_dTdp4, s_d2T   ] ] )
        return res

        
        
    def f_fv(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  (  3 * v.p5 * Ts[4] +
                  4 * v.p4 * Ts[3] +
                  3 * s.p3 * Ts[2] +
                  2 * s.p2 * Ts[1] +
                      s.p1         - 
                      s.v1           )
        
        if np.abs(res)>=threshold:
            #err_handle()
            clr.print_red_text("fv()=%20.18lf, should be 0."%(res))
        return res
                                      
                                      
    def f_fv_dx(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  np.array( [ 3 * Ts[4],
                           4 * Ts[3],
                           (     12 * v.p5 * Ts[3] +
                                 12 * v.p4 * Ts[2] +
                                  6 * s.p3 * Ts[1] +
                                  2 * s.p2           ) ])
        return res
        
    def f_fv_d2x(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        fv_dTdp5 =       12        * Ts[3]
        fv_dTdp4 =       12        * Ts[2]
        fv_d2T   = (     36 * v.p5 * Ts[2] +
                         24 * v.p4 * Ts[1] + 
                          6 * s.p3           )
        
        # the Hessian matrix is symmetric
        res =  np.array( [ [        0,        0, fv_dTdp5 ] ,
                           [        0,        0, fv_dTdp4 ] ,
                           [ fv_dTdp5, fv_dTdp4, fv_d2T   ] ] )
        return res
        
        
    def f_fa(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  ( 10 * v.p5 * Ts[3] +
                 12 * v.p4 * Ts[2] +
                  6 * s.p3 * Ts[1] +
                  2 * s.p2          )
        
        if np.abs(res)>=threshold:
            #la = 0
            #err_handle()
            clr.print_red_text("fa()=%20.18lf, should be 0."%(res))
        return res

    def f_fa_dx(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  np.array( [ 10 * Ts[3],
                           12 * Ts[2],
                           (     30 * v.p5 * Ts[2] +
                                 24 * v.p4 * Ts[1] +
                                  6 * s.p3           ) ])
        return res

    def f_fa_d2x(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        fa_dTdp5 =       30        * Ts[2]
        fa_dTdp4 =       24        * Ts[1]
        fa_d2T   = (     60 * v.p5 * Ts[1] +
                         24 * v.p4           )
        
        # the Hessian matrix is symmetric
        res =  np.array( [ [        0,        0, fa_dTdp5 ] ,
                           [        0,        0, fa_dTdp4 ] ,
                           [ fa_dTdp5, fa_dTdp4, fa_d2T   ] ] )
        return res

    def f_fj(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  ( 20 * v.p5 * Ts[2] +
                 24 * v.p4 * Ts[1] +
                  6 * s.p3           )
        if res!=0:
            #lj = 0
            #err_handle()
            clr.print_red_text("fj()=%lf, should be 0."%(res))
        return res

    def f_fj_dx(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  np.array( [ 20 * Ts[2],
                           24 * Ts[1],
                           (     40 * v.p5 * Ts[1] +
                                 24 * v.p4           ) ])
        return res

        
    def f_fj_d2x(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        fj_dTdp5 =       40        * Ts[1]
        fj_dTdp4 =       24
        fj_d2T   = (     40 * v.p5           )
        
        # the Hessian matrix is symmetric
        res =  np.array( [ [        0,        0, fj_dTdp5 ] ,
                           [        0,        0, fj_dTdp4 ] ,
                           [ fj_dTdp5, fj_dTdp4, fj_d2T   ] ] )
        return res

        
    def f_f1(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        mask = 1
        res =  ( 30 * v.p5 * Ts[1] +
                 24 * v.p4          )
        if res!=0:
            if res > 0:
                clr.print_red_text("f1=%32.30lf > 0"%res)
            l1 = 0
            mask = 0
        
        self.ci_mask[0] = mask
        return res

    def f_f1_dx(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        res =  np.array( [ 30 * Ts[1],
                           24,
                           (     30 * v.p5           ) ])
        return res

    def f_f1_d2x(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        f1_dTdp5 =       30
        f1_dTdp4 =        0
        f1_d2T   = (      0                )
        
        # the Hessian matrix is symmetric
        res =  np.array( [ [        0,        0, f1_dTdp5 ] ,
                           [        0,        0, f1_dTdp4 ] ,
                           [ f1_dTdp5, f1_dTdp4, f1_d2T   ] ] )
        return res

                  
    def f_f2(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        mask = 1
        res =  ( Ts[2] - s.T1*Ts[1] )
        if res!=0:
            if res > 0:
                clr.print_red_text("f2=%32.30lf > 0"%res)
            l3 = 0
            mask = 0
        self.ci_mask[1] = mask
        return res

    def f_f2_dx(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        res =  np.array( [  0,
                            0,
                            2*Ts[1] - s.T1 ])
        return res

    def f_f2_d2x(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        f2_dTdp5 =        0
        f2_dTdp4 =        0
        f2_d2T   = (      2                )
        
        # the Hessian matrix is symmetric
        res =  np.array( [ [        0,        0, f2_dTdp5 ] ,
                           [        0,        0, f2_dTdp4 ] ,
                           [ f2_dTdp5, f2_dTdp4, f2_d2T   ] ] )
        return res
                  
    def f_f3(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        m4 = 1
        s_res, param = self.f_s(v)
        res =  ( s_res -
                 1/2.0 * s1     )
        if res!=0:
            if res > 0:
                clr.print_red_text("f3=%32.30lf > 0"%res)
            l4 = 0
            m4 = 0
        param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
        return res
                           
    def f_f3_dx(self, v):
        res, param =  f_s_dx(param)
        return res

    def f_f3_d2x(self, v):
        res, param = f_s_d2x(param)
        return res
                               
    def f_f4(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        m5 = 1
        def f4(p5):
            return ( 15/8.0 * v.p5 * Ts[3] +
                          3 * v.p4 * Ts[2] +
                          3 * s.p3 * Ts[1] +
                          2 * s.p2         - 
                              a_max        )
        res = f4(p5)
        print("-----------0 f_f4()  p5=%32.30lf   res=%32.30lf\n"%(p5, res))
                              
        if res > 0:
        
            #p5 = (a_max - 2*p2 - 3*p3*Ts[1] - 3*p4*Ts[2]) / (15/8.0*Ts[3])
        
            #clr.print_red_text("f4=%32.30lf > 0"%res)
            #counter = 0
            while np.abs(res) > 1e-13 and res > 0 and False:
                dp5 = -res / (15/8.0*Ts[3])
                v.p5 += dp5
                #counter += 1
                res = f4()
                print("************** f_f4()  p5=%32.30lf   res=%32.30lf\n"%(p5, res))
                #if(counter >= 3):
                #    sys.exit(0)
            else:
                pass
                #sys.exit(0)
            res = f4(p5)
            print("-----------1 f_f4()  p5=%32.30lf   res=%32.30lf\n"%(p5, res))
                
                          
        if np.abs(res) > 1e-7:
            if res > 0:
                clr.print_red_text("f4=%32.30lf > 0"%res)
            l5 = 0
            m5 = 0
            #clr.print_red_text
        print("f_f4()  m5=%s"%(repr(m5)))
        param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
        return res

    def f_f4_dx(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        res =  np.array( [ 15/8.0 * Ts[3],
                           3 * Ts[2],
                           ( 45/8.0 * v.p5 * Ts[2] +
                                  6 * v.p4 * Ts[1] +
                                  3 * s.p3           ) ])
        return res

    def f_f4_d2x(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        f4_dTdp5 =   45/8.0 * Ts[2]
        f4_dTdp4 =        6 * Ts[1]
        f4_d2T   = ( 45/4.0 * v.p5 * Ts[1] +
                          6 * v.p4           )
        
        # the Hessian matrix is symmetric
        res =  np.array( [ [        0,        0, f4_dTdp5 ] ,
                           [        0,        0, f4_dTdp4 ] ,
                           [ f4_dTdp5, f4_dTdp4, f4_d2T   ] ] )
        return res
                      
    def f_ce(self, v):

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
            #err_handle()
            clr.print_red_text("fv, fa=(%lf, %lf)"
                                 ", should be all 0."%(fv, fa))
        
        res =  np.array( [ 0, 0 ] )
        return res
                  
    def f_ce_dx(self, v):

        fv_dx, param = f_fv_dx(param)
        fa_dx, param = f_fa_dx(param)
        #fj_dx, param = f_fj_dx(param)#
        res =  np.concatenate( ( [fv_dx] ,
                                 [fa_dx] ,
                                 #[fj_dx] 
                                         ) , axis = 0 )
        return res

        
    def f_ci(self, v):
        
        f1, param = f_f1(param)
        f2, param = f_f2(param)
        f3, param = f_f3(param)
        f4, param = f_f4(param)
        
        s = self.static_param
        Ts = v.get_Ts()
        
        print("f_ci()  f1=%lf f2=%lf f3=%lf f4=%lf"
              ""%(f1, f2, f3, f4))
        print("f_ci()  m1=%d m2=%d m3=%d m4=%d m5=%d\n"
              ""%(m1, m2, m3, m4))
        
        #lf = [(l1, f1),
        #      (l2, f2),
        #      (l3, f3),
        #      (l4, f4)]
        
        # still, they should all be zero, normally.
        
        ni,param = get_ni(param)
        
        print("f_ci()  f1=%lf f2=%lf f3=%lf f4=%lf"
              ""%(f1, f2, f3, f4))
        print("f_ci()  m1=%d m2=%d m3=%d m4=%d\n"
              ""%(m1, m2, m3, m4))
        
        if(ni):
            fs = [0] * ni
        else:
            fs = []
        
        #print_param(param)
        
        res =  np.array( fs )
        
        param = p5, p4, lv, la, lj, l1, l2, l3, l4, l5, Ts, m1, m2, m3, m4, m5, ni
        return res
        
        
    def f_ci_dx(self, v):

        df1, param = f_f1_dx(param)
        df2, param = f_f2_dx(param)
        df3, param = f_f3_dx(param)
        df4, param = f_f4_dx(param)
        
        ni,param = get_ni(param)
        
        if(ni):
            s = self.static_param
            Ts = v.get_Ts()
            
            lf = [(m1, df1),
                  (m2, df2),
                  (m3, df3),
                  (m4, df4)]
            
            ci_dx = list(map(lambda y:[y[1]],filter(lambda x:x[0],lf)))
            
            res =  np.concatenate( ( ci_dx ) , axis = 0 )
        else:
            res =  np.array([[]])
                                 
        return res
                                      
    def f_c(self, v):
        ce, param = f_ce(param)
        ci, param = f_ci(param)
        
        ni,param = get_ni(param)
        #print(ce, ci, ni)
        
        if(ni):
            res =  np.concatenate( ( ce ,
                                     ci ), axis = 0 )
        else:
            res =  ce
            
        s = self.static_param
        Ts = v.get_Ts()
        print("f_c()  m1=%d m2=%d m3=%d m4=%d\n"
              ""%(m1, m2, m3, m4))
            
            
        return res

    def f_c_dx(self, v):
        ce_dx, param = f_ce_dx(param)
        ci_dx, param = f_ci_dx(param)
        
        ni,param = get_ni(param)
        print("f_c_dx() ", ce_dx, ci_dx)
        if(ni):
            res =  np.concatenate( ( ce_dx , 
                                     ci_dx ), axis = 0 )
        else:
            res = ce_dx
            
        return res
                                          
                 
    def f_L_dx(self, v):



        s_dx, param  = f_s_dx(param)
        ce_dx, param  = f_ce_dx(param)
        ci_dx, param  = f_ci_dx(param)

        s = self.static_param
        Ts = v.get_Ts()

        #print_param(param)
        
        le = np.array([lv,
                       la,
                       #lj
                         ]).reshape((1,2))
        
        lf = [(m1, l1),
              (m2, l2),
              (m3, l3),
              (m4, l4)]
              
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
        
        return res
                 


    def f_L_d2x(self, v):

        res = np.zeros((3,3))
        
        fv_d2x, param = f_fv_d2x(param)
        fa_d2x, param = f_fa_d2x(param)
        #fj_d2x, param = f_fj_d2x(param)
        
        s = self.static_param
        Ts = v.get_Ts()
            
        res += ( fv_d2x +
                 fa_d2x
                 #fj_d2x 
                         )
        
        lf = ((m1, l1, f_f1_d2x),
              (m2, l2, f_f2_d2x),
              (m3, l3, f_f3_d2x),
              (m4, l4, f_f4_d2x))

        ni,param = get_ni(param)
        
        if(ni):
            res += reduce(lambda y0,y1:y0+y1, 
                          map(lambda y:y[1]*y[2](param)[0], 
                              filter(lambda x:x[0], lf)))
        
        return res
        

        
        
class test(object):
    def __init__(self):
        self.var = Param()
        print(self.var)
        
        self.mod_in(self.var.copy())
        
        print(self.var)
        
    def mod_in(self,v):
        v.p5 = 100

        
test()