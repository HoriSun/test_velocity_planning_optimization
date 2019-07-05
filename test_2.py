from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
import mpmath
import sys
from hl import Color
import matplotlib.animation as ani
from collections import Iterable
import threading

clr = Color()


def start_daemon(target):
    t = threading.Thread(target = target)
    t.setDaemon(True)
    t.start()
    return t


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
        self.p3 = self.j0/6.0
        
        self.threshold = 1e-10
        

class Param(object):
    
    # hard-coded, must be correct
    ice = {"v":0,"a":1}
    ici = {"1":0,"2":1,"3":2,"4":3}
    nce = ["v","a"]
    nci = ["1","2","3","4"]
    
    
    class DummyList(object):
        '''Access other stuffs with callbacks and a bracket-get-set way'''
        def __init__(self):
            self.get_cb = []
            self.set_cb = []
            self.counter = 0
            
        def register_item_cb(self, index, 
                             getcb, setcb):
            if(index!=self.counter):
                raise AssertionError("index(%d) != self.counter(%d)"
                                     ""%(index, self.counter))
            else:
                self.counter += 1
            #self.get_cb[index] = getcb
            #self.set_cb[index] = setcb
            self.get_cb.append( getcb )
            self.set_cb.append( setcb )
            
        def __getitem__(self, slice):
            #print("getitem %s"%(slice))
            #print("get %s %s"%(slice))
            funs = self.get_cb[slice]
            if(not isinstance(funs, list)):
                return funs()
            else:
                return list(map(lambda x:x(),funs))
            
        def __setitem__(self, slice, values):
            #print("setitem %s %s"%(slice, values))
            funs = self.set_cb[slice]
            
            if(not isinstance(funs, list)):
                funs = [funs]
            if(not isinstance(values, Iterable)):
                values = [values]
            
            for i in range(len(funs)):
                funs[i](values[i])
            
            
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
        self.attr_override = {}
        
        self.p5 = 0.0
        self.p4 = 0.0
        self.set_T(0.0)
        
        self.la = 0.0
        self.lv = 0.0
        
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0
        self.l4 = 0.0
        
        getlv = lambda:self.lv
        getla = lambda:self.la
        
        getl1 = lambda:self.l1
        getl2 = lambda:self.l2
        getl3 = lambda:self.l3
        getl4 = lambda:self.l4
        
        def setlv(x): self.lv = x
        def setla(x): self.la = x
        
        def setl1(x): self.l1 = x
        def setl2(x): self.l2 = x
        def setl3(x): self.l3 = x
        def setl4(x): self.l4 = x
        
        self.lce = Param.DummyList()
        self.lce.register_item_cb(0, getlv, setlv)
        self.lce.register_item_cb(1, getla, setla)
        
        self.lci = Param.DummyList()
        self.lci.register_item_cb(0, getl1, setl1)
        self.lci.register_item_cb(1, getl2, setl2)
        self.lci.register_item_cb(2, getl3, setl3)
        self.lci.register_item_cb(3, getl4, setl4)
            
        self.attr_override["lce"] = {"post_proc": lambda x:x[::]}
        self.attr_override["lci"] = {"post_proc": lambda x:x[::]}
        self.attr_override["lce__"] = {"alter_name":"lce"}
        self.attr_override["lci__"] = {"alter_name":"lci"}
        
    def __delattr__(self, *args, **kwargs):
        print('delattr(%s,%s), do nothing'%(args,kwargs))
        #return object.__delattr__(self, *args, **kwargs)
        return None

    def __getattribute__(self, *args, **kwargs):
        name = args[0]
        #print("getattr: (%s)"%(args))
        #print("(%s)=='lce'"%(name=="lce"))
        #print("(%s)=='lci'"%(name=="lci"))
        #print("wtf?"*10)
        
        if(isinstance(name, str)):
            
            attr_override = {}
            try:
                attr_override = object.__getattribute__(self, "attr_override")
                #print(type(attr_override))
            except AttributeError as e:
                pass
            
            if (name in attr_override):
                o = attr_override[name]
                if("alter_name" in o):
                    alter_name = o["alter_name"]
                else:
                    alter_name = name
                    
                res = object.__getattribute__(self, alter_name)
                
                if("post_proc" in o):
                    return attr_override[name]["post_proc"](res)
                else:
                    return res
            else:
                res = object.__getattribute__(self, *args, **kwargs)
                return res
        return res
        
        
    def __setattr__(self, *args, **kwargs):
        #print('setattr(%s,%s)'%(args,kwargs))
        name, obj = args
        if(name == "lce"):
            if(not hasattr(self,name)):
                return object.__setattr__(self, *args, **kwargs)
            else:
                clr.print_red_text("resetting lce as (%s)"%(repr(obj)))
                if(isinstance(obj,Iterable)):
                    if(len(obj) != self.lce__.counter):
                        clr.print_red_text("object(%s) to reset has invalid "
                                           "length(%d), should be (%d). "
                                           "do nothing."
                                           ""%(repr(obj), len(obj), self.lce__.counter))
                        return None
                    else:
                        self.lce__[::] = obj
                else:
                    clr.print_red_text("object(%s) to reset is not iterable, "
                                       "do nothing"%(repr(obj)))
                    return None
        
        elif(name == "lci"):
            if(not hasattr(self,name)):
                return object.__setattr__(self, *args, **kwargs)
            else:
                clr.print_red_text("resetting lci as (%s)"%(repr(obj)))
                if(isinstance(obj,Iterable)):
                    if(len(obj) != self.lci__.counter):
                        clr.print_red_text("object(%s) to reset has invalid "
                                           "length(%d), should be (%d). "
                                           "do nothing."
                                           ""%(repr(obj), len(obj), self.lci__.counter))
                        return None
                    else:
                        self.lci__[::] = obj
                else:
                    clr.print_red_text("object(%s) to reset is not iterable, "
                                       "do nothing"%(repr(obj)))
                    return None
        else:
            return object.__setattr__(self, *args, **kwargs)
        
    def init_param(self, p):
        self.copy_from(p)
        
    def copy_from(self,p):
        self.p5 = p.p5
        self.p4 = p.p4
        self.set_T(p.get_T())
        
        self.lv = p.lv
        self.la = p.la
        self.l1 = p.l1
        self.l2 = p.l2
        self.l3 = p.l3
        self.l4 = p.l4
        
        
    def init_each(self, args):
        p5, p4, T, lv, la, l1, l2, l3, l4 = args
        
        self.p5 = p5
        self.p4 = p4
        self.set_T(T)
        
        self.lv = lv
        self.la = la
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        
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
        if(0):
            return ("( "
                    "p5=%32.30lf, p4=%32.30lf, T=%32.30lf, "
                    "lv=%32.30lf, la=%32.30lf, "
                    "l1=%32.30lf, l2=%32.30lf, l3=%32.30lf, l4=%32.30lf )"
                    ""%(self.p5, self.p4, self.get_T(),
                        self.lv, self.la,
                        self.l1, self.l2, self.l3, self.l4))
        else:
            return ("( %s )"%(", ".join(["% 9.6lf"]*9))%(
                        self.p5, self.p4, self.get_T(),
                        self.lv, self.la,
                        self.l1, self.l2, self.l3, self.l4))
              
    def copy(self):
        return Param(self)
        
        
        
        
        
        
        
        
class PolyVelProfile(object):

    class RegionEnumType(object):
        '''Immutable enumerations'''
        __INSIDE = -1
        __ON_EDGE = 0
        __OUTSIDE = 1
        
        @property
        def INSIDE(self):
            return type(self).__INSIDE
        @property
        def ON_EDGE(self):
            return type(self).__ON_EDGE
        @property
        def OUTSIDE(self):
            return type(self).__OUTSIDE
        
        @classmethod
        def __setattr__(cls, *args, **kwargs):
            return
        def __str__(self):
            return self.__repr__()
        def __repr__(self):
            return ("INSIDE(%s) ON_EDGE(%s) OUTSIDE(%s)"
                    ""%(PolyVelProfile.RegionEnum.INSIDE,
                        PolyVelProfile.RegionEnum.ON_EDGE,
                        PolyVelProfile.RegionEnum.OUTSIDE))
                        
    RegionEnum = RegionEnumType()

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
        sp.p3 = sp.j0/6.0
        
        sp.threshold = 1e-10
        
        self.static_param = sp
        
        self.var = Param()
        
        self.ci_mask = [0] * 4  # 1 for activated, 0 for deactivated
        
        #self.run(self.var)
    
    def run(self, v):
        self.init_var(v)
    
        
    def get_n_ci_activated(self, masks):
        return len(list(filter(lambda x:x, masks)))
        
        
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

        
    def f_f1(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        res =  ( 30 * v.p5 * Ts[1] +
                 24 * v.p4          )
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
        
        res =  ( Ts[2] - s.T1*Ts[1] )
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
        
        s_res = self.f_s(v)
        res =  ( s_res -
                 1/2.0 * s.s1     )
        return res
        
        
    def f_f3_dx(self, v):
        res = self.f_s_dx(v)
        return res

        
    def f_f3_d2x(self, v):
        res = self.f_s_d2x(v)
        return res
        
        
    def f_f4(self, v):
        s = self.static_param
        Ts = v.get_Ts()
        
        def f4(p5):
            return ( 15/8.0 * v.p5 * Ts[3] +
                          3 * v.p4 * Ts[2] +
                          3 * s.p3 * Ts[1] +
                          2 * s.p2         - 
                              s.a_max        )
        res = f4(v.p5)
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
        fv = self.f_fv(v)
        fa = self.f_fa(v)
        #fj, param = self.f_fj(param)
        
        #print("fv=%lf fa=%lf\n"
        #      ""%(fv, fa))
              
        res = np.array([fv, fa])
              
        #fva_list.append([fv,fa])
        
        #if(any(map(lambda x:np.abs(x)>=threshold,
        #           res))):
        #    #err_handle()
        #    clr.print_red_text("fv, fa=(%lf, %lf)"
        #                       ", should be all 0."%(fv, fa))
        
        #res =  np.array( [ 0, 0 ] )
        return res
                  
                  
    def f_ce_dx(self, v):

        fv_dx = self.f_fv_dx(v)
        fa_dx = self.f_fa_dx(v)
        res =  np.concatenate( ( [fv_dx] ,
                                 [fa_dx] ,
                                 #[fj_dx] 
                                         ) , axis = 0 )
        return res

        
    def f_ci(self, v, mask=None, force_all=False):
        
        s = self.static_param
        Ts = v.get_Ts()
        
        f1 = self.f_f1(v)
        f2 = self.f_f2(v)
        f3 = self.f_f3(v)
        f4 = self.f_f4(v)
        
        #print("f_ci()  ( %lf  %lf  %lf  %lf )"
        #      ""%(f1, f2, f3, f4))
        #print("f_ci()  m1=%d m2=%d m3=%d m4=%d\n"
        #      ""%mask)
        
        #lf = [(l1, f1),
        #      (l2, f2),
        #      (l3, f3),
        #      (l4, f4)]
        
        # still, they should all be zero, normally.
        
        if(not mask):
            mask = [0,0,0,0]
            
        def not_activated(a, threshold):
            #return np.abs(a)>threshold
            return a < 0-threshold
            
        threshold = 0.0
            
        mask[0] = 0 if not_activated(f1, threshold) else 1
        mask[1] = 0 if not_activated(f2, threshold) else 1
        mask[2] = 0 if not_activated(f3, threshold) else 1
        mask[3] = 0 if not_activated(f4, threshold) else 1
        
        fs = []
        
        if(not force_all):
            if(mask[0]):
                fs.append(f1)
            if(mask[1]):
                fs.append(f2)
            if(mask[2]):
                fs.append(f3)
            if(mask[3]):
                fs.append(f4)
        else:
            fs = [f1,f2,f3,f4]
                
        #print_param(param)
        
        res =  np.array( fs )
        
        return res
        
        
    def f_ci_dx(self, v, mask=None):

        df1 = self.f_f1_dx(v)
        df2 = self.f_f2_dx(v)
        df3 = self.f_f3_dx(v)
        df4 = self.f_f4_dx(v)
       
        
        if(not mask):
            mask = [0,0,0,0]
            
        ni = self.get_n_ci_activated(mask)
        if(not ni):
            ci_dx = None
        else:
            ci_dx = []
            if(mask[0]):
                ci_dx.append([df1])
            if(mask[1]):
                ci_dx.append([df2])
            if(mask[2]):
                ci_dx.append([df3])
            if(mask[3]):
                ci_dx.append([df4])

                
        if(not ci_dx):
            res = np.array([[]])
        else:
            res =  np.concatenate( ( ci_dx ) , axis = 0 )
                                
        return res
            
            
    def f_c(self, v, imask=None):
        ce = self.f_ce(v)
        ci = self.f_ci(v,imask)
        
        #print("f_c()  m1=%d m2=%d m3=%d m4=%d\n"
        #      ""%tuple(imask))

        if(imask):
            ni = self.get_n_ci_activated(imask)
        else:
            ni = 4
        
        if(ni):
            res =  np.concatenate( ( ce ,
                                     ci ), axis = 0 )
        else:
            res =  ce
            
            
            
        return res

    def f_c_dx(self, v, imask=None):
        ce_dx = self.f_ce_dx(v)
        ci_dx = self.f_ci_dx(v,imask)
        
        if(imask):
            ni = self.get_n_ci_activated(imask)
        else:
            ni = 4
        
        #print("f_c_dx() =========\n%s\n%s"%(ce_dx, ci_dx))
        
        if(ni):
            res =  np.concatenate( ( ce_dx , 
                                     ci_dx ), axis = 0 )
        else:
            res = ce_dx
            
        return res
                                          
                 
    def f_L_dx(self, v, imask=None):

        s_dx   = self.f_s_dx(v)
        ce_dx  = self.f_ce_dx(v)
        ci_dx  = self.f_ci_dx(v, imask)

        s = self.static_param
        Ts = v.get_Ts()

        #print_param(param)
        
        le = np.array([v.lv,
                       v.la,
                       #lj
                         ]).reshape((1,2))
        
        if(not imask):
            imask = [0,0,0,0]
        
        lf = [(imask[0], v.l1),
              (imask[1], v.l2),
              (imask[2], v.l3),
              (imask[3], v.l4)]
              
        lilist = list(map(lambda y:y[1], filter(lambda x:x[0], lf)))
        
        li = np.array(lilist).reshape((1,len(lilist)))
        
        if(0):
            print("s_dx=%s"%s_dx)
        
            print("le=%s"%le)
            print("li=%s"%li)
        
            print("ce_dx=%s"%ce_dx)
            print("ci_dx=%s"%ci_dx)
        
        res = s_dx + le.dot(ce_dx)
        ni = self.get_n_ci_activated(imask)
        
        if(ni):
            res += li.dot(ci_dx)
            
        res = np.array(res)
        
        return res
                 


    def f_L_d2x(self, v, imask):

        res = np.zeros((3,3))
        
        fv_d2x = self.f_fv_d2x(v)
        fa_d2x = self.f_fa_d2x(v)
        
        s = self.static_param
        Ts = v.get_Ts()
            
        res += ( fv_d2x +
                 fa_d2x
                 #fj_d2x 
                         )
        
        if not imask:
            imask = [0,0,0,0]
        
        lf = ((imask[0], v.l1, self.f_f1_d2x),
              (imask[1], v.l2, self.f_f2_d2x),
              (imask[2], v.l3, self.f_f3_d2x),
              (imask[3], v.l4, self.f_f4_d2x))

        ni = self.get_n_ci_activated(imask)
        
        if(ni):
            res += reduce(lambda y0,y1:y0+y1, 
                          map(lambda y:y[1]*y[2](v)[0], 
                              filter(lambda x:x[0], lf)))
        
        return res
        
    
    def f_fv_check(self, v, threshold=0.0):
        res = (np.abs(v)>=threshold)
        if res:
            clr.print_red_text("fv()=%20.18lf, should be 0."%(res))
        return res
        
        
    def f_fa_check(self, a, threshold=0.0):
        res = (np.abs(a)>=threshold)
        if res:
            clr.print_red_text("fa()=%20.18lf, should be 0."%(res))
        return res
        
    def f_f1_check(self, f1, threshold=0.0):
        mask = 1
        
        diff = np.abs(f1)
        res = (diff>=threshold)
        
        state = PolyVelProfile.RegionEnum.EDGE
        
        if res:
            if f1 > 0:
                clr.print_red_text("f1=%32.30lf > 0"%f1)
                state = PolyVelProfile.RegionEnum.OUTSIDE
            else:
                #self.l1 = 0
                mask = 0
                state = PolyVelProfile.RegionEnum.INSIDE
        else:
            state = PolyVelProfile.RegionEnum.EDGE
        
        return state, mask
        
        
        
    def f_f2_check(self, f2, threshold=0.0):
        mask = 1
        
        diff = np.abs(f2)
        res = (diff>=threshold)
        
        state = PolyVelProfile.RegionEnum.EDGE
        
        if res:
            if f2 > 0:
                clr.print_red_text("f2=%32.30lf > 0"%f2)
                state = PolyVelProfile.RegionEnum.OUTSIDE
            else:
                #self.l2 = 0
                mask = 0
                state = PolyVelProfile.RegionEnum.INSIDE
        else:
            state = PolyVelProfile.RegionEnum.EDGE
        
        
        return state, mask
        
        
    def f_f3_check(self, f3):
        mask = 1
        
        diff = np.abs(f3)
        res = (diff>=threshold)
        
        state = PolyVelProfile.RegionEnum.EDGE
        
        if res:
            if f3 > 0:
                clr.print_red_text("f3=%32.30lf > 0"%f3)
                state = PolyVelProfile.RegionEnum.OUTSIDE
            else:
                #self.l1 = 0
                mask = 0
                state = PolyVelProfile.RegionEnum.INSIDE
        else:
            state = PolyVelProfile.RegionEnum.EDGE
        
        
        return state, mask
        
        
    def f_f4_check(self, f4):
        mask = 1
        
        diff = np.abs(f4)
        res = (diff>=threshold)
        
        state = PolyVelProfile.RegionEnum.EDGE
        
        if res:
            if f4 > 0:
                clr.print_red_text("f4=%32.30lf > 0"%f4)
                state = PolyVelProfile.RegionEnum.OUTSIDE
            else:
                #self.l1 = 0
                mask = 0
                state = PolyVelProfile.RegionEnum.INSIDE
        else:
            state = PolyVelProfile.RegionEnum.EDGE
        
        
        return state, mask
        
    
    def init_var(self, v):
        s = self.static_param
        
        v.lce = [1.0] * 2
        v.lci = [1.0] * 4
        
        T = 2.0
        v.set_T(T)
        Ts = v.get_Ts()
        
        dv = s.v1 - s.v0
        
        A_ = np.array( [ [ 3 * Ts[4], 4 * Ts[3] ] ,
                         [ 5 * Ts[4], 6 * Ts[3] ] ] )
        b_ = np.array( [ dv, 0 ] )
        v.p5, v.p4 = np.linalg.solve( A_, b_ )
        
        
        #print("v: %s"%(v))
        
        is_inside = self.check_var_inside_inequality_constraint( v )
        if not is_inside:
            self.move_var_inside_inequality_constraint( v )
            
        #print("v: %s"%(v))
        
        
    def check_var_inside_inequality_constraint(self, v):
        clr.print_green_text("check_var_inside_inequality_constraint(%s)"%(v))
        mask = self.ci_mask[::]
        #print("mask: %s"%(mask))
        c = self.f_c(v, mask)
        #print("c: %s"%(c))
        #print("mask: %s"%(mask))
        
        if(self.get_n_ci_activated(mask)):
            while(self.get_n_ci_activated(mask)):
            #if(self.get_n_ci_activated(mask)):
                #clr.print_blue_text("+"*20)
                #print("v before iter: %s"%(v))
        
                dv = self.iter(v, mask)
                c = self.f_c(v, mask)
                clr.print_blue_text(v)
                #print("c=%s"%(c))
                #print("mask=%s"%(mask))
                
                #print("v after iter: %s"%(v))
        
                #self.var_add(v, dv, mask)
                
        return True
        
    def move_var_inside_inequality_constraint(self, v):
        pass
        
        
    def var_add(self, v, dv, mask):
        v.p5 += dv[0]
        v.p4 += dv[1]
        v.set_T( v.get_T() + dv[2] )
        
        v.lv += dv[3]
        v.la += dv[4]
        
        if(self.get_n_ci_activated(mask)):
            j = 5
            k = 0
            for i in range(4):
                if mask[i]:
                    v.lci[k] += dv[j]
                    i+=1
                    j+=1
                    
    def iter(self, v, mask):
        #print("====iter.begin====")
        c = self.f_c(v, mask)
        c = np.zeros(len(c)).reshape((1,len(c)))
        c_dx  = self.f_c_dx(v, mask)
        L_dx  = self.f_L_dx(v, mask)
        L_d2x = self.f_L_d2x(v, mask)
        ni = self.get_n_ci_activated(mask)
        nc = 2+ni
        #print("\nc=\n",c)
        #print("\nc_dx=\n",c_dx)
        #print("\nL_dx=\n",L_dx)
        #print("\nL_d2x=\n",L_d2x)
        #print("\nmask, ni=\n",mask, ni)
        
        A_0 = np.concatenate( (L_d2x,  c_dx.T            ), axis=1 )
        A_1 = np.concatenate( (c_dx,   np.zeros((nc, nc))), axis=1 )
        
        #print("\nA_0=\n", A_0)
        #print("\nA_1=\n", A_1)
        
        A   = np.concatenate( (A_0, A_1),                   axis=0 )
        
        b   = np.concatenate( (L_dx.T, np.zeros((nc, 1 ))), axis=0 )
        
        #print("\nA=\n", A)
        #print("\nb=\n", b)
        
        dx = np.linalg.solve( A, b )
        
        #print("\ndx=\n", dx)
        
        dx = dx.T[0]
        
        dp5, dp4, dT, dlv, dla = dx[:5]
        
        nv = 3+nc
        
        dlci = [0]*4
        
        j = 3+2
        for i in range(4): #iter through masks
            if(mask[i]):
                dlci[i] = dx[j]
                j+=1
        
        #print("\ndlci=\n", dlci)
        
        #print("v in iter before update: %s"%(v))
        
        v.p5 += dp5
        v.p4 += dp4
        v.set_T( v.get_T() + dT )
        
        v.lv += dlv
        v.la += dla
        
        for i in range(4):
            #print("dlci[%d]=%lf"%(i,dlci[i]))
            #print("v.lci[%d]=%lf"%(i,v.lci[i]))
            v.lci__[i] = v.lci[i] + dlci[i] # special feature added. Double underscores as a setter, returns the original variable, while the raw reference without underscores returns a list
            #print("v.lci[%d]=%lf\n"%(i,v.lci[i]))
            
        assert v.p5 < 0
        assert v.p4 > 0
        
        
        print("ce=%s"%(self.f_ce(v)))
        print("ci=%s"%(self.f_ci(v, force_all=True)))
        
        
        #print("v in iter after  update: %s"%(v))
        
        #print("====iter.end====")
        
        
        
        
        
        
        
        
class Plotter(object):
    def __init__(self, v, sp):
        self.v = v # reference of the Param object
        clr.print_green_text(self.v)
        self.sp = sp # reference of the StaticParam object
        clr.print_green_text(self.sp)
        self.inited = False
        self.npoints = 100
    
    def plot(self):
          
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
                
        v = self.v
        s = self.sp
            
        clr.print_red_text(v)
                
        T = v.get_T()
        t = np.linspace(0,T,self.npoints)
        param = v.p5, v.p4, s.p3, s.p2, s.p1, s.p0
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
        margin = 0.5
        
        #if(not result_found):
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
        #n_dj_max = np.argmax(dj)
        
        n_s_min = np.argmin(s)
        n_v_min = np.argmin(v)
        n_a_min = np.argmin(a)
        n_j_min = np.argmin(j)
        #n_dj_min = np.argmin(dj)
        
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
        
        #print(v)
        #print(s)
        #plt.plot(t,s,color="red",label="s")
        #plt.plot(t,v,color="blue",label="v")
        #plt.plot(t,a,color="green",label="a")
        
        #plt.legend()
        #plt.axis("equal")
        #plt.show()
        return self.artists
    
    def init_anim(self):
        self.s_line, = self.ax.plot([], [], "o-", color="red",   label="s")
        self.v_line, = self.ax.plot([], [], "o-", color="blue",  label="v")
        self.a_line, = self.ax.plot([], [], "o-", color="green", label="a")
        self.j_line, = self.ax.plot([], [], "o-", color="orange", label="j")
        #self.dj_line, = self.ax.plot([], [], "o-", color="purple", label="dj")
        self.s_max_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        self.s_min_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        self.v_max_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        self.v_min_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        self.a_max_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        self.a_min_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        self.j_max_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        self.j_min_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        #self.dj_max_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        #self.dj_min_anno = self.ax.annotate("", xytext=(0,0), xy=(0,0))
        
        self.x = np.linspace(0, 1, self.npoints)
        
        if not self.inited:
            self.ax.legend()
            self.inited = True
        
        self.artists = ( self.s_line, 
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
                     
        return self.artists

    
    def loop(self, frame_index):
        self.plot()
        return self.artists
    
    def show(self):
        from matplotlib.animation import FuncAnimation
        self.fig, self.ax = plt.subplots()
        self.anim = FuncAnimation(self.fig, self.loop, frames=np.arange(1), init_func=self.init_anim,
                                  interval=5, 
                                  blit=True)
        plt.show()

        
        
        
          
def test_2():
    pvp = PolyVelProfile()
    
    plotter = Plotter(pvp.var, pvp.static_param)
    
    thrd = start_daemon(lambda:pvp.run(pvp.var))
    
    plotter.show()
    
test_2()

def test_0():
            
    class test(object):
        def __init__(self):
            self.var = Param()
            print(self.var)
            
            self.mod_in(self.var.copy())
            
            
            
            print(self.var)
            
            self.var.lce = [23,34]
            print(self.var)
            self.var.lci = range(5,9)
            print(self.var)
            print("---",np.array(self.var.lce))
            print("---",np.array(self.var.lci))
            
        def mod_in(self,v):
            v.p5 = 100
            v.lci[1] = 9999
            print(v)

            
    test()
    
    PolyVelProfile.RegionEnum.INSIDE = 9999
    PolyVelProfile.RegionEnum.OUTSIDE = 999900
    PolyVelProfile.RegionEnum.EDGE = 9999000000
    print(PolyVelProfile.RegionEnum)
    
#test_0()

def test_1():
    class test_lambda(object):
        def __init__(self):
            self.l0 = 0
            self.l1 = 0
            self.l2 = 0

            self.getl0 = lambda :self.l0
            self.getl1 = lambda :self.l1
            self.getl2 = lambda :self.l2

        def setl0(self, x):
            #print("setl0", self, x)
            self.l0 = x

        def setl1(self, x):
            #print("setl1", self, x)
            self.l1 = x

        def setl2(self, x):
            #print("setl2", self, x)
            self.l2 = x

        def __repr__(self):
            return self.__str__()
            
        def __str__(self):
            return "test_lambda< l0=%s, l1=%s, l2=%s >"%(self.l0,self.l1,self.l2)
            

    t1 = test_lambda()
            
    a = Param.DummyList()
    a.register_item_cb(0, t1.getl0, t1.setl0)
    a.register_item_cb(1, t1.getl1, t1.setl1)
    a.register_item_cb(2, t1.getl2, t1.setl2)

    print(t1)
    a[0] = 1
    print(t1)
    a[1] = 2
    print(t1)
    a[2] = 3
    print(t1)