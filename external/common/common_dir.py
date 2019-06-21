import numpy as np
"""
覆盖判定模块
"""
def mdir2vec(DIR):
    '''Change direction to vector'''
    DIR = DIR/180.0*np.pi
    if 0 < DIR < np.pi/2:
        xu = 1.0
        yu = np.tan(np.pi/2-DIR)
    elif np.pi/2 < DIR < np.pi:
        xu = 1.0
        yu = -np.tan(DIR-np.pi/2)
    elif np.pi < DIR < 3*np.pi/2:
        xu = -1.0
        yu = -np.tan(3*np.pi/2-DIR)
    elif 3*np.pi/2 < DIR < 2*np.pi:
        xu = -1.0
        yu = np.tan(2*np.pi-DIR)
    elif DIR == 0 or DIR == 2*np.pi:
        xu = 0.0
        yu = 1.0
    elif DIR == np.pi/2:
        xu = 1.0
        yu = 0.0
    elif DIR == np.pi:
        xu = 0.0
        yu = -1.0
    else:
        xu = -1.0
        yu = 0.0
    return (xu,yu)

def mvector_angle(v1, v2):
    try:
        s1 = v1[0]*v2[0]+v1[1]*v2[1]
        s2 = np.sqrt(v1[0]**2+v1[1]**2)*np.sqrt(v2[0]**2+v2[1]**2)
        if s2>0:
            s = np.arccos(s1/s2)/np.pi*180
        else:
            s = None
    except:
        s = None
    return s

def mis_in_beam(DIRs, Beamwidth, xs, ys, xr, yr):
    '''Judge whether a neighbor cell(xr,yr) is in the antenna beam(beamwidth in dgree)
      of a serving cell(xs,ys) of direction DIRs(in degree).'''
    if mvector_angle(mdir2vec(DIRs),(xr-xs,yr-ys)) <= Beamwidth/2.0:
        return True
    else:
        return False