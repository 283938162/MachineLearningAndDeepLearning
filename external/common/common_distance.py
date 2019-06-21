import numpy as np
import math

"""
功能说明: 公共模块-距离计算模块
依赖关系: distance.py(距离计算模块)
作者日期: 韦晓灵|2018-5-31 11:13:08
"""

def mdir2slope(dir):
    '''斜率计算'''
    DIR=dir/180.0*np.pi
    if 0<DIR<np.pi/2:
        slope=np.tan(np.pi/2-DIR)
    elif np.pi/2<DIR<np.pi:
        slope = -np.tan(DIR-np.pi/2)
    elif np.pi<DIR<3*np.pi/2:
        slope=np.tan(3*np.pi/2-DIR)
    elif 3*np.pi/2<DIR<2*np.pi:
        slope=-np.tan(DIR-3*np.pi/2)
    elif DIR==0 or DIR==2*np.pi:
        slope=None
    elif DIR==np.pi/2:
        slope=0
    elif DIR==np.pi:
        slope=None
    else:
        slope=0
    return slope
def getDistance(lat0,lng0,lat1,lng1):
    '''距离计算'''
    PI=3.14159265
    R=6.371229*1000000
    x=(lng1-lng0)*PI*R*math.cos(((lat1+lat0)/2)*PI/180)/180
    y=(lat1-lat0)*PI*R/180

    return math.sqrt(x*x+y*y)