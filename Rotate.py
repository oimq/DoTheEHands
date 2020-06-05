from numpy import array
from math import cos, sin, radians

def getRotTheta(theta) :
    rt = radians(theta)
    T_z_t = array([[cos(rt), -sin(rt),       0,       0],
                   [sin(rt),  cos(rt),       0,       0],
                   [      0,        0,       1,       0],
                   [      0,        0,       0,       1]])
    return T_z_t

def getRotAlpha(alpha) :
    ra = radians(alpha)
    T_x_a = array([[      1,        0,       0,       0],
                   [      0,  cos(ra),-sin(ra),       0],
                   [      0,  sin(ra), cos(ra),       0],
                   [      0,        0,       0,       1]])
    return T_x_a

def getPrisDist(dist) :
    T_z_d = array([[      1,        0,       0,       0],
                   [      0,        1,       0,       0],
                   [      0,        0,       1,    dist],
                   [      0,        0,       0,       1]])
    return T_z_d


def getPrisAist(aist) :
    T_x_a = array([[      1,        0,       0,    aist],
                   [      0,        1,       0,       0],
                   [      0,        0,       1,       0],
                   [      0,        0,       0,       1]])
    return T_x_a

def getRevoluteJointMatrix(theta, dista, alpha, aista) :
    radth = radians(theta)
    radal = radians(alpha)
    T_x_r = array([[   cos(radth), -1*cos(radal)*sin(radth),    sin(radal)*sin(radth),    aista*cos(radth)],
                   [   sin(radth),    cos(radal)*cos(radth), -1*sin(radal)*cos(radth),    aista*sin(radth)],
                   [          0.0,               sin(radal),               cos(radal),           1.0*dista],
                   [          0.0,                      0.0,                      0.0,                 1.0]])
    return T_x_r