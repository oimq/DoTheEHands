from numpy import array, matmul
from Rotate import *

######## finger propertise ########

# define fingers
NUM_FINGER = 5
NUM_JOINT = 3
NUM_POINT = NUM_JOINT
LINK_LENGTH = 150
WORLD_PRIS = 490

# define coordinate index
THETA = 0
DISTA = 1
ALPHA = 2
AISTA = 3

# for line coordinates
point_list = list()
dh_list = list()

# world frame : absolute frame
world_frame = array([0.0, 0.0, 0.0, 1.0])

# base frame : for visualize the rotate world frame
base_frame = array([0.0, 0.0, 0.0, 1.0])
base_joint = array([0.0, 0.0, 0.0, 0.0])

# first five elements means first joints, next element means second joints.
def create_point_dh_list() :
    global dh_list, point_list
    for i in range(NUM_FINGER) :
        # theta, distance, alpha, aistance
        dh_list.append([array([0.0, 0.0, 0.0, 0.0]) for j in range(NUM_JOINT)])
        point_list.append([array([0.0, 0.0, 0.0, 1.0]) for j in range(NUM_POINT)])
    # print("----- create_cor_llist -----")
    # print(dh_list)
    # print(point_list)

# initialize finger's positions
def init_dh_llist() :
    global x, y, z, dh_list
    # each finger based coordinates of base frame
    for i in range(NUM_FINGER) :
        # about world frame dh
        dh_list[i][0][DISTA] = i * 100 + 100
        dh_list[i][0][AISTA] = WORLD_PRIS
    for i in range(NUM_FINGER) :
        for j in range(1, NUM_JOINT) :
            dh_list[i][j][AISTA] -= LINK_LENGTH
    # adjust thumb finger...
    dh_list[4][0][ALPHA] = -90.0
    dh_list[4][0][THETA] = 20.0
    dh_list[4][0][AISTA] += 200.0
    # print("----- init_dh_llist -----")
    # print(dh_list)

# get transposition matrix
def get_transposition(th, di, al, ai) :
    mat = matmul(getPrisDist(ai), getRotTheta(al))
    mat = matmul(getRotAlpha(di), mat)
    mat = matmul(getPrisAist(th), mat)
    return mat

def get_coordinate(dh_list, vec) :
    mat = array([[1.0,   0.0,    0.0,    0.0],
                [0.0,   1.0,    0.0,    0.0],
                [0.0,   0.0,    1.0,    0.0],
                [0.0,   0.0,    0.0,    1.0]])
    for dh in dh_list :
        mat = matmul(getRevoluteJointMatrix(dh[0], dh[1], dh[2], dh[3]), mat)
    return matmul(mat, vec)