from tkinter import Tk as Screen, Canvas, Button
from CoordinateSystem import *
from CellSystem import FanoCell
from Signal import SIGNALS, NUM_SIGBITS

from random import randint

######## Canvas propertise ########
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 750
POINT_SIZE = 20
LINE_WIDTH = 3

######## Prepare drawing code ########
screen = Screen()
canvas = Canvas(width = CANVAS_WIDTH, height = CANVAS_HEIGHT)
screen.update()
canvas.pack()

######## Coordinate code ########
create_point_dh_list()
init_dh_llist()

#print("line_list :", line_list)

######### Drawing code #########
canvas_points = list()
canvas_lines = list()
for i in range(len(point_list)) :
    rgb_red = 0
    for cori in range(1, len(point_list[i])) :
        canvas_lines.append(canvas.create_line(
            point_list[i][cori-1][2],
            point_list[i][cori-1][0],
            point_list[i][cori][2],
            point_list[i][cori][0],
            width = 30
        ))
    for cor in point_list[i] :
        p = POINT_SIZE+(cor[1]/100)
        canvas_points.append(canvas.create_oval(
            cor[2]-p,
            cor[0]-p,
            cor[2]+p,
            cor[0]+p,
        fill = (lambda x : "#%02x%02x%02x" % x)((rgb_red, 255, 255))))
        rgb_red += 100


######### Dynamic code #########
def drawing() :
    global canvas_points
    l = 0
    for i in range(len(point_list)) :
        for cori in range(1, len(point_list[i])) :
            canvas.coords(canvas_lines[l],
                          point_list[i][cori-1][2],
                          point_list[i][cori-1][0],
                          point_list[i][cori][2],
                          point_list[i][cori][0],
                          )
            l_width = ((point_list[i][cori-1][1])+200)/10
            if l_width > 0 :
                canvas.itemconfig(canvas_lines[l],
                                  width=l_width)
            l += 1
    c = 0
    for i in range(len(point_list)) :
        for cor in point_list[i] :
            p_size = POINT_SIZE+(cor[1]/20)
            canvas.coords(canvas_points[c],
                          cor[2]-p_size,
                          cor[0]-p_size,
                          cor[2]+p_size,
                          cor[0]+p_size)
            c += 1

    canvas.after(100, drawing)
drawing()

### for graphic effect, we need another world dh
world_dh = array([0,0,0,0])

# Implement the cell + dh
def implement() :
    for i in range(len(point_list)) :
        point_list[i][0] = get_coordinate([dh_list[i][0], world_dh], world_frame)
    for i in range(len(point_list)) :
        point_list[i][1] = get_coordinate([dh_list[i][1], dh_list[i][0], world_dh], world_frame)
    for i in range(len(point_list)) :
        point_list[i][2] = get_coordinate([dh_list[i][2], dh_list[i][1], dh_list[i][0], world_dh], world_frame)
    screen.after(50, implement)
implement()

# Finger joint atrributes
finger_theta_list = []
## Initialize finger_theta_list
for i in range(NUM_FINGER) :
    finger_theta_list.append(list())
    for j in range(NUM_JOINT) :
        finger_theta_list[i].append(0.0)

TUMB_FINGER = 0
PONT_FINGER = 1
MIDL_FINGER = 2
RING_FINGER = 3
BABY_FINGER = 4

# Define fingers rotation(theta)
def rotate() :
    global var_theta, var_flag, finger_theta_list
    for i in range(NUM_FINGER) :
        for j in range(1, NUM_JOINT) :
            dh_list[i][j][THETA] = finger_theta_list[i][j]
    screen.after(100, rotate)
rotate()

# Cell Initialize code
cell_list = list()
## Initialize cell_list
for i in range(NUM_FINGER) :
    cell_list.append(list())
    for j in range(NUM_JOINT-1) :
        cell_list[i].append(FanoCell())
### CAUTION : Remember DH thoerem. we only need upper two joint groups

## Set mask of cells which are in cell_list
sigindex = 0
for cells in cell_list :
    for cell in cells :
        cell.setmask(SIGNALS[sigindex])
        sigindex += 1

######### mouse event #########
# make mouse drag and drop event
mouse_x = 0
mouse_y = 0
alpha_init = 0
theta_init = 0
dista_init = 0
aista_init = 0

# VISUALLY GOOD!
world_dh[THETA] = -20
world_dh[ALPHA] = -30
world_dh[DISTA] = 100
world_dh[AISTA] = -180

def mouse_1_press(event) :
    global mouse_x, mouse_y, alpha_init, theta_init, world_dh
    mouse_x = event.x
    mouse_y = event.y
    alpha_init = world_dh[ALPHA]
    theta_init = world_dh[THETA]

def mouse_1_release(event) :
    global mouse_x, mouse_y, world_dh, alpha_init, theta_init
    for dh in dh_list :
        world_dh[ALPHA] = alpha_init + ((event.x - mouse_x) / CANVAS_WIDTH * 45)
        world_dh[THETA] = theta_init + ((mouse_y - event.y) / CANVAS_HEIGHT * 45)

def mouse_3_press(event) :
    global mouse_x, mouse_y, dista_init, aista_init, world_dh
    mouse_x = event.x
    mouse_y = event.y
    dista_init = world_dh[DISTA]
    aista_init = world_dh[AISTA]

def mouse_3_release(event) :
    global mouse_x, mouse_y, world_dh, alpha_init, theta_init
    for dh in dh_list :
        world_dh[DISTA] = dista_init + ((event.x - mouse_x) / CANVAS_WIDTH * 200)
        world_dh[AISTA] = aista_init + ((event.y - mouse_y) / CANVAS_HEIGHT * 200)

canvas.bind("<ButtonPress-1>", mouse_1_press)
canvas.bind("<B1-Motion>", mouse_1_release)

canvas.bind("<ButtonPress-3>", mouse_3_press)
canvas.bind("<B3-Motion>", mouse_3_release)

# create bit flip mode
def button_flip() :
    global flip_mode, flip_button
    flip_mode = not flip_mode
    if flip_mode :
        flip_button['text'] = "비트 플립 모드 활성화"
        flip_button['bg'] = 'yellow'
    else :
        flip_button['text'] = "비트 플립 모드 비활성화"
        flip_button['bg'] = 'red'

flip_button = Button(screen, command = button_flip, text="비트 플립 모드 비활성화", bg="red")
flip_button.pack()
######### RUNNING CODE #########
# we transmit signal to cells
# cell read the transmitted signal and would be activated.
def change_joint_theta() :
    global finger_theta_list, cell_list
    print("------------------------ CHANGE JOINT ------------------------")
    for i in range(NUM_FINGER) :
        for j in range(NUM_JOINT-1) :
            finger_theta_list[i][j+1] = cell_list[i][j].power() * -10
    screen.after(100, change_joint_theta)
change_joint_theta()

## we generate sig by signal
sig = 0b00000000000000000000000000000000

# input the signal, then compare all cell's masks
def apply_signal(signal) :
    global cell_list
    print("------------------------ APPLY SIGNAL ------------------------")
    for cells in cell_list :
        for cell in cells :
            cell.apply(signal)

# this is for one two signal
power = 0
pflag = 1
finger = 6
fflag = 1
# define flip mode
flip_mode = False
BER = 1
# generate signal
def generate_signal() :
    global power, pflag, finger, fflag, flip_mode, BER
    # local signal for test
    # RANDOM SIGNAL
    # sig = SIGNALS[randint(0, 9)] + (SIGNALS[randint(0, 9)]<<NUM_SIGBITS)
    # ONE TWO THREE FOUR FIVE SIGNAL
    sig = SIGNALS[finger] + (SIGNALS[power]<<NUM_SIGBITS)
    power += pflag
    if power >= 10 :
        power = 9
        pflag = -1
    elif power < 0 :
        power=1
        pflag = +1
    # if finger >= NUM_FINGER*2 :
    #     finger = NUM_FINGER*2 - 1
    #     power = 9
    #     pflag *= -1
    # elif finger < 0 :
    #     finger = 0
    #     power = 0
    #     pflag *= -1
    #print("GENERATE SIGNAL :", ("{0:0"+str(NUM_SIGBITS*2)+"b}").format(sig), " |  FINGER :", finger, " |  POWER :", power)

    # bit flip code
    if flip_mode :
        if randint(1, int(1/BER)) == 1:
            str_sig = ("{0:0"+str(NUM_SIGBITS*2)+"b}").format(sig)
            flip_index = randint(0, len(str_sig)-1)
            if str_sig[flip_index] == "0" :
                str_sig = str_sig[:flip_index]+"1"+str_sig[flip_index+1:]
            else :
                str_sig = str_sig[:flip_index]+"0"+str_sig[flip_index+1:]
            sig = int(str_sig, 2)
    apply_signal(sig)
    screen.after(100, generate_signal)
generate_signal()

screen.mainloop()