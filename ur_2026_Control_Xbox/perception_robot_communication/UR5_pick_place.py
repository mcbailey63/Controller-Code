# create a ROS workspace for the UR pick and place team
# create script for CV to detect the center of the bottles' bounding boxes moving on the conveyor belt
# send velocities and positions of the bottles to the ur arm movement script
# ur arm movement script chooses the right most bottle (center point) as the target
    # predicts where the bottle will be based on its current position and velocity
    # how fast can the UR move?
    # decide a location where the UR should pick up the robot and go there 
    # pick up the bottle when it is predicted that the bottle is that location

# code assumes bottles are placed on the conveyor with no initial velocity
#  "    "   "  bottles do not slip against conveyor
#  "    "   "  conveyor speed is constant
#  "    "   "  bottle travels in a straight line along conveyor

#!/usr/bin/python

import time
import json
import rtde_control, rtde_receive
import math
from robotiq_gripper_control import RobotiqGripper

'''
Current error: Traceback (most recent call last):
  File "UR5_pick_place.py", line 126, in <module>
    time.sleep(bottle_delay)
ValueError: sleep length must be non-negative
'''

box_ip = "192.168.1.103"
rtde_c = rtde_control.RTDEControlInterface(box_ip)
rtde_r = rtde_receive.RTDEReceiveInterface(box_ip)
gripper = RobotiqGripper(rtde_c)

gripper_close_t = 1 # time to close gripper in seconds
gripper_rot_vel = math.pi # gripper rotational velocity (180 deg/s)

ee_accel = 1.2 # end effector acceleration
ee_vel = 0.75 # end effector velocity [m/s]
ur_moving = False # is the UR5 performing a motion?
pendant_z_offset = 0.400 # 400 mm offset between the UR5 pendant z reading, and ur rtde TCP z reading
ee_z_hover = -0.317 + pendant_z_offset # hover height when moving to bottles [m], measured manually
ee_z_grab = -0.380 + pendant_z_offset # grab height for bottle [m], measured manually
ee_rx_grab = 2.160 # [rad], measured manually
ee_ry_grab = 2.233 # [rad], measured manually
ee_rz_grab = 0 # [rad], measured manually 

rot_ur_conveyor = -5*math.pi/180 # [rad], rotation from -x in conveyor frame to x in UR5 frame 

y_bin_position = (0.139, -0.782, -0.354+pendant_z_offset) # yellow bin position (x, y, z) [m] in the UR5 frame, measured manually

dth = 0.1 # discrete step in theta for intersection point calculation
steps = math.floor((1.5*math.pi)/dth) # total whole steps in intersection calc

def ur_to_conveyor(x, y, phi):
    ''' 
    Transforms coordinates in UR5 frame to conveyor frame 
    x, y are in UR5 frame, phi is the rotation between frames
    '''
    # camera camera/conveyor origin to UR5 origin in conveyor frame [m, m]
    r_conveyor2ur = (17.375*0.0254, -19*0.0254) #
    # (19.375*0.0254, -19*0.0254) # using center found with opencv 
    # (17.375*0.0254, -17.5*0.0254) # using center found with real sense viewer
    r_out = [0, 0]
    r_out[0] = r_conveyor2ur[0] - x*math.cos(phi) + y*math.sin(phi)  
    r_out[1] = r_conveyor2ur[1] - x*math.sin(phi) - y*math.cos(phi)
    return r_out

def conveyor_to_ur(x, y, phi):
    ''' 
    Transforms coordinates in conveyor frame to UR5 frame 
    x, y are in conveyor frame, phi is the rotation between frames
    '''
    # [m, m] UR5 origin to camera/conveyor origin in UR5 frame
    r_ur2conveyor = (0.495, -0.355) #
    # (0.495, -0.383) # measured with UR5 and opencv camera center
    # (0.451, -0.355) # measured with UR5 and realsense viewer center
    # (17.375*0.0254, -17.5*0.0254) # measured with ruler 
    r_out = [0, 0]
    r_out[0] = r_ur2conveyor[0] - x*math.cos(phi) - y*math.sin(phi)  
    r_out[1] = r_ur2conveyor[1] + x*math.sin(phi) - y*math.cos(phi)
    return r_out

connection_tries = 0
if not rtde_c.isConnected():
    while connection_tries < 3:
        rtde_c.reconnect()
        time.sleep(0.1)
        if rtde_c.isConnected():
            break
        connection_tries += 1

if rtde_c.isConnected():
    print("Connection successful!")
else:
    print("Connection not working")
    rtde_c.stopScript()

gripper.activate()
gripper.set_force(50)
gripper.set_speed(100)
gripper.open()

try:
    while rtde_c.isConnected():
        # read target vel and target position
        with open("CV_pick_place_data.json", "r") as file:
            try:
                data = json.load(file)
            except Exception as e:
                print("Error:", e)

        if not data["vel_reset"]:
            target_vel = data["target_vel"]
            target_pos = data["target_pos"] # target position in direction of conveyor motion
            target_pos2 = data["target_pos2"] # target position in direction normal to conveyor motion
            curr_ee = rtde_r.getActualTCPPose() # get current ee position, [x y z rx ry rz]
            # estimate where the meeting point should be (hovering over bottle)
            # x_target(t) = x_arm(t); y_target(t) = y_arm(t) <= equations based on velocities and initial positions of bottle and ur5 ee
            ee_x_conveyor, ee_y_conveyor = ur_to_conveyor(curr_ee[0], curr_ee[1], rot_ur_conveyor)
            for s in range(steps):
                th = 1.5*math.pi - s*dth # search from pi to 0
                tc = (-target_pos2 - ee_y_conveyor)/(ee_vel*math.sin(th))
                target_x_tc = -target_pos - target_vel*tc # position of target along conveyor direction at tc
                ur_x_tc = ee_x_conveyor + ee_vel*math.cos(th)*tc # position of UR5 along conveyor direction at tc
                x_tc_diff = ur_x_tc - target_x_tc # difference in arrival positions along conveyor direction at tc
                if x_tc_diff >= 0: # at tc (when UR5 intersects with target_pos2), UR5 x value should be larger than bottle x value
                    xc_conveyor = ur_x_tc
                    yc_conveyor = -target_pos2
                    break
            print("th: ", th, "tc: ", tc)
            print("Pick up position in conveyor coords: ", xc_conveyor, yc_conveyor)
            print("x_tc_diff: ", x_tc_diff)
            if target_vel!=0: 
                bottle_delay = x_tc_diff/abs(target_vel)
                xc_ur, yc_ur = conveyor_to_ur(xc_conveyor, yc_conveyor, rot_ur_conveyor)
            else: 
                bottle_delay=0
                xc_ur, yc_ur = conveyor_to_ur(target_x_tc, yc_conveyor, rot_ur_conveyor)
            ur_moving = True
            print("Moving to pick up location...")
            rtde_c.moveL([xc_ur, yc_ur, ee_z_hover, ee_rx_grab, ee_ry_grab, ee_rz_grab], ee_vel, ee_accel, asynchronous=False)
            print("Waiting ", bottle_delay, "s for bottle to arrive...")
            time.sleep(bottle_delay)
            #ee_grab_vel_x, ee_grab_vel_y = conveyor_to_ur(-target_vel,0,rot_ur_conveyor)
            #ee_grab_vel_z = (ee_z_grab - ee_z_hover)/gripper_close_t
            #rtde_c.speedL([ee_grab_vel_x, ee_grab_vel_y, ee_grab_vel_z, 0, 0, 0], ee_accel, gripper_close_t)
            grab_pos = rtde_r.getActualTCPPose()
            grab_pos[2] = ee_z_grab
            print("Moving to grabbing position...")
            rtde_c.moveL(grab_pos, ee_vel, ee_accel, asynchronous=False)
            gripper.close()
            print("Grabbed bottle...or missed")
            print("Moving bottle to bin...")
            rtde_c.moveL([y_bin_position[0], y_bin_position[1], y_bin_position[2], ee_rx_grab, ee_ry_grab, ee_rz_grab], 
                        ee_vel, ee_accel,asynchronous=False)
            gripper.open()
            print("Bottle (or nothing) dropped, movement complete!")
            '''
            elif ur_moving:
                print("Velocity reset...waiting for UR5 movement to finish")
                time.sleep(5)
                ur_moving = False
                continue
            else:
                continue
            '''
except KeyboardInterrupt as e:
    print(e)    

rtde_c.stopScript()

# gripper open and close, 
# breaking out the while loop to stop the script
# changing target in the middle of moveL
# what happens when moveL is asynchronous