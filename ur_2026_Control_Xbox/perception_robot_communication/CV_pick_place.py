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

#!/usr/bin/python

import cv2 as cv
import numpy as np
import time
import math
import json

'''
d435 Intel RealSense Camera Settings:
Enable Auto Exposure: Off
Exposure: 156.000 (modify as necessary)
Enable Auto White Balance: Off 
White Balance: 3370.000 (modify as necessary)
'''

# for 640 x 480 resolution
# UR5 mounted cam
# pix_per_in = 17
# ceiling mounted cam
pix_per_in = 12
pix_per_cm = pix_per_in/2.54

cam_center = (320, 240) # in pixels
conveyor_motion = 1 # direction of conveyor in camera frame (x: 0, y: 1)

# bottle properties
bottle_w = 7.2 # cm
bottle_h = 20.2 # cm

# Test 0, 1, 2, 3 to find that the usb camera, may change every time the camera is reconnected
cap = cv.VideoCapture(2)
bkgnd_cap = False
frame_count = 0

# hsv mask: (H, S, V)
# hsv mask: (H, S, V), H: 0-179, S: 0-255, V: 0-255
# modify these lb and ub
ub_hsv_mask_r = (179,255,255)
lb_hsv_mask_r = (0,0,0)

ub_hsv_mask_g = (179,255,255)
lb_hsv_mask_g = (0, 0, 0)

# nominal hsv_y = (30.4, 216.75, 255)
ub_hsv_mask_y = (80, 255, 255)
lb_hsv_mask_y = (10, 100, 100)

belt_speed = -0.11 # conveyor belt speed [m/s]
target_prev_time = 0 # prev time the target was detected
target_prev_pos = 0
vel_reset = True # does the target velocity need to be reset bc the target has changed?
moving_avg_pos = 0 # moving average of target position values
moving_avg_vel = 0 # moving average of target velocity values
moving_avg_pos2 = 0 # moving average of the target position normal to the direction of conveyor motion
moving_n = 5 # window length of the moving average
prev_n_pos = [] # current window of values for position moving avg
prev_n_vel = [] # current window of values for velocity moving avg
prev_n_pos2 = [] #  "       "       "       "   pos2    "       "

def mask_and_bound(img, lb_mask, ub_mask, color):
    # create a list of dictionaries for bounding boxes of a bottle of a color
    # {"center": ..., "box": ..., "color": ..., "detect_time": ...}
    img_masked = cv.inRange(img, lb_mask, ub_mask)

    # Morphological operations
    erosion_size = 1
    erosion_size_2 = 2
    element = cv.getStructuringElement(cv.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                    (erosion_size, erosion_size))
    img_masked = cv.morphologyEx(img_masked, cv.MORPH_OPEN, element, iterations=2) # opening operation
    element_2 = cv.getStructuringElement(cv.MORPH_RECT, (2 * erosion_size_2 + 1, 2 * erosion_size_2 + 1),
                                    (erosion_size_2, erosion_size_2))
    img_masked = cv.morphologyEx(img_masked, cv.MORPH_CLOSE, element_2, iterations=2) # closing operation

    contours, _ = cv.findContours(img_masked, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    output = []
    for i, c in enumerate(contours):
        rect = cv.boundingRect(c) # rect: (x, y, w, h)
        rect_x, rect_y, rect_w, rect_h = rect[0], rect[1], rect[2], rect[3]
        min_dim, max_dim = min(rect_w, rect_h), max(rect_w, rect_h)
        # don't consider bounding boxes that are too small
        # smallest dimension shouldn't be less than the bottle width
        if bottle_w > min_dim/pix_per_cm + 0.1: continue
        # largest dimension shouldn't be smaller than bounding box height if the bottle was at 45 degrees
        elif bottle_h*math.sin(math.pi/4) > max_dim/pix_per_cm + 0.1: continue
        # bottle center with respect to center of camera
        center = ((rect_x+rect_w/2 - cam_center[0])/pix_per_cm, (rect_y+rect_h/2 - cam_center[1])/pix_per_cm)
        detect_time = time.time_ns()
        output.append({"center": center, "box": rect, "color": color, "time": detect_time})
    return output

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    cv.imshow('frame', frame) # show frame

    if not bkgnd_cap and frame_count>30: # collect the background image at the 30th frame
        print("Collecting background image...")
        bkgnd_img = frame
        print("Background image collected")
        bkgnd_cap = True
    elif not bkgnd_cap: # only need to count frames up to when you get the background image
        frame_count+=1
    elif bkgnd_cap:    
        post_sub = cv.subtract(frame, bkgnd_img) # background subtraction
        #cv.imshow('subtracted', post_sub)
        #cv.imshow('background', bkgnd_img)

        post_sub_hsv = cv.cvtColor(post_sub, cv.COLOR_BGR2HSV)
        #r_objs = mask_and_bound(post_sub_hsv, lb_hsv_mask_r, ub_hsv_mask_r, "red")
        #g_objs = mask_and_bound(post_sub_hsv, lb_hsv_mask_g, ub_hsv_mask_g, "green")
        y_objs = mask_and_bound(post_sub_hsv, lb_hsv_mask_y, ub_hsv_mask_y, "yellow")
        all_objs = y_objs #r_objs + g_objs + y_objs
        if all_objs:
            # print("Bottles detected: ", len(all_objs) )
            # sort list to have the bottle closest to the end of the conveyor appear first
            all_objs = sorted(all_objs, key=lambda x: x["center"][conveyor_motion], reverse=False) # modify if necessary
            for obj in all_objs:
                obj_x, obj_y, obj_w, obj_h = obj["box"][0], obj["box"][1], obj["box"][2], obj["box"][3]
                post_sub = cv.rectangle(post_sub, (obj_x,obj_y), (obj_x+obj_w, obj_y+obj_h), (0, 0, 255), 2)
            cv.imshow('subtracted', post_sub)
        else:
            # if there are no bottles detected don't try to sort the list; continue iterating
            target_prev_time = 0
            target_prev_pos = 0
            target_vel = 0
            vel_reset = True
            prev_n_pos.clear()
            prev_n_vel.clear()
            prev_n_pos2.clear()
            cv.imshow('subtracted', post_sub)
            if cv.waitKey(1) == ord('q'): break
            print("No bottles detected, vel reset: ", vel_reset)
            continue

        target = all_objs[0] # choose the bottle closest to the end of the conveyor as the target
        target_time = target["time"]*10**-9 # time the target was detected in seconds
        target_pos = (target["center"][0]*0.01, target["center"][1]*0.01) # m from camera center in camera frame
        if target_prev_time:
            delta_t = target_time - target_prev_time
            target_vel = (target_pos[conveyor_motion] - target_prev_pos[conveyor_motion])/delta_t # target velocity [m/s] in the direction of conveyor belt motion
            if len(prev_n_pos) == moving_n: # remove oldest element in the moving average list if needed     
                prev_n_pos.pop(0)
            if len(prev_n_vel) == moving_n:
                prev_n_vel.pop(0)
            if len(prev_n_pos2) == moving_n:
                prev_n_pos2.pop(0)
            prev_n_pos.append(target_pos[conveyor_motion]) # add most recent mesaurement to moving average list
            prev_n_vel.append(target_vel)
            prev_n_pos2.append(target_pos[int(not conveyor_motion)])
            moving_avg_pos = sum(prev_n_pos)/len(prev_n_pos)
            moving_avg_vel = sum(prev_n_vel)/len(prev_n_vel)
            moving_avg_pos2 = sum(prev_n_pos2)/len(prev_n_pos2)
            vel_reset = False
            # if the target is removed from the camera view => switch to a new target => velocity jump => reset target velocity
            if abs(belt_speed-target_vel)>0.1: # max +- 0.035 m/s difference if bottle is still
                # reset finding target vel, pos, prev_time; continue iterating
                target_vel = 0
                target_prev_time = 0
                target_prev_pos = 0
                vel_reset = True
                continue
        else:
            target_vel = 0

        target_prev_time = target_time
        target_prev_pos = target_pos
        print("target moving avg pos:", moving_avg_pos, " target moving avg vel:", moving_avg_vel, " vel reset: ", vel_reset, "moving avg pos2:", moving_avg_pos2,)

        # send target vel and target pos to other script that will control the ur5
        # figure out why json file gets nothing written to it i.e. is empty sometimes
        data = {"target_pos": moving_avg_pos, "target_vel": moving_avg_vel, "target_pos2": moving_avg_pos2, "vel_reset": vel_reset}
        with open("CV_pick_place_data.json", "w") as file:
            try:
                json.dump(data, file)
                #print("Writing data")
            except Exception as e:
                print("Error:", e, "data:", data)


    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()