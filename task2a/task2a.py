'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2A of Niti Vahan (NV) Theme.
*
*  Filename:         task2a.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          NV_6142
# Author List:      Suyash Maurya , Pathikrit Roy Choudhury
# Filename:         task2a.py
# Functions:        detect_lane, compute_control, ackermann_wheel_angles
# Global variables: SRC, DST, M, LANE_WIDTH_PX, HALF_LANE_PX, LANE_WIDTH_M, M_PER_PX, VEHICLE_X, ASPECT, WHEELBASE, TRACK_WIDTH, WHEEL_OFFSET, W_EFF, CRUISE, V_STEADY, K_STANLEY, K_SOFT, STEER_FILTER_ALPHA, MAX_DELTA, _state


####################### IMPORT MODULES #######################
import sys
import time
import math
import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2A.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# The integer signal the evaluation raises when the vehicle must stop.
STOP_SIGNAL = 'NV_stop'
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

# Lane Detection Calibration Constants
# Calibrated bird's-eye perspective transform for CoppeliaSim camera
SRC = np.float32([[211, 200], [394, 200], [512, 410], [40, 410]])
DST = np.float32([[160, 0], [480, 0], [480, 480], [160, 480]])
M = cv2.getPerspectiveTransform(SRC, DST)

LANE_WIDTH_PX = 320.0
HALF_LANE_PX = 160.0
LANE_WIDTH_M = 0.20                             
M_PER_PX = LANE_WIDTH_M / LANE_WIDTH_PX 
VEHICLE_X = 320.0                  
ASPECT = 1.0                        

# Vehicle steering geometry constants (Task 1A)
WHEELBASE = 0.120                   
TRACK_WIDTH = 0.110                
WHEEL_OFFSET = 0.0275              
W_EFF = TRACK_WIDTH / 2.0 - WHEEL_OFFSET

# Controller gains and limits (Task 1B tuned for Task 2A)
CRUISE = 1.3                       
V_STEADY = 0.0224                  
K_STANLEY = 1.0                     
K_SOFT = 0.04                       
STEER_FILTER_ALPHA = 0.35           
MAX_DELTA = math.radians(35.0)      

# Persistent state across frames
_state = {
    "lane_width_px": 320.0,
    "last_e": 0.0,
    "last_psi": 0.0,
    "last_delta": 0.0,
    "lost_count": 0
}


def ackermann_wheel_angles(delta):
    '''
    Convert virtual front steering angle delta (radians, positive = left)
    into the two physical front wheel joint angles using Task 1A geometry.
    '''
    if abs(delta) < 1e-6:
        return 0.0, 0.0
    turn_radius = WHEELBASE / math.tan(delta)
    left_angle = math.atan(WHEELBASE / (turn_radius - W_EFF))
    right_angle = math.atan(WHEELBASE / (turn_radius + W_EFF))
    return left_angle, right_angle


def detect_lane(frame):
    '''
    Purpose:
    ---
    Extract cross-track error (e in metres) and road heading angle (psi in radians)
    from a single BGR camera frame using calibrated bird's-eye warping and
    adaptive corridor line tracking. Works seamlessly in both outer and inner lanes.

    Input Arguments:
    ---
    `frame` : [ numpy.ndarray ] of shape (480, 640, 3)

    Returns:
    ---
    `detection` : [ dict ]
        {
            'valid' : bool,
            'e'     : float (metres, positive = vehicle is to the right of centre),
            'psi'   : float (radians, positive = lane bends to the left ahead)
        }
    '''
    global _state

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 180, 0), (179, 255, 255))
    yellow = cv2.inRange(hls, (15, 100, 80), (35, 255, 255))
    mask = cv2.bitwise_or(white, yellow)

    # Exclude vehicle hood (bottom) and distant scenery (top)
    clean_mask = np.zeros_like(mask)
    clean_mask[160:420, :] = mask[160:420, :]
    warped = cv2.warpPerspective(clean_mask, M, (640, 480))

    # Base search near bottom of warped view
    hist_bottom = warped[280:460, :].sum(axis=0)
    midpoint = 320
    hist_l = hist_bottom[:midpoint]
    hist_r = hist_bottom[midpoint:]

    l_valid = hist_l.max() > 800
    r_valid = hist_r.max() > 800

    lx_base = int(np.argmax(hist_l)) if l_valid else 160
    rx_base = int(np.argmax(hist_r)) + midpoint if r_valid else 480

    poly_l = None
    poly_r = None

    if l_valid:
        ys_l, xs_l = np.nonzero(warped[120:460, max(0, lx_base - 70):min(midpoint + 20, lx_base + 70)])
        if len(ys_l) > 30:
            xs_l = xs_l + max(0, lx_base - 70)
            poly_l = np.polyfit(ys_l, xs_l, 1)

    if r_valid:
        ys_r, xs_r = np.nonzero(warped[120:460, max(midpoint - 20, rx_base - 70):min(640, rx_base + 70)])
        if len(ys_r) > 30:
            xs_r = xs_r + max(midpoint - 20, rx_base - 70)
            poly_r = np.polyfit(ys_r, xs_r, 1)

    Y_eval = 400.0  
    if poly_l is not None and poly_r is not None:
        xl = poly_l[0] * Y_eval + poly_l[1]
        xr = poly_r[0] * Y_eval + poly_r[1]
        centre_x = (xl + xr) / 2.0
        slope = (poly_l[0] + poly_r[0]) / 2.0
        _state["lane_width_px"] = 0.95 * _state["lane_width_px"] + 0.05 * (xr - xl)
    elif poly_l is not None:
        xl = poly_l[0] * Y_eval + poly_l[1]
        centre_x = xl + _state["lane_width_px"] / 2.0
        slope = poly_l[0]
    elif poly_r is not None:
        xr = poly_r[0] * Y_eval + poly_r[1]
        centre_x = xr - _state["lane_width_px"] / 2.0
        slope = poly_r[0]
    else:
        return {
            "valid": False,
            "e": _state["last_e"],
            "psi": _state["last_psi"]
        }

    e = (VEHICLE_X - centre_x) * M_PER_PX
    psi = math.atan(slope * ASPECT)

    _state["last_e"] = e
    _state["last_psi"] = psi
    return {
        "valid": True,
        "e": e,
        "psi": psi
    }


def compute_control(detection, stop_requested):
    '''
    Purpose:
    ---
    Translate detect_lane() errors into left and right front-wheel steering
    angles and motor speed. Uses a smoothed Stanley controller to eliminate
    steering oscillations, and stops the vehicle when stop_requested is True.

    Input Arguments:
    ---
    `detection`      : [ dict ] from detect_lane()
    `stop_requested` : [ bool ] True when evaluation holds vehicle stationary

    Returns:
    ---
    `left_angle`, `right_angle` : [ float ] in radians (positive = left)
    `speed`                     : [ float ] motor target speed (rad/s)
    '''
    global _state

    if stop_requested:
        left_angle, right_angle = ackermann_wheel_angles(_state["last_delta"])
        return left_angle, right_angle, 0.0

    if detection["valid"]:
        _state["lost_count"] = 0
        e = detection["e"]
        psi = detection["psi"]

        # Stanley steering control: delta = psi + atan2(k*e, v + k_soft)
        delta_raw = psi + math.atan2(K_STANLEY * e, V_STEADY + K_SOFT)
        delta_raw = max(-MAX_DELTA, min(MAX_DELTA, delta_raw))

        # Low-pass filter to guarantee smooth steering and prevent oscillation
        delta = STEER_FILTER_ALPHA * delta_raw + (1.0 - STEER_FILTER_ALPHA) * _state["last_delta"]
        _state["last_delta"] = delta
    else:
        _state["lost_count"] += 1
        delta = _state["last_delta"]

    left_angle, right_angle = ackermann_wheel_angles(delta)

    # Moderate speed slightly if detector is lost or turning hard
    if _state["lost_count"] > 15:
        speed = CRUISE * 0.4
    else:
        speed = CRUISE

    return left_angle, right_angle, speed




##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera and writing to the joints.
# None of it involves a decision you are being marked on - it is here so that
# every team reads the same picture and drives the same vehicle.
#
#####################################################################

def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.

    Each getObject() is a network round trip, which is why they all happen
    here rather than inside the loop.
    '''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2A.ttt open?\n'
            f'  Simulator said: {err}')

    handles = {}
    for name, path in (('vision_sensor', VISION_SENSOR),
                       ('steer_left', STEER_LEFT),
                       ('steer_right', STEER_RIGHT),
                       ('motor_left', MOTOR_LEFT),
                       ('motor_right', MOTOR_RIGHT)):
        try:
            handles[name] = sim.getObject(path)
        except Exception as err:
            raise RuntimeError(
                f'No object at "{path}" in the open scene.\n'
                f'  Open Task_2A.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    The vision sensor hands back a raw byte buffer and a resolution, not an
    image, and three corrections are needed to turn one into the other. None
    of these problems announces itself - each produces a detector that looks
    like it is working on a frame that is wrong:

      * handleVisionSensor() - this scene's sensor renders only when asked.
        Without it every read returns the SAME frame for the whole lap.
      * cv2.flip(frame, 0) - the buffer's first row is the BOTTOM of the
        image, so the road comes out vertically mirrored.
      * RGB2BGR - the buffer is RGB and OpenCV assumes BGR. Without the swap
        every HSV threshold you tuned in Task 1C selects the wrong colour.

    Note `res[1], res[0]` in the reshape too: the resolution comes back as
    (width, height) while NumPy wants rows first.

    Returns None if no frame is ready yet, which happens on the first few
    iterations after the simulation starts.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def drive(sim, handles, left_angle, right_angle, speed):
    '''Write the two steering angles and the drive speed to the joints.'''
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def main():
    sim, handles = connect_and_locate()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, places the vehicle in the lane being tested
    # and starts the simulation - so wait for that rather than starting your
    # own, which would race the harness into position.
    #
    # Your script must never call startSimulation(), stopSimulation() or move
    # the vehicle. Reading the camera and driving the joints is all it does.
    deadline = time.monotonic() + 120.0
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - driving.')

    frames = 0
    try:
        # The harness stops the simulation when the lap is over, which is the
        # cue to exit - hence testing the state rather than looping forever.
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            # getInt32Signal() returns None until the signal is first set, so
            # "not set" reads as "not stopping" and the first frames behave.
            stop_requested = bool(sim.getInt32Signal(STOP_SIGNAL))

            detection = detect_lane(frame)
            left_angle, right_angle, speed = compute_control(detection,
                                                             stop_requested)
            drive(sim, handles, left_angle, right_angle, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        # A call raising because the simulation stopped mid-frame is normal at
        # the end of a run; anything else is a bug worth seeing.
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        # Without this, whatever speed was last commanded is still being
        # applied and the vehicle drives on unsteered.
        try:
            sim.setJointTargetVelocity(handles['motor_left'], 0.0)
            sim.setJointTargetVelocity(handles['motor_right'], 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
