'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2B of Niti Vahan (NV) Theme.
*
*  Filename:         sign_obstacle_detection.py
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
# Author List:      Suyash Maurya, Pathikrit Roy Choudhury
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle
# Global variables: _REF_TEMPLATES


####################### IMPORT MODULES #######################
import cv2
import numpy as np
##############################################################

# The only sign labels detect_signs() is allowed to return.
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_signs(image):
    detections = []
    if image is None or image.size == 0:
        return detections

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Detect red signs: School (triangular), Speed40 (circular)
    m1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(m1, m2)
    cnts_r, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts_r:
        area = cv2.contourArea(c)
        if area >= 400:
            x, y, w, h = cv2.boundingRect(c)
            ar = float(w) / max(h, 1)
            if 0.5 <= ar <= 1.5:
                roi_hsv = hsv[y:y+h, x:x+w]
                white_ratio = np.mean((roi_hsv[:, :, 1] <= 45) & (roi_hsv[:, :, 2] >= 140))
                # Signs have a white interior with symbols/text; obstacles do not
                if white_ratio >= 0.20:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                    circ = 4.0 * np.pi * area / (peri * peri + 1e-5)
                    if len(approx) == 3 or circ < 0.65:
                        label = "School"
                    else:
                        label = "Speed40"
                    detections.append({"label": label, "area": int(round(area))})

    # 2. Detect blue signs: Parking, Hospital
    tmpl_p, tmpl_h = _get_reference_templates()
    b, g, r = cv2.split(image)
    blue_color = (b > r.astype(int) + 25) & (b > g.astype(int)) & (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] >= 60)
    blue_mask = blue_color.astype(np.uint8) * 255
    cnts_b, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts_b:
        area = cv2.contourArea(c)
        if area >= 400:
            x, y, w, h = cv2.boundingRect(c)
            ar = float(w) / max(h, 1)
            extent = area / float(w * h)
            if 0.65 <= ar <= 1.45 and extent >= 0.60:
                crop = cv2.resize(cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY), (32, 32))
                sp = cv2.matchTemplate(crop, tmpl_p, cv2.TM_CCOEFF_NORMED)[0][0]
                sh = cv2.matchTemplate(crop, tmpl_h, cv2.TM_CCOEFF_NORMED)[0][0]
                if max(sp, sh) >= 0.18:
                    if sp > sh:
                        label = "Parking"
                    else:
                        label = "Hospital"
                    detections.append({"label": label, "area": int(round(area))})

    return detections


def detect_obstacle(image):
    present = False
    area = 0
    if image is None or image.size == 0:
        return {"present": present, "area": area}

    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red HSV mask (detects obstacle under all lighting conditions)
    m1 = cv2.inRange(hsv, (0, 65, 45), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 65, 45), (180, 255, 255))
    red_mask = cv2.bitwise_or(m1, m2)

    # Exclude vehicle's own front chassis / markers at the bottom of the image
    red_mask[int(H * 0.78):, :] = 0

    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_area = 0

    for c in cnts:
        c_area = cv2.contourArea(c)
        if c_area < 250:  # Road studs are <= 180px; obstacles are >= 250px
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 20 or h < 20:
            continue

        ar = float(w) / max(h, 1)
        if not (0.35 <= ar <= 2.5):
            continue

        # Obstacle base must be on or near the road surface, not high in the sky
        if y + h < 45:
            continue

        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        if ha == 0:
            continue
        solidity = c_area / ha
        if solidity < 0.70:  # A 3D box is convex; irregular scenery/overhangs are not
            continue

        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * c_area / (peri * peri + 1e-5)
        if circ < 0.40:
            continue

        # Differentiate from red signs (Speed40, School):
        # Measure white symbol pixels strictly inside the contour mask
        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1, offset=(-x, -y))
        roi_hsv = hsv[y:y+h, x:x+w]

        white_pixels = np.sum((roi_hsv[:, :, 1] <= 55) & (roi_hsv[:, :, 2] >= 130) & (c_mask > 0))
        total_c_pixels = np.sum(c_mask > 0)
        white_ratio = white_pixels / max(total_c_pixels, 1)

        if white_ratio >= 0.18:
            continue  # Traffic sign with white interior face

        # Distant background scenery check: high up near the horizon at outer image edges
        if y < 45 and (x > 0.70 * W or x < 0.30 * W):
            continue

        if c_area > best_area:
            best_area = c_area
            best_candidate = c

    if best_candidate is not None:
        present = True
        area = int(round(best_area))

    return {"present": present, "area": area}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# ------------------------------------------------------------------

_REF_TEMPLATES = None

def _get_reference_templates():
    global _REF_TEMPLATES
    if _REF_TEMPLATES is not None:
        return _REF_TEMPLATES

    import os
    import zlib
    import base64

    # Embedded 32x32 normalized grayscale templates
    P_B64 = b'eJyt0z0SgkAMBeBbQt7CjjZWlt7AwkvYeBMdT2FpY22jIOxPHBBh0M1S6Gu/TZgkAzNBCQF59vm8tkLMDlW6MJCSuaMmR6JjxlBWZmBwekV0w0NKFXe2E85G9+4Gb2da2+bBUnKA2g6h/t1HVyP/qoeeqHdxvza8j89X5VH37z0L3m3n029FcT9t+uVL84n3C/pU/a/97X+8/V8iHg4DNpOZGMnWiQ8Sf1Eeh9FlRzmnJT+6U4eS1/wEB1skRg=='
    H_B64 = b'eJwr1sELDHQGB/j/////MhhHD8TDI685wPLa5OnvzoeCIuzyqID+8lIwTtrQlJemQL+VlZWVJoyT/P79+/eo8oMWWOCXVgUAQM1epg=='

    tmpl_p = np.frombuffer(zlib.decompress(base64.b64decode(P_B64)), dtype=np.uint8).reshape((32, 32))
    tmpl_h = np.frombuffer(zlib.decompress(base64.b64decode(H_B64)), dtype=np.uint8).reshape((32, 32))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_p_path = os.path.join(script_dir, "reference_images", "P.jpg")
    ref_h_path = os.path.join(script_dir, "reference_images", "Hospital.png")

    if os.path.exists(ref_p_path):
        p_img = cv2.imread(ref_p_path)
        if p_img is not None:
            p_hsv = cv2.cvtColor(p_img, cv2.COLOR_BGR2HSV)
            p_blue = (p_hsv[:, :, 0] >= 95) & (p_hsv[:, :, 0] <= 130) & (p_hsv[:, :, 1] >= 50)
            ys, xs = np.where(p_blue)
            if len(ys) > 0 and len(xs) > 0:
                p_crop = p_img[ys.min():ys.max(), xs.min():xs.max()]
                tmpl_p = cv2.resize(cv2.cvtColor(p_crop, cv2.COLOR_BGR2GRAY), (32, 32))

    if os.path.exists(ref_h_path):
        h_img = cv2.imread(ref_h_path)
        if h_img is not None:
            h_hsv = cv2.cvtColor(h_img, cv2.COLOR_BGR2HSV)
            h_blue = (h_hsv[:, :, 0] >= 95) & (h_hsv[:, :, 0] <= 130) & (h_hsv[:, :, 1] >= 50)
            ys, xs = np.where(h_blue)
            if len(ys) > 0 and len(xs) > 0:
                h_crop = h_img[ys.min():ys.max(), xs.min():xs.max()]
                tmpl_h = cv2.resize(cv2.cvtColor(h_crop, cv2.COLOR_BGR2GRAY), (32, 32))

    _REF_TEMPLATES = (tmpl_p, tmpl_h)
    return _REF_TEMPLATES

##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_signs(detections, image_name):
    '''
    Purpose:
    ---
    Check that detect_signs() returned the expected structure and normalise
    it, so a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `detections` :  [ object ]     whatever detect_signs() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ list of dict ]   [{"label": str, "area": int}, ...]
    '''
    where = "detect_signs() on {}".format(image_name)

    if not isinstance(detections, list):
        raise TypeError("{} must return a list, got {}".format(where, type(detections).__name__))

    clean = []
    for index, entry in enumerate(detections):
        entry_where = "{}, entry {}".format(where, index)

        if not isinstance(entry, dict):
            raise TypeError("{} must be a dict, got {}".format(entry_where, type(entry).__name__))

        missing = {"label", "area"} - set(entry.keys())
        if missing:
            raise ValueError("{} is missing the key(s): {}".format(entry_where, ", ".join(sorted(missing))))

        label = entry["label"]
        if label not in SIGN_LABELS:
            raise ValueError("{} returned label = '{}', expected one of {}".format(
                entry_where, label, ", ".join(SIGN_LABELS)))

        area = _validate_area(entry["area"], entry_where)
        clean.append({"label": label, "area": area})

    return clean


def validate_obstacle(result, image_name):
    '''
    Purpose:
    ---
    Check that detect_obstacle() returned the expected structure and
    normalise it.

    Input Arguments:
    ---
    `result` :      [ object ]     whatever detect_obstacle() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ dict ]       {"present": bool, "area": int}
    '''
    where = "detect_obstacle() on {}".format(image_name)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"present", "area"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    present = result["present"]
    if not isinstance(present, (bool, np.bool_)):
        raise TypeError("{} returned present of type {}, expected a bool".format(
            where, type(present).__name__))
    present = bool(present)

    area = _validate_area(result["area"], where)
    if not present and area != 0:
        raise ValueError("{} returned present=False but area={}, expected 0".format(where, area))

    return {"present": present, "area": area}


def _validate_area(area, where):
    if isinstance(area, bool) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned area of type {}, expected a number".format(
            where, type(area).__name__))
    area = int(round(float(area)))
    if area < 0:
        raise ValueError("{} returned a negative area: {}".format(where, area))
    return area
