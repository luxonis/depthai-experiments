import math
import numpy as np

# calculate vector from camera to landmark
def get_vector_direction(camera_position, landmark):
    vector = []

    for i in range(3):
        vector.append(landmark[i] - camera_position[i])

    return np.array(vector)

# method - guess and estimate
#def get_ray_intersection(p1_origin, p1_dir, p2_origin, p2_dir):
#    start, end = 0, 3
#
#    for _ in range(4):
#        min_distance = get_min_distance(p1_origin, p1_dir, p2_origin, p2_dir, start, end)
#        increment = (end-start) / 10
#        print(increment)
#        start = min_distance - increment
#        end = min_distance + increment
#
#    return np.add(p1_origin,np.dot(p1_dir,min_distance))
#
#def distance_between_point_ray(p, a, u):
#    ap = np.subtract(p,a)
#    ap_cross = np.cross(ap, u)
#    return np.linalg.norm(ap_cross) / np.linalg.norm(u)
#
#def get_min_distance(p1_origin, p1_dir, p2_origin, p2_dir, start, end):
#    increment = (end - start) / 10
#    j = start
#
#    while j <= end:
#        last_distance = distance_between_point_ray(np.add(p1_origin, np.dot(p1_dir,(j - increment))), p2_origin, p2_dir)
#
#        current_distance = distance_between_point_ray(np.add(p1_origin, np.dot(p1_dir, increment)), p2_origin, p2_dir)
#
#        next_distance = distance_between_point_ray(np.add(p1_origin, np.dot(p1_dir,(j + increment))), p2_origin, p2_dir)
#
#        if ((last_distance - current_distance) > 0 and (current_distance - next_distance) < 0) or \
#            ((last_distance - current_distance) < 0 and (current_distance - next_distance) > 0):
#            return j
#
#        j += increment
#
#    return None



# ----------------------------------------------------
# pure math calculation method - not too stable

# left_vector, right_vector d1 d2
# left_camera_position, right_camera_position p1, p2
def get_vector_intersection(left_vector, left_camera_position, right_vector, right_camera_position):
    n = np.cross(left_vector, right_vector)
    n1 = np.cross(left_vector, n)
    n2 = np.cross(right_vector, n)

    #c1 = p1 + ( (p1-p2) * n2)/(d1*n2) )d1
    top = np.dot(np.subtract(right_camera_position,left_camera_position), n2)
    bottom = np.dot(left_vector,n2)
    divided = top/bottom
    mult = divided*left_vector
    c1 = left_camera_position + mult
#    print(c1)

    #c2 = p2 + ( (p1-p2) * n1)/(d2*n1) )d2
    top = np.dot(np.subtract(left_camera_position,right_camera_position), n1)
    bottom = np.dot(right_vector,n1)
    divided = top/bottom
    mult = divided*right_vector
    c2 = right_camera_position + mult
#    print(c2)

    center = (c1+c2)/2
#    print(center)
    return center
#
