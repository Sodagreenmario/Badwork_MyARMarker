import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
import cv2
from auto_canny import auto_canny
from sklearn.cluster import AffinityPropagation
from numpy.linalg import cholesky
from objloader_simple import *
import os
from imutils.video import VideoStream
from imutils.video import FPS
import time

def valid_point(centroid,circle_clu, mx, my):
    valid = False
    for i in range(centroid.shape[0]):
        # suppose centroid[0] is circle point
        for j in range(circle_clu.shape[0]):
            if (abs(centroid[i][0] - mx) > 0.1*mx) and (abs(centroid[i][1] - my) > 0.1*my):
                continue
            if (abs(centroid[i][0]-circle_clu[j][0]) < 10) and (abs(centroid[i][1]-circle_clu[j][1]) < 10) :
                valid = True
                return centroid[i]
    return centroid[1]

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def render(img, obj, projection, img_ori, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h = 0
    w = 0

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (128, 128, 128))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def valid_img(circle_point):
    c_x = circle_point[0]
    c_y = circle_point[1]

    if c_x < t_xmin or c_x > t_xmax:
        return False
    if c_y < t_ymin or c_y > t_xmax:
        return False

def Ellipse_Detect(img):
    # Resize the image at raio 10
    resized = imutils.resize(img, width=int(img.shape[1] / 5))
    ratio = img.shape[0] / resized.shape[0]

    # Edge detection using auto canny method
    img_canny = auto_canny(resized)

    # Threshold the img_canny using binary method
    ret, thresh = cv2.threshold(img_canny, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Fit ellipse and draw the fitted ellipse, circle_info is used to remark information about ellipse
    circle_info = []
    img_res = img.copy()
    get_one = False
    for cnt in contours:
        if len(cnt) > 50:
            get_one = True
            s1 = cv2.contourArea(cnt)
            el1 = cv2.fitEllipse(cnt)
            [cx, cy] = el1[0]
            # print(cx*ratio,cy*ratio)
            [b, a] = el1[1]
            # print(b*ratio,a*ratio)
            theta = el1[2]
            # print(theta)
            circle_info.append(el1)
            new_el1 = tuple(([cx * ratio, cy * ratio], [b * ratio, a * ratio], theta))
            s2 = math.pi * el1[1][0] * el1[1][1]
            #if s2 is 0:
                #continue
            #if (s1 / s2) > 0.2:

                #img_res = cv2.ellipse(img, new_el1, (0, 255, 0), 20)
            #else:
             #   continue
    if get_one is False:
        return img

    # add random samples
    randn_num = (np.random.randn(64, 64)).reshape(-1, 1)
    randn_cnt = randn_num.shape[0]
    info_cnt = np.array(circle_info).reshape(-1, 1).shape[0]
    each_cnt = randn_cnt // info_cnt

    # Get the (cx,cy) as the center of fitted ellipse
    circle_info_clu = []
    circle_axis = []
    for info in circle_info:
        [cx, cy] = np.array(np.array(info[0]) * ratio, dtype=int)
        [ea, eb] = np.array(np.array(info[1]) * ratio, dtype=int)
        circle_info_clu.append([cx, cy])
        circle_axis.append([cx, cy, ea, eb])

    # added 20181014
    mean_x, mean_y = 0, 0
    clu = np.array(circle_info_clu)
    mean_x, mean_y = np.mean(clu, axis=0)
    #print(mean_x, mean_y)

    # Clusters
    if (np.array(circle_info_clu).shape[0] >= 4):
        #estimator = KMeans(n_clusters=3, init='k-means++', n_init=150)
        estimator = AffinityPropagation(preference=-20)
    else:
        #estimator = KMeans(n_clusters=1, init='k-means++', n_init=150)
        return img
    #print(np.array(circle_info_clu).shape)
    estimator.fit(np.array(circle_axis).reshape(-1,4))
    centroids = estimator.cluster_centers_

    # Get the Selected circle point
    median_x, median_y, median_b, median_a = np.median(centroids, axis=0)
    circle_point = valid_point(centroids, np.array(circle_info_clu), mean_x, mean_y)
    circle_point = np.array(circle_point, dtype=int)

    # Draw the circle_point
    cv2.circle(img_res, tuple([circle_point[0], circle_point[1]]), 10, (0, 0, 255), -1)

    #if valid_img(circle_point) is False:
     #   return img
        #if former_circle_info is None:
         #   return img
        #else:
    #    #    circle_info = former_circle_info
  #  else:
     #   former_circle_info = circle_info


    # Calculate matrix to get the long axis and the short axis
    M_point = []
    img_vec = img_res.copy()
    for info in circle_info:
        [cx, cy] = np.array(np.array(info[0]) * ratio, dtype=int)
        if (abs(np.array([cx, cy], dtype=int)[0] - circle_point[0]) < 30
            and abs(np.array([cx, cy], dtype=int)[1] - circle_point[1] < 30)):
            [b, a] = np.array(np.array(info[1]) * ratio, dtype=int) / 2
            theta = np.array(np.array(info[2]), dtype=int)
            sin_t = math.sin(theta)
            cos_t = math.cos(theta)
            A = a * a * sin_t * sin_t + b * b * cos_t * cos_t
            B = 2 * (b * b - a * a) * sin_t * cos_t
            C = a * a * cos_t * cos_t + b * b * sin_t * sin_t
            m = [[A, B / 2], [B / 2, C]]
            eigval, eigvec = np.linalg.eig(m)

            pointer_a_s = np.array([cx, cy] + eigvec[0] * a, dtype=int)
            pointer_a_t = np.array([cx, cy] - eigvec[0] * a, dtype=int)
            if pointer_a_s[0] > pointer_a_t[0]:
                pointer_a_s, pointer_a_t = pointer_a_t, pointer_a_s

            pointer_b_s = np.array([cx, cy] + eigvec[1] * b, dtype=int)
            pointer_b_t = np.array([cx, cy] - eigvec[1] * b, dtype=int)
            if pointer_b_s[1] > pointer_b_t[0]:
                pointer_b_s, pointer_b_t = pointer_b_t, pointer_b_s


            point_set = np.array([pointer_a_s, pointer_a_t, pointer_b_s, pointer_b_t])
            #print(point_set)
            p_le = np.min(point_set, axis=0)[0]
            p_do = np.min(point_set, axis=0)[1]
            p_ri = np.max(point_set, axis=0)[0]
            p_up = np.max(point_set, axis=0)[1]

            M_point.append(m)
            break

    src_pts = np.array([[-150, 0], [0, 150], [150, 0], [0, -150]])
    des_pts = np.array([pointer_a_s, pointer_b_s, pointer_a_t, pointer_b_t])

    h, status = cv2.findHomography(src_pts, des_pts)

    homography = h
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    projection = projection_matrix(camera_parameters, homography)

    obj = OBJ(os.path.join(os.getcwd(), 'models/fox.obj'), swapyz=True)
    final = render(img_vec.copy(), obj, projection, img, False)

    return pointer_a_s, pointer_b_s, pointer_a_t, pointer_b_t

'''
video_capture = cv2.VideoCapture(0)
while 1:
    _, frame = video_capture.read()
    canvas = Ellipse_Detect(frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
'''
tracker = cv2.TrackerKCF_create()

initBB = None
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
fps = None
t_xmin, t_xmax, t_ymin, t_ymax = 0, 0, 0, 0
ori_x, ori_y, ori_w, ori_h = 0, 0, 0, 0

obj = OBJ(os.path.join(os.getcwd(), 'models/cube.obj'), swapyz=True)
camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
src_pts = np.array([[-150, 0], [0, 150], [150, 0], [0, -150]])
pointer_a_s, pointer_b_s, pointer_a_t, pointer_b_t = [0, 0], [0, 0], [0, 0], [0, 0]

while True:
    frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=640)
    print(frame.shape)
    (H, W) = frame.shape[:2]

    if initBB is not None:
        (success, box) = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            t_xmin, t_xmax, t_ymin, t_ymax = x, x + w, y, y + h
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            delta_x = x - ori_x
            delta_y = y - ori_y
            delta_w = w - ori_w
            delta_h = w - ori_w
            ori_x, ori_y = x, y
            if abs(delta_x)>1 or abs(delta_y)>1:
                pointer_a_s = pointer_a_s + [delta_x, delta_y]
                pointer_b_s = pointer_b_s + [delta_x, delta_y] + [delta_w, 0]
                pointer_a_t = pointer_a_t + [delta_x, delta_y] + [delta_w, delta_h]
                pointer_b_t = pointer_b_t + [delta_x, delta_y] + [0, delta_h]
            des_pts = np.array([pointer_a_s, pointer_b_s, pointer_a_t, pointer_b_t])
            homography, status = cv2.findHomography(src_pts, des_pts)
            print("There:", des_pts, homography)
            projection = projection_matrix(camera_parameters, homography)
            frame = render(frame, obj, projection, frame, False)

        fps.update()
        fps.stop()

        info = [
			("Tracker", "kcf"),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
        tracker.init(frame, initBB)
        fps = FPS().start()

        pointer_a_s, pointer_b_s, pointer_a_t, pointer_b_t = Ellipse_Detect(frame)
        des_pts = np.array([pointer_a_s, pointer_b_s, pointer_a_t, pointer_b_t])
        homography, status = cv2.findHomography(src_pts, des_pts)
        print(homography)
        projection = projection_matrix(camera_parameters, homography)
        frame = render(frame, obj, projection, frame, False)
        initBB_2 = np.array(initBB)
        ori_x, ori_y = initBB_2[0], initBB_2[1]
        ori_w, ori_h = initBB_2[2], initBB_2[3]

    elif key == ord("q"):
        break


cv2.destroyAllWindows()