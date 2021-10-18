import cv2
import numpy as np
import math
import glob
import os

def maske(original_img, mask_img):
    height, weight = mask_img.shape
    for i in range(height):
        for k in range(weight):
            if original_img[i][k] > 120:
                original_img[i][k] = 255
            else:
                if mask_img[i][k] != 255:
                    original_img[i][k] = 255
                else:
                    original_img[i][k] = 0
    return original_img

def c_l(lines, org_img, width, height):
    max_size = max(width, height) ** 2
    for i in range(0, len(lines)):
        a = np.cos(np.deg2rad(90-lines[i][1]))
        b = np.sin(np.deg2rad(90-lines[i][1]))
        x0 = a * lines[i][0]
        y0 = b * lines[i][0]
        x1 = int(x0 + max_size * (-b))
        y1 = int(y0 + max_size * a)
        x2 = int(x0 - max_size * (-b))
        y2 = int(y0 - max_size * a)

        org_img = cv2.line(org_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return org_img

def h_l(edge_info):

    theta_array = []
    for i in range(180):
        theta_array.append(np.deg2rad(i))  # 0-180 degrees
    cos_t = np.cos(theta_array)
    sin_t = np.sin(theta_array)
    
    width, height = edge_info.shape
    rho_r = round(math.sqrt(width**2 + height**2))
    
    accumulator = np.zeros((2*rho_r, len(theta_array)), dtype=np.uint8)  #
    white_edges = np.where(edge_info == 255)  # ıt holds which point is white
    white_points = list(zip(white_edges[0], white_edges[1]))

    for i in range(len(white_points)):
        for t in range(len(theta_array)):
            rho = int(round(white_points[i][0]*cos_t[t] + white_points[i][1]*sin_t[t]))
            accumulator[rho, t] += 1

    edge = np.where(accumulator > 90)  # Here is for the threshold value of hough transform
    all_lines = list(zip(edge[0], edge[1]))
    return all_lines


def read_file(directory_path):
    filenames = glob.glob(os.path.join(directory_path, '*.png'))
    file_names_ = []
    for file_name in filenames:
        file_names_.append(file_name.split("\\")[1])
    return file_names_

if __name__ == '__main__':
    file_names = read_file("Dataset/Original_Subset/")
    for name in file_names:
        bgr_img = cv2.imread("Dataset/Original_Subset/" + name)
        mask = cv2.imread('Dataset/Detection_Subset/' + name, 0)
        mask_de = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        bgr_img_show = cv2.resize(bgr_img, (0, 0), None, .5, .5) 
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        masked = maske(gray_img, mask)
        edges = cv2.Canny(masked, 50, 150, apertureSize=3)
        edges1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        canshow = cv2.resize(edges1, (0, 0), None, .5, .5)  
        linesbarcode = h_l(edges)
        output_img = c_l(linesbarcode, bgr_img, edges.shape[0], edges.shape[1])
        showoutput = cv2.resize(output_img, (0, 0), None, .5, .5)  
        mask1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        maskoutput = c_l(linesbarcode, mask1, edges.shape[0], edges.shape[1])
        maskoutput_show = cv2.resize(maskoutput, (0, 0), None, .5, .5) 
        numpy_horizontal_concat = np.concatenate((bgr_img_show, canshow, showoutput, maskoutput_show), axis=1)
        cv2.imshow("output_image", numpy_horizontal_concat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
