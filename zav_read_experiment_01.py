import cv2
import time
import glob
import os.path
import numpy as np
from matplotlib import pyplot as plt
import skimage
import skimage.measure
import skimage.morphology

def img_to_map(img):
    map = img.copy().astype(np.float)
    map[img == 128] = -1.0
    map[img == 0] = 1.0
    map[img == 200] = 0.0
    return map   

def map_to_img(c_map):
    img = c_map.copy()
    img[c_map == 0.0] = 200
    img[c_map == 1.0] = 0
    img[c_map == -1.0] = 128
    img = img.astype(np.uint8)
    return img

def create_img_of_centroids(centroids, c_map):
    c_map = map_to_img(c_map)
    img = np.zeros([c_map.shape[0], c_map.shape[1], 3], dtype=np.uint8)
    img[:,:,0] = c_map.copy()
    img[:,:,1] = c_map.copy()
    img[:,:,2] = c_map.copy()
    for centroid in centroids:
        img = cv2.circle(img, (int(centroid[1]), int(centroid[0])), 4, (0,0,255), 4)
        #img[int(centroid[0]), int(centroid[1]), 0] = 0
        #img[int(centroid[0]), int(centroid[1]), 1] = 0
        #img[int(centroid[0]), int(centroid[1]), 2] = 255
    return img


def get_kernels_response(map):
    diff_y = cv2.filter2D(map, -1, np.array(([0, -1], [0, 1]), dtype="int"))
    diff_x = cv2.filter2D(map, -1, np.array(([0, 0], [-1, 1]), dtype="int"))
    diff_xy = cv2.filter2D(map, -1, np.array(([-1, 0], [0, 1]), dtype="int"))
    diff_yx = cv2.filter2D(map, -1, np.array(([0, -1], [1, 0]), dtype="int"))
    return np.abs(diff_x + diff_y + diff_xy + diff_yx)

def detect_frontiers(basename, map):
    response = get_kernels_response(map.copy())
    # cv2.imwrite(os.path.join("./response/", basename), response_to_img(response))
    obstacles = (map.copy() == 1.0).astype(np.float)
    # cv2.imwrite(os.path.join("./obstacles/", basename), response_to_img(obstacles))
    response_obstacles = get_kernels_response(obstacles)
    # cv2.imwrite(os.path.join("./response_obstacle/", basename), response_to_img(response_obstacles))
    frontier_map = response - 2 * response_obstacles
    # cv2.imwrite(os.path.join("./frontier/", basename), response_to_img(frontier_map))
    frontier_map_bin = frontier_map > 0
    # cv2.imwrite(os.path.join("./frontier_bin/", basename), response_to_img(frontier_map_bin))

    labels = skimage.measure.label(frontier_map_bin, background=0)
    centroids = []
    areas = []
    for i in range(1, np.max(labels) + 1):
        props = skimage.measure.regionprops((labels == i).astype(int))
        centroids.append(props[0].centroid)
        areas.append(props[0].area)
    return centroids, areas, frontier_map_bin


def detect_consecutive(basename, current_map, previous_map,  k_size):
    cmap = current_map.copy()
    diff_map = current_map.copy().astype(np.float)

    if previous_map is not None:
        mask = np.uint8(previous_map != -1.0)
        # cv2.imwrite(os.path.join("./mask/", basename), response_to_img(mask))
        kernel = np.ones((k_size, k_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # cv2.imwrite(os.path.join("./morph_mask/", basename), response_to_img(mask))    
        diff_map[mask == 1] = 1.0
    else:
        diff_map = current_map.copy()
    # cv2.imwrite(os.path.join("./diff/", basename), map_to_img(diff_map))
    centroids, areas, frontier_map = detect_frontiers(basename, diff_map)
    return centroids, areas, frontier_map

def clear_path(c_map, c1, c2, dist):    
    vec = (c2 - c1) / float(dist)
    point = c1 
    for i in range(int(dist)):
        point = point + vec       
        if c_map[int(point[0]), int(point[1])] == 1:
            return 0
    return 1

def clusters_from_frontiers(centroids, areas, frontier_map, current_map, max_dist=10):
    clusters = np.zeros([len(centroids), len(centroids)], dtype=np.uint8)
    clusters_props = []

    for i, c in enumerate(centroids):
        for j, c2 in enumerate(centroids[i+1:]):
            c1a = np.asarray(c)
            c2a = np.asarray(c2)
            dist = np.linalg.norm(c1a - c2a)
            if dist > max_dist:
                continue
            else:
                if clear_path(current_map, c1a, c2a, dist):
                    clusters[i, i + j + 1] = 1
    temp_clusters = -1 * np.ones([clusters.shape[0], 1], dtype=np.int8)
    n_clusters = 0

    for i in range(clusters.shape[0]):
        if temp_clusters[i] == -1:
            temp_clusters[i] = n_clusters
            n_clusters = n_clusters + 1
        for j in range(clusters.shape[1]):            
            if clusters[i, j] == 1:
                if temp_clusters[j] == -1:
                    temp_clusters[j] = temp_clusters[i]
    final_centroids = []
    final_areas = []

    for i in range(n_clusters):
        occ = 0
        temp_centroid_x = 0
        temp_centroid_y = 0    
        temp_area = 0    
        for j in range(temp_clusters.shape[0]):
            if temp_clusters[j] == i:
                temp_centroid_x += centroids[j][1]
                temp_centroid_y += centroids[j][0]
                temp_area += areas[j]
                occ += 1
        final_centroids.append([temp_centroid_y / occ, temp_centroid_x / occ])
        final_areas.append(temp_area) 
    return final_centroids, final_areas


def main():
    files = sorted(glob.glob("./input/*.png"))

    css = []
    centroidss = []
    f_centroidss = []

    arss = []
    areass = []
    f_areass = []

    sum_css = []
    sum_centroidss = []
    sum_f_centroidss = []

    sum_cs = 0
    sum_centroids = 0
    sum_f_centroids = 0

    for i in range(len(files)):
        if i == 0: 
            previous_map = None
            current_map = img_to_map(cv2.imread(files[i], 0))
        else:
            previous_map = img_to_map(cv2.imread(files[i-1], 0))
            current_map = img_to_map(cv2.imread(files[i], 0))
        cs, ars, fmap = detect_consecutive(os.path.basename(files[i]), current_map, None, 5)            
        centroids, areas, frontier_map = detect_consecutive(os.path.basename(files[i]), current_map, previous_map, 5)
        f_centroids = []
        f_areas = 0
        if len(centroids) > 0:
            f_centroids, f_areas = clusters_from_frontiers(centroids, areas, frontier_map, current_map, 20)
            img = create_img_of_centroids(f_centroids, current_map)
            cv2.imwrite("./output/output_" + str(i) + ".png", img)
            img2 = create_img_of_centroids(centroids, current_map)
            cv2.imwrite("./output/output_c" + str(i) + ".png", img2)
        else:            
            areas = 0
            centroids = []            
        css.append(len(cs))
        centroidss.append(len(centroids))
        f_centroidss.append(len(f_centroids))
        arss.append(np.mean(ars))
        areass.append(np.mean(areas))
        f_areass.append(np.mean(f_areas))
        sum_cs = sum_cs + len(cs)
        sum_centroids = sum_centroids + len(centroids)
        sum_f_centroids = sum_f_centroids + len(f_centroids)

        sum_css.append(sum_cs)
        sum_centroidss.append(sum_centroids)
        sum_f_centroidss.append(sum_f_centroids)

        #print(len(cs), len(centroids), len(f_centroids), np.mean(ars), np.mean(areas), np.mean(f_areas))
        img3 = create_img_of_centroids(cs, current_map)
        cv2.imwrite("./output/output_n" + str(i) + ".png", img3)
    
    print("Centroids non-consecutive")
    print(css)
    print("Centroids consecutive")
    print(centroidss)
    print("Centroids minimal")
    print(f_centroidss)

    print("Sum Centroids non-consecutive")
    print(sum_css)
    print("Sum Centroids consecutive")
    print(sum_centroidss)
    print("Sum Centroids minimal")
    print(sum_f_centroidss)    

    print("areas non-consecutive")
    print(arss)
    print("areas consecutive")
    print(areass)
    print("areas minimal")
    print(f_areass)


if __name__ == '__main__':
    main()


