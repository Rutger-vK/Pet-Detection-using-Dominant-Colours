import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from removeBackground import remove_background


def getImage(path):
    img = cv2.imread(path)
    aspect_ratio = img.shape[1] / img.shape[0]
    width = 500
    height = int(500 * aspect_ratio)
    img = cv2.resize(img, (height, width))
    return img

def get_colours(clusters):
    counter = Counter(clusters.labels_)  # count how many pixels per cluster
    counter_list = list(counter.items())
    counter_list.sort(key=lambda x: x[1])
    counter_list.reverse()

    # Remove the black
    new_counter = []
    total_pixels = 0
    for index, count in counter_list:
        colour = clusters.cluster_centers_[index]
        if not all(val < 5 for val in colour):
            new_counter.append((index, count))
            total_pixels += count

    colours = []
    for index, count in counter_list:
        colour = clusters.cluster_centers_[index]
        if not all(val < 5 for val in colour):
            percentage = np.round(count / total_pixels, 2)
            colours.append((list(colour), percentage))

    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    step = 0

    for idx, (centers, percentage) in enumerate(colours):
        palette[:, step:int(step + percentage * width + 1), :] = centers
        step += int(percentage * width + 1)

    return colours, palette

def get_dominant_colors(image_path):
    img_original = getImage(image_path)
    img_removed_background = remove_background(img_original)
    img_removed_background = cv2.cvtColor(img_removed_background, cv2.COLOR_BGR2RGB)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    clt = KMeans(n_clusters=7)
    cluster = clt.fit(img_removed_background.reshape(-1, 3))
    colours, colours_img = get_colours(cluster)
    return img_original, img_removed_background, colours, colours_img