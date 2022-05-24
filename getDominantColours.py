import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from copy import copy
from removeBackground import remove_background


def getImage(path):
    img = cv2.imread(path)
    aspect_ratio = img.shape[1] / img.shape[0]
    width = 500
    height = int(500 * aspect_ratio)
    img = cv2.resize(img, (height, width))
    return img


def removeBackground(img):
    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and uppper limits
    corners = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
    # corners = [(0, 0), (-1, -1)]
    for x, y in corners:
        corner = img[x, y]
        if all(val == 0 for val in corner):
            continue

        lower = np.array([pixel - 10 for pixel in corner])
        upper = np.array([pixel + 10 for pixel in corner])

        # Create mask to only select black
        thresh = cv2.inRange(img, lower, upper)

        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # invert morp image
        mask = 255 - morph

        # apply mask to image
        img = cv2.bitwise_and(img, img, mask=mask)

        black_mask = np.all(img == 0, axis=-1)
        alpha = np.uint8(np.logical_not(black_mask)) * 255
        img = np.dstack((img, alpha))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def showComparison(img_1, img_2, img_3):
    f, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[2].imshow(img_3)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    f.tight_layout()
    plt.show()


def palette(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    counter = Counter(k_cluster.labels_)
    dominant = [None for _ in range(len(counter))]
    for i, ordered_i in enumerate(counter):
        dominant[i] = k_cluster.cluster_centers_[ordered_i]

    for colour in copy(dominant):
        if all(val < 1 for val in colour):
            dominant.remove(colour)

    steps = width / len(dominant)
    for idx, centers in enumerate(dominant):
        palette[:, int(idx * steps):(int((idx + 1) * steps)), :] = centers

    return palette


def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)  # count how many pixels per cluster

    for i in copy(counter):
        colour = k_cluster.cluster_centers_[i]
        if all(val < 1 for val in colour):
            n_pixels = n_pixels - counter[i]
            counter.pop(i)

    dominant = [None for _ in range(len(counter))]
    perc = [None for _ in range(len(counter))]
    for i, ordered_i in enumerate(counter):
        dominant[i] = k_cluster.cluster_centers_[ordered_i]
        perc[i] = np.round(counter[ordered_i] / n_pixels, 2)

    print(counter)
    print(perc)
    step = 0

    for idx, centers in enumerate(dominant):
        palette[:, step:int(step + perc[idx] * width + 1), :] = centers
        step += int(perc[idx] * width + 1)

    return palette


def testPalette_perc(clusters):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

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
            colours.append((colour, percentage))
            print(colour)

    step = 0

    for idx, (centers, percentage) in enumerate(colours):
        palette[:, step:int(step + percentage * width + 1), :] = centers
        step += int(percentage * width + 1)

    return palette

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
    # cv2.imwrite('output/' + image_path.split("/")[1], img_removed_background)
    img_removed_background = cv2.cvtColor(img_removed_background, cv2.COLOR_BGR2RGB)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
    clt = KMeans(n_clusters=7)
    cluster = clt.fit(img_removed_background.reshape(-1, 3))
    colours, colours_img = get_colours(cluster)
    return img_original, img_removed_background, colours, colours_img

def main(image_path):
    img_original = getImage(image_path)
    img_removed_background = remove_background(img_original)
    # img_removed_background = removeBackground(img_original)
    img_removed_background = cv2.cvtColor(img_removed_background, cv2.COLOR_BGR2RGB)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)

    clt = KMeans(n_clusters=7)
    clt_1 = clt.fit(img_removed_background.reshape(-1, 3))
    img_colours = testPalette_perc(clt_1)

    # show_img_compar(img, palette(clt_1))
    showComparison(img_original, img_removed_background, img_colours)


if __name__ == '__main__':
    main("images/ava5.jpg")
