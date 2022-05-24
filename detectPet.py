import learnImages
from getDominantColours import get_dominant_colors
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename

import json
from os import path

def difference(x, y):
    if x > y:
        return x - y
    else:
        return y - x


def learn_images():
    if not path.exists('learnt_pet_image_colours.json'):
        learnImages.learn_images()


def get_learnt_data():
    with open('learnt_pet_image_colours.json') as json_file:
        return json.load(json_file)


def getImage():
    return askopenfilename()


def detect_pets_name(learnt_data, image_dominant_colours):
    similarities = []
    for pet_name, pet_colours in learnt_data.items():
        similarities_total = 0
        for file_name, colours in pet_colours.items():
            similarities_total += get_similarity_score(image_dominant_colours, colours)
        similarities.append((pet_name, similarities_total / len(pet_colours.items())))
    if all(score == 0 for _, score in similarities):
        return None
    similarities.sort(key=lambda x: x[1])
    return similarities[-1][0], similarities


def get_similarity_score(img1Colours, img2_colours):
    similar_colours = []
    for i in range(len(img1Colours)):
        for j in range(len(img2_colours)):
            test1 = compare_colour(img1Colours[i], img2_colours[j])
            if test1 and (img1Colours[i], img2_colours[j]) not in similar_colours:
                similar_colours.append((i, img1Colours[i], j, img2_colours[j]))
    similarity_score = 0
    if len(similar_colours) < 2:
        return 0
    for col1_pos, (col1, percentage1), col2_pos, (col2, percentage2) in similar_colours:
        percentage_diff = difference(percentage1, percentage2)
        # Get colour positions difference (closer the better)
        pos_diff = difference(col1_pos, col2_pos)
        similarity_score += (1 - percentage_diff) + ((10 - pos_diff) * 10)
    return similarity_score


def compare_colour(col1, col2):
    col1 = col1[0]
    col2 = col2[0]
    threshold = 20
    lower_bound = lambda x: x - threshold
    upper_bound = lambda x: x + threshold
    return all(lower_bound(col2[i]) <= col1[i] <= upper_bound(col2[i]) for i in range(3)) or \
           all(lower_bound(col1[i]) <= col2[i] <= upper_bound(col1[i]) for i in range(3))


def show_results(img_original, img_removed_background, imag_dominant_colours, detected_pet, similarities):
    probabilities_text = "Probabilities:\n"
    total = sum(score for _, score in similarities)
    for name, score in similarities:
        probabilities_text += f"{name}: {(score/total)*100:.2f}%\n"

    d = dict((x, y) for x, y in similarities)
    print(f"{(d['murphy']/total)*100:.2f}%, {(d['ava']/total)*100:.2f}%, {(d['poesje']/total)*100:.2f}%, {(d['ben']/total)*100:.2f}%")

    showComparison(img_original, img_removed_background, imag_dominant_colours, probabilities_text, detected_pet)


def showComparison(img_1, img_2, img_3, probabilities_text, detected_pet):
    f, ax = plt.subplots(2, 3, figsize=(8, 4))
    ax[0][0].imshow(img_1)
    ax[0][1].imshow(img_2)
    ax[1][0].imshow(img_3)

    ax[0][2].annotate(probabilities_text, xy=(0, 0), fontsize=18)
    ax[1][2].annotate(f"Detected pet: {detected_pet}", xy=(0, 0.5), fontsize=18)


    ax[0][0].axis('off')
    ax[0][1].axis('off')
    ax[0][2].axis('off')
    ax[1][0].axis('off')
    ax[1][1].axis('off')
    ax[1][2].axis('off')
    f.tight_layout()
    plt.show()


def main():
    Tk().withdraw()
    learn_images()
    learnt_data = get_learnt_data()
    image_path = getImage()
    img_original, img_removed_background, img_dominant_colours, dominant_colours_img = get_dominant_colors(image_path)
    detected_pet, similarities = detect_pets_name(learnt_data, img_dominant_colours)
    show_results(img_original, img_removed_background, dominant_colours_img, detected_pet, similarities)


if __name__ == '__main__':
    main()
