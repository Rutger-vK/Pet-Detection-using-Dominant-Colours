from getDominantColours import get_dominant_colors

import json
from os import listdir
from os.path import isfile, join
import re


def learn_images():
    images_path = 'images/'
    images = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    pets = dict()

    print("learning images")


    # seperate into pets
    for image in images:
        split_index = re.search(r"\d", image)
        pet_name = image[:split_index.start()]
        if pet_name not in pets:
            pets[pet_name] = {image: ()}
        else:
            pets[pet_name][image] = ()
    count = 0

    for pet_name, pet_colours in pets.items():
        for image in pet_colours:
            image_path = images_path + image
            pets[pet_name][image] = get_dominant_colors(image_path)[2]
            count += 1
            print(f"processed {count}/{len(images)} images")

    pets_json = json.dumps(pets)

    with open('learnt_pet_image_colours.json', 'w') as outfile:
        outfile.write(pets_json)
