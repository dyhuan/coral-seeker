# pylint:disable=no-member (Removes linting problems with cv)

import os
import shutil

import re
import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import cv2 as cv

species = ["Branching", "Encrusting", "Favia", "Favites"]
session = HTMLSession()

def get_species_images(search):
    body = requests.get("http://www.coralsoftheworld.org/species_factsheets/").content
    soup = BeautifulSoup(body, "html.parser")
    children = soup.find("optgroup", label=search).contents

    if not os.path.exists(f"images/{search}"):
        os.mkdir(f"images/{search}")
    else:
        for f in os.listdir(f"images/{search}"):
            os.remove(f"images/{search}/{f}")

    index = 0
    for child in children:
        if child == "\n":
            continue

        print(f"downloading image {index}")

        child_text = child.text.lower().replace(" ", "-")
        child_html = session.get(
            f"http://www.coralsoftheworld.org/species_factsheets/species_factsheet_summary/{child_text}/")
        child_html.html.render()
        child_body = child_html.html.html
        child_soup = BeautifulSoup(child_body, "html.parser")
        img_src = child_soup.find("a", href=re.compile(r"^/image.*")).find("img").get("src").split("preview/")[-1][:-1]
        img = requests.get(f"http://www.coralsoftheworld.org/media/images/{img_src}.jpg", stream=True)
        print(img.url)
        print(img.status_code)
        with open(f"images/{search}/{search}_{index}.jpg", "wb") as f:
            shutil.copyfileobj(img.raw, f)
        index += 1


def process_images(search):
    if not os.path.exists(f"processed_images/{search}"):
        os.mkdir(f"processed_images/{search}")
    else:
        for f in os.listdir(f"processed_images/{search}"):
            os.remove(f"processed_images/{search}/{f}")

    for i, img_file in enumerate(os.listdir(f"images/{search}")):
        print(f"processing image {i}")
        img = cv.imread(f"images/{search}/{img_file}")
        ratio = 200 / min(img.shape[0], img.shape[1])
        new_shape = (int(ratio * img.shape[1]), int(ratio * img.shape[0]))
        img = cv.resize(img, new_shape)
        img = img[img.shape[0] // 2 - 100:img.shape[0] // 2 + 100, img.shape[1] // 2 - 100:img.shape[1] // 2 + 100]
        cv.imwrite(f"processed_images/{search}/{img_file}", img)
        img2 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        cv.imwrite(f"processed_images/{search}/2_{img_file}", img2)
        img2 = cv.rotate(img, cv.ROTATE_180)
        cv.imwrite(f"processed_images/{search}/3_{img_file}", img2)
        img2 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        cv.imwrite(f"processed_images/{search}/4_{img_file}", img2)
        img2 = cv.flip(img, 1)
        cv.imwrite(f"processed_images/{search}/5_{img_file}", img2)


if not os.path.exists("images"):
    os.mkdir("images")

if not os.path.exists("processed_images"):
    os.mkdir("processed_images")

for s in species:
    # get_species_images(s)
    process_images(s)