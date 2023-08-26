import os
from bs4 import BeautifulSoup
import cv2 as cv
import requests


def process_images(coral: str):
    if not os.path.exists(f"processed_images/{coral}"):
        os.mkdir(f"processed_images/{coral}")
    else:
        for f in os.listdir(f"processed_images/{coral}"):
            os.remove(f"processed_images/{coral}/{f}")
            
    print(f"processing {coral}")

    for i, img_file in enumerate(os.listdir(f"images/{coral}")):
        print(f"processing image {img_file}")
        img = cv.imread(f"images/{coral}/{img_file}")
        ratio = 200 / min(img.shape[0], img.shape[1])
        new_shape = (int(ratio * img.shape[1]), int(ratio * img.shape[0]))
        img = cv.resize(img, new_shape)
        img = img[img.shape[0] // 2 - 100:img.shape[0] // 2 + 100, img.shape[1] // 2 - 100:img.shape[1] // 2 + 100]
        cv.imwrite(f"processed_images/{coral}/{img_file}", img)
        # img2 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        # cv.imwrite(f"processed_images/{coral}/2_{img_file}", img2)
        # img2 = cv.rotate(img, cv.ROTATE_180)
        # cv.imwrite(f"processed_images/{coral}/3_{img_file}", img2)
        # img2 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        # cv.imwrite(f"processed_images/{coral}/4_{img_file}", img2)
        # img2 = cv.flip(img, 1)
        # cv.imwrite(f"processed_images/{coral}/5_{img_file}", img2)


if not os.path.exists("processed_images"):
    os.mkdir("processed_images")

for category in os.listdir("images"):
    process_images(category)