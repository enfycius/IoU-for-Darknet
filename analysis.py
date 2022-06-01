import numpy as np
import cv2
import json

with open("./result.json", "r") as result_json:
    result = json.load(result_json)

    center_x = result[0]['objects'][0]['relative_coordinates']['center_x']
    width = result[0]['objects'][0]['relative_coordinates']['width']
    center_y = result[0]['objects'][0]['relative_coordinates']['center_y']
    height = result[0]['objects'][0]['relative_coordinates']['height']

    image = cv2.imread("./test.png")

    x1 = round(image.shape[1] * center_x) - round(image.shape[1] * width / 2)
    y1 = round(image.shape[0] * center_y) - round(image.shape[0] * height / 2)
    x2 = round(image.shape[1] * center_x) + round(image.shape[1] * width / 2)
    y2 = round(image.shape[0] * center_y) + round(image.shape[0] * height / 2)

    print(image.shape[0])
    print(image.shape[1])

    print(round(center_y * image.shape[0]))

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)

