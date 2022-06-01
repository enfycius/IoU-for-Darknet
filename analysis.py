import numpy as np
import cv2
import json

with open("./result.json", "r") as result_json:
    result = json.load(result_json)

    for i in range(0, len(result)):
        file_name = result[i]['filename'][13:-4]
        image = cv2.imread("./drive/" + file_name + ".png")
        f = open("./drive/" + file_name + ".txt")
        data = f.readlines()
        f.close()

        for j in range(0, len(result[i]['objects'])):
            center_x = result[i]['objects'][j]['relative_coordinates']['center_x']
            center_y = result[i]['objects'][j]['relative_coordinates']['center_y']
            width = result[i]['objects'][j]['relative_coordinates']['width']
            height = result[i]['objects'][j]['relative_coordinates']['height']

            x1 = round(image.shape[1] * center_x) - round(image.shape[1] * width / 2)
            y1 = round(image.shape[0] * center_y) - round(image.shape[0] * height / 2)
            x2 = round(image.shape[1] * center_x) + round(image.shape[1] * width / 2)
            y2 = round(image.shape[0] * center_y) + round(image.shape[0] * height / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        for j in range(0, len(data)):
            center_x = float(data[j].split(' ')[1])
            center_y = float(data[j].split(' ')[2])
            width = float(data[j].split(' ')[3])
            height = float(data[j].split(' ')[4][:-1])

            x1 = round(image.shape[1] * center_x) - round(image.shape[1] * width / 2)
            y1 = round(image.shape[0] * center_y) - round(image.shape[0] * height / 2)
            x2 = round(image.shape[1] * center_x) + round(image.shape[1] * width / 2)
            y2 = round(image.shape[0] * center_y) + round(image.shape[0] * height / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite("./test/" + "test_" + file_name + ".png", image)

