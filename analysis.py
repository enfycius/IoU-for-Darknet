import numpy as np
import cv2
import json

def iou(gt, pred):
    result = []
    data = {}
    thres_hold = 0.086

    for i in range(0, len(gt)):
        for j in range(0, len(gt)):
            try:
                if((abs(gt[i][4] - pred[j][4])) < thres_hold and ((abs(gt[i][5] - pred[j][5])) < thres_hold)):
                    try:
                        x1 = max(gt[i][0], pred[j][0])
                        y1 = max(gt[i][1], pred[j][1])
                        x2 = min(gt[i][2], pred[j][2])
                        y2 = min(gt[i][3], pred[j][3])

                        inter_Area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

                        gt_Area = (gt[i][2] - gt[i][0] + 1) * (gt[i][3] - gt[i][1] + 1)
                        pred_Area = (pred[j][2] - pred[j][0] + 1) * (pred[j][3] - pred[j][1] + 1)
                    except:
                        # result.append({i: "N/A"})
                        pass
                    else:
                        if(inter_Area == 0.0):
                            continue
                        try:
                            if(float(data[pred[j][6]]) <= inter_Area / float(gt_Area + pred_Area - inter_Area) * 100):
                                data[pred[j][6]] = inter_Area / float(gt_Area + pred_Area - inter_Area) * 100
                        except:
                            data[pred[j][6]] = inter_Area / float(gt_Area + pred_Area - inter_Area) * 100
            except:
                # result.append({i: "N/A"})
                pass
            else:
                pass

    for id, iou in data.items():
        result.append("{}: {}".format(id, str(iou) + "%\n"))
        
    return result

with open("./result.json", "r") as result_json:
    result = json.load(result_json)

    for i in range(0, len(result)):
        file_name = result[i]['filename'][13:-4]
        image = cv2.imread("./drive/" + file_name + ".png")
        f = open("./drive/" + file_name + ".txt")
        data = f.readlines()
        f.close()

        gt = []
        pred = []

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

            gt.append([x1, y1, x2, y2, center_x, center_y])
        
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
            cv2.putText(image, "ID: {}".format(j), (x1, y1),
		    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)

            pred.append([x1, y1, x2, y2, center_x, center_y, j])

        sorted(gt, key=lambda gt: gt[4])
        sorted(pred, key=lambda pred: pred[4])

        iou_Result = iou(gt, pred)
        
        f = open("./iou/" + file_name + ".txt", 'w')
        
        for i in range(0, len(iou_Result)):
            f.write(str(iou_Result[i]))

        f.write('\n')
        f.write("GT Length: " + str(len(gt)) + ' ' + "Pred Length: " + str(len(pred)))
        
        f.close()

        cv2.imwrite("./test/" + "test_" + file_name + ".png", image)

