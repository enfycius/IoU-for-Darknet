import numpy as np
import cv2
import json
import math

def iou(gt, pred):
    result = []
    items = []
    tp = {}
    fp = {}
    fn = {}
    data = {}
    ft_gt = {}
    thres_hold = 0.04
    length = None
    alpha = 0.5

    if(len(gt) < len(pred)):
        length = len(pred)
    else:
        length = len(gt)

    for i in range(0, length):
        for j in range(0, length):
            try:
                if(math.sqrt((gt[i][4] - pred[j][4]) * (gt[i][4] - pred[j][4]) + (gt[i][5] - pred[j][5]) * (gt[i][5] - pred[j][5])) < thres_hold):
                    if(gt[i][7] == pred[j][6]):
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
                                if(float(data[gt[i][6]]) <= inter_Area / float(gt_Area + pred_Area - inter_Area) * 100):
                                    data[gt[i][6]] = {gt[i][7] : inter_Area / float(gt_Area + pred_Area - inter_Area) * 100}
                                    cv2.rectangle(pred[j][7], (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    cv2.putText(pred[j][7], "MID: {}".format(gt[i][6]), (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

                                    cv2.rectangle(gt[i][8], (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    cv2.putText(gt[i][8], "MID: {}".format(gt[i][6]), (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

                            except:
                                data[gt[i][6]] = {gt[i][7] : inter_Area / float(gt_Area + pred_Area - inter_Area) * 100}
                                cv2.rectangle(pred[j][7], (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(pred[j][7], "ID: {}".format(gt[i][6]), (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

                                cv2.rectangle(gt[i][8], (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(gt[i][8], "ID: {}".format(gt[i][6]), (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
                                
            except:
                # result.append({i: "N/A"})
                pass
            else:
                pass

    for i in range(0, len(gt)):
        try:
            ft_gt[gt[i][7]]
        except:
            ft_gt[gt[i][7]] = 0
            ft_gt[gt[i][7]] += 1
        else:
            ft_gt[gt[i][7]] += 1

    tp_prev = {}
    fp_prev = {}

    for id, dat in data.items():
        for classes, iou in dat.items():
            tp[classes], tp_prev[classes] = TP(tp, classes, tp_prev, iou, alpha)
            fp[classes], fp_prev[classes] = FP(fp, classes, fp_prev, iou, alpha)
            fn[classes] = FN(fn, ft_gt, classes, iou, alpha)
         
            items.append("ID: {}, Classes: {}, IoU: {}".format(id, classes, str(iou)))

    # fn = FN(fn, ft_gt[:])

    if pred == []:
        for i in range(0, len(gt)):
            try:
                fn[gt[i][7]]
            except:
                fn[gt[i][7]] = 1
            else:
                fn[gt[i][7]] += 1

    result.append([tp, fp, fn, items])
    
    return result

def TP(tp, classes, prev, iou, alpha):
    try:
        prev[classes]
    except:
        prev[classes] = 0

    if(iou >= (alpha*100)):
        try:
            tp[classes] += 1
            prev[classes] += 1
        except:
            tp[classes] = 1
            prev[classes] += 1
    else:
        try:
            tp[classes]
        except:
            tp[classes] = prev[classes]

    return [tp[classes], prev[classes]]
            
def FP(fp, classes, prev, iou, alpha):
    try:
        prev[classes]
    except:
        prev[classes] = 0
    
    if(iou < (alpha*100)):
        try:
            fp[classes] += 1
            prev[classes] += 1
        except:
            fp[classes] = 1
            prev[classes] += 1
    else:
        try:
            fp[classes]
        except:
            fp[classes] = prev[classes]

    return [fp[classes], prev[classes]]

def FN(fn, ft_gt, classes, iou, alpha):
    if(iou >= (alpha*100)):
        ft_gt[classes] -= 1
        fn[classes] = ft_gt[classes]
    else:
        fn[classes] = ft_gt[classes]
    
    return fn[classes]

def Precision(tp_t, fp_t):
    precision = {}
    tp_s = {}
    fp_s = {}

    for tp_f in tp_t:
        for classes, tp_c in tp_f.items():
            try:
                tp_s[classes]
            except:
                tp_s[classes] = tp_c
            else:
                tp_s[classes] += tp_c

    for fp_f in fp_t:
        for classes, fp_c in fp_f.items():
            try:
                fp_s[classes]
            except:
                fp_s[classes] = fp_c
            else:
                fp_s[classes] += fp_c

    for classes, tp_c in tp_s.items():
        try:
            precision[classes]
        except:
            precision[classes] = tp_c / (tp_c + fp_s[classes])
    
    return precision
    
def Recall(tp_t, fn_t):
    recall = {}
    tp_s = {}
    fn_s = {}

    for tp_f in tp_t:
        for classes, tp_c in tp_f.items():
            try:
                tp_s[classes]
            except:
                tp_s[classes] = tp_c
            else:
                tp_s[classes] += tp_c

    for fn_f in fn_t:
        for classes, fn_c in fn_f.items():
            try:
                fn_s[classes]
            except:
                fn_s[classes] = fn_c
            else:
                fn_s[classes] += fn_c

    for classes, tp_c in tp_s.items():
        try:
            recall[classes]
        except:
            recall[classes] = tp_c / (tp_c + fn_s[classes])
    
    return recall

def F1(precision, recall):
    f1 = {}

    for classes, precision in precision.items():
        try:
            f1[classes]
        except:
            f1[classes] = (precision * recall[classes]) / ((precision + recall[classes]) / 2)

    return f1

with open("./result.json", "r") as result_json:
    result = json.load(result_json)
    tp_t = []
    fp_t = []
    fn_t = []

    for i in range(0, len(result)):
        file_name = result[i]['filename'][13:-4]
        image = cv2.imread("./drive/" + file_name + ".png")
        f = open("./drive/" + file_name + ".txt")
        data = f.readlines()
        f.close()

        gt = []
        pred = []

        for j in range(0, len(result[i]['objects'])):
            class_id = result[i]['objects'][j]['class_id']
            center_x = result[i]['objects'][j]['relative_coordinates']['center_x']
            center_y = result[i]['objects'][j]['relative_coordinates']['center_y']
            width = result[i]['objects'][j]['relative_coordinates']['width']
            height = result[i]['objects'][j]['relative_coordinates']['height']

            x1 = round(image.shape[1] * center_x) - round(image.shape[1] * width / 2)
            y1 = round(image.shape[0] * center_y) - round(image.shape[0] * height / 2)
            x2 = round(image.shape[1] * center_x) + round(image.shape[1] * width / 2)
            y2 = round(image.shape[0] * center_y) + round(image.shape[0] * height / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            pred.append([x1, y1, x2, y2, center_x, center_y, class_id, image])
        
        for j in range(0, len(data)):
            class_id = int(data[j].split(' ')[0])
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

            gt.append([x1, y1, x2, y2, center_x, center_y, j, class_id, image])

        sorted(gt, key=lambda gt: gt[4])
        sorted(pred, key=lambda pred: pred[4])

        iou_Result = iou(gt, pred)
        
        f = open("./iou/" + file_name + ".txt", 'w')

        tp_t.append(iou_Result[0][0])
        fp_t.append(iou_Result[0][1])
        fn_t.append(iou_Result[0][2])
        
        f.write("Classes, TP: " + str(iou_Result[0][0]) + '\n')
        f.write("Classes, FP: " + str(iou_Result[0][1]) + '\n')
        f.write("Classes, FN: " + str(iou_Result[0][2]) + '\n')
        f.write(str(iou_Result[0][3]) + '\n')

        f.write("GT Length: " + str(len(gt)) + ' ' + "Pred Length: " + str(len(pred)))
        
        f.close()

        cv2.imwrite("./test/" + "test_" + file_name + ".png", image)

    precision = Precision(tp_t, fp_t)
    recall = Recall(tp_t, fn_t)
    f1 = F1(precision, recall)

    print(precision)
    print(recall)
    print(f1)

    f = open("./iou/" + "result" + ".txt", 'w')

    f.write("Precision: " + str(precision) + '\n')
    f.write("Recall: " + str(recall) + '\n')
    f.write("F1: " + str(f1))

    f.close()