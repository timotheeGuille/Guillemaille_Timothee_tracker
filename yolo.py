import numpy as np
import cv2
import param
import sort


def applyYolo(imgIn,mot_tracker1):
    Width = imgIn.shape[1]
    Height = imgIn.shape[0]
    scale = 0.00392

    classes = None

    with open(param.classesFilePath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    
    # read pre-trained model and config file
    net = cv2.dnn.readNet(param.yoloWeights, param.yoloConfig)
    
    # create input blob 
    blob = cv2.dnn.blobFromImage(imgIn, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    dets=[]
    centers_x=[]
    centers_y=[]
    playersPositionX=[]
    playersPositionY=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and (class_id==0 or class_id==29) :
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                dets.append([round(x),round(y), round(x+w), round(y+h),confidence])
                playersPositionX.append(center_x)
                playersPositionY.append(center_y + h / 2)



    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    #update sort Tracker

    track_bbs_ids = mot_tracker1.update(dets)

    return boxes,indices,class_ids,track_bbs_ids,playersPositionX,playersPositionY


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_predictions(imgIn,boxes,indices,class_ids,track_bbs_ids,colorPlayers):
        for i in indices:
          if i<len(track_bbs_ids):
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            id=track_bbs_ids[i][4]
            draw_prediction(imgIn, class_ids[i],round(x), round(y), round(x+w), round(y+h),id,colorPlayers[i])

        return imgIn

def draw_prediction(img, class_id,x, y, x_plus_w, y_plus_h,id,color):

    label = str("human") + str(" n=")+str(id) 


    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_players(imgIn,centers_x,centers_y,indices,playersColor):
    for i in indices:
            centerX=centers_x[i]
            centerY=centers_y[i]
            draw_player(imgIn,centerX,centerY ,playersColor)

    return imgIn

def draw_player(img,x, y,playersColor):
    for xi,yi,color in zip(x,y,playersColor):
        cv2.circle(img, (xi,yi), 10, color, 2)

def warp_point(x, y,matrix) :
    d = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]

    return (
        int((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / d), # x
        int((matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / d), # y
    )


