
import numpy as np
import cv2
import param

def main():

 classes = None

 with open(param.classesFilePath, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
 # generate different colors for different classes 
 COLORS = np.random.uniform(250, 250, size=(len(classes), 3))
 
 
 net = cv2.dnn.readNet(param.yoloWeights, param.yoloConfig)


 #cap = cv2.VideoCapture(0)
 #
 cap = cv2.VideoCapture(param.videoInPath)

 if (cap.isOpened()== False):
   print("Error opening video stream or file")


 while(cap.isOpened()):

   # Capture frame-by-frame
   ret, frame = cap.read()
   if ret == True:
     frameLabel=applyYolo(frame,net,classes, COLORS)
     # Display the resulting frame
     cv2.imshow('Frame',frameLabel)
     # Press Q on keyboard to  exit

     if cv2.waitKey(25) & 0xFF == ord('q'):
       break
   else:
     break


 cap.release()
 # Closes all the frames
 cv2.destroyAllWindows()



#******************************************************#
def applyYolo(imgIn,net,classes, COLORS):
    Width = imgIn.shape[1]
    Height = imgIn.shape[0]
    scale = 0.00392

    
    
    # create input blob 
    blob = cv2.dnn.blobFromImage(imgIn, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(imgIn, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),COLORS,classes)


    return imgIn


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,color,classes):

    label = str(classes[class_id])

    color = color[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



main()
