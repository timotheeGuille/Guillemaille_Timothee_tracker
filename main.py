import cv2
import numpy as np
import param

import yolo
import sort
import time
import team

from math import floor
from unittest.mock import seal
import numpy as np




#redefine imshow 
def imshow(img,name):
    imgSmall=cv2.resize(img,(0,0),fx=0.90,fy=0.9)
    cv2.imshow(name,imgSmall)



#get input video and img
cap = cv2.VideoCapture(param.video2InPath)
soccerField=cv2.imread(param.soccerField)


if (cap.isOpened()== False):
  print("Error opening video stream or file")



fr=cap.get(4)/soccerField.shape[0]
soccerR=cv2.resize(soccerField,(0,0),fx=fr,fy=fr)

frame_width = int(cap.get(3)+soccerR.shape[1])
frame_height = int(cap.get(4))

prev_frame_time = 0
new_frame_time = 0

#define the video output saver
out = cv2.VideoWriter('output/outpy001.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))


#Sort tracker
mot_tracker1 = sort.Sort(max_age=100, min_hits=1, iou_threshold=0.01)


#matrix of fixed Point
pts1=np.float32([[171,319],[1237,346],[700,443],[64,404]])
pts2=np.float32([[53,28],[1223,28],[639,406],[178,406]])


teamColor=team.TeamColor()

nbframe=0
while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
        
    nbframe+=1
    cv2.imwrite("img.png",frame)

    boxes,indices,class_ids,track_bbs_ids,playersPositionX,playersPositionY=yolo.applyYolo(frame,mot_tracker1)

   

    playersColor=teamColor.predictTeamPlayers(frame,boxes)
  

    #draw player boxes on video
    frameLabel=yolo.draw_predictions(frame,boxes,indices,class_ids,track_bbs_ids,playersColor)


    # Display the resulting frame
    if(param.showFixedpoint):
      cv2.circle(frameLabel, (171,319), 10, (255,0,0), 2)
      cv2.circle(frameLabel, (1237,346), 10, (255,255,0), 2)
      cv2.circle(frameLabel, (700,443), 10, (255,0,255), 2)
      cv2.circle(frameLabel, (64,404), 10, (255,0,255), 2)

    imshow(frameLabel,"1")
 
    #compute perspective matrix and apply to all player position
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    centersWrap=[yolo.warp_point(playersPositionX_i,playersPositionY_i,matrix) for playersPositionX_i,playersPositionY_i in zip(playersPositionX,playersPositionY) ]
    centersWrapX=list(zip(*centersWrap))[0]
    centersWrapY=list(zip(*centersWrap))[1]
  
    #draw player on bird view
    soccerField=cv2.imread(param.soccerField)
    soccerFieldLabel=yolo.draw_player(soccerField,centersWrapX,centersWrapY,playersColor)

    imshow(soccerField,"2")

    #concat 2 video. show and save the video 

    fr=frameLabel.shape[0]/soccerField.shape[0]
    soccerR=cv2.resize(soccerField,(0,0),fx=fr,fy=fr)

    v_img = cv2.hconcat([soccerR, frameLabel])

    imshow(v_img,"3")
    out.write(v_img)


    #print fps 
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    print("nbframe=",nbframe,"  fps=",fps)

  # Press Q  to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break


cap.release()

# Closes all the frames

cv2.destroyAllWindows()






    