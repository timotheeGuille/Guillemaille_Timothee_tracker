from math import floor
from unittest.mock import seal
import numpy as np
import cv2
from sklearn.cluster import KMeans, k_means


class TeamColor :
    def __init__(self):
        self.bufferPlayerMeanColor=[]
        self.kmeans =None
        self.frameCounter=0

    def updateTeamColor(self):

        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(self.bufferPlayerMeanColor)

    def addColorBuffer(self,meanColors):
        self.bufferPlayerMeanColor+=meanColors
        sizeExcess=max(len(self.bufferPlayerMeanColor)-500,0)
        self.bufferPlayerMeanColor=self.bufferPlayerMeanColor[sizeExcess:]
 

    def predictTeamPlayers(self,frame,boxes) :

        players=self.getPlayer(frame,boxes)
        meanColors=self.getMeanColors(players)

        self.addColorBuffer(meanColors)

        if(self.frameCounter%20 == 0 or self.frameCounter<20):
            self.updateTeamColor()
        self.frameCounter+=1
            
        playersLabel=self.kmeans.predict(meanColors)
        color1=self.kmeans.cluster_centers_[0]
        color2=self.kmeans.cluster_centers_[1]
        playersColor=[color1 if item == 0 else color2 for item in playersLabel]

        return playersColor

    def getPlayer(self,frame,boxes):
            
        players=[]

        for b in boxes:
            (x,y,w,h)=b
            h=np.ceil(h)
            w=np.ceil(w)
            x1=int(x+w/2-w/10)
            x2=int(x+w/2+w/10)
            y1=int(y+h/2-h/10)
            y2=int(y+h/2+h/10)
            player=frame[y1:y2,x1:x2]
            players.append(player)

        return players #(list img)


    def getMeanColors(self,players):
        
        meanColors =[]
        for p in players:
            R=p[:,:,0]
            G=p[:,:,1]
            B=p[:,:,2]

            meanR=np.mean(R.reshape((-1)))
            meanG=np.mean(G.reshape((-1)))
            meanB=np.mean(B.reshape((-1)))
            meanColor=(meanR,meanG,meanB)
            meanColors.append(meanColor)
        return meanColors












