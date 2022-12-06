import numpy as np
import cv2
from collections import deque
import time


img1 = np.random.choice((0,255),(10,10),True,[0.24,0.76])
array = np.array(img1,dtype=np.uint8)

# Dimensions of the new maze

scale_percent = 1000
width = int(array.shape[1] * scale_percent / 100)
height = int(array.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resizing the image

img = cv2.resize(array,dim , interpolation = cv2.INTER_AREA)

# Making sure that the image has only white and black pixels

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] >= 127:
            img[i][j] = 255
        else:
            img[i][j] = 0

# Marking the start and end point with a grey pixel

img[1][1] = 128
img[98][98]= 128

# Size of the resized maze

print(img.shape)

# Original Maze

cv2.namedWindow('Orig',cv2.WINDOW_NORMAL)
cv2.imshow('Orig',array.astype(np.uint8))

# Resized Maze

cv2.namedWindow('MAZE',cv2.WINDOW_NORMAL)
cv2.imshow('MAZE',img.astype(np.uint8))


starttime = time.time()
# Implementation of BFS using queue

class Node():
    def __init__(self, index, parent):
        self.x = index[0]
        self.y = index[1]
        self.parent = parent

def showPath(end, start):
    current = end
    while(current != start):    
        img[current.x][current.y] = 190
        current = current.parent
    
def bfs(start):  
    q = deque()
    q.append(start)
    
 
    cv2.namedWindow('path', cv2.WINDOW_NORMAL)
    while len(q):
        current = q.popleft()
        i, j = current.x, current.y
        cv2.imshow('path', img)
        cv2.waitKey(1)
        
        if j+1 < img.shape[1]:     
            if (img[i][j+1] != 0).any() and (img[i][j+1] != 127).any():
                if [i,j] == [98,98]:
                    break   
                
                img[i][j+1] = 127  
                n = Node((i, j+1), current)
                q.append(n)
        if i+1 < img.shape[0]:
            if (img[i+1][j] != 0).any() and (img[i+1][j] != 127).any():
                if [i,j] == [98,98]:
                    break
                
                img[i+1][j] = 127
                n = Node((i+1, j), current)
                q.append(n)
        if j-1 > 0:
            if (img[i][j-1] != 0).any() and (img[i][j-1] != 127).any():
                if [i,j]==[98,98]:
                    break
                
                img[i][j-1] = 127
                n = Node((i, j-1), current)
                q.append(n)
        if i-1 > 0:
            if (img[i-1][j] != 0).any() and (img[i-1][j] != 127).any():
                if [i,j]==[98,98]:
                    break
                
                img[i-1][j] = 127
                n = Node((i-1, j), current)
                q.append(n)
                        
    showPath(current, start)


bfs(Node((1,1),None))

finishtime = time.time()



print('Start - (1,1)')
print('End - (98,98)')
print('Time taken for BFS : ', finishtime-starttime)
cv2.namedWindow('final', cv2.WINDOW_NORMAL)
cv2.imshow("final", img)
cv2.waitKey(0)