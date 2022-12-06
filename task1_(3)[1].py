
import numpy as np
import cv2
import time


img = np.random.choice((0,255),(10,10),True,[0.35,0.65])
array = np.array(img,dtype=np.uint8)

# Dimensions of the new maze

scale_percent = 1000
width = int(array.shape[1] * scale_percent / 100)
height = int(array.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resizing the image

resized = cv2.resize(array,dim , interpolation = cv2.INTER_AREA)

# Making sure that the image has only white and black pixels

for i in range(resized.shape[0]):
    for j in range(resized.shape[1]):
        if resized[i][j] >= 127:
            resized[i][j] = 255
        else:
            resized[i][j] = 0

# Marking the start and end point with a grey pixel

resized[1][1] = 127
resized[98][98]= 127

# Size of the resized maze

print(resized.shape)

# Elements of the original array

print(array)

# Original Maze

cv2.namedWindow('Orig',cv2.WINDOW_NORMAL)
cv2.imshow('Orig',array.astype(np.uint8))

# Resized Maze

cv2.namedWindow('MAZE',cv2.WINDOW_NORMAL)
cv2.imshow('MAZE',resized.astype(np.uint8))
 
starttime = time.time()

# Dijkstra's Algorithm 

start = (1,1)
end = (98,98)

def distance(x,y):
    return (x[0]-y[0])**2 + (x[1]-y[1])**2

def acceptable(img,x,y):
    return (x>=0 and y>=0 and x<img.shape[0] and y<img.shape[1])

def dijkstra(img,start,end):
    dist = np.full((100,100),np.inf)
    dist[start]=0
    parent = np.zeros((100,100,2))
    visited = np.zeros((100,100))
    current =start
    visited[start]=1
    while current != end:
        print(current)
        visited[current]=1
        for i in range(-1,2):
            for j in range(-1,2):
                point= (current[0]+i,current[1]+j)
                if acceptable(img,point[0],point[1]) and visited[point] != 1 and img[point] != 0 :
                    if distance(point,current)+ dist[current] < dist[point]:
                        dist[point]= distance(point,current) + dist[current]
                        
                        parent[point[0],point[1]]=current[0],current[1]
                        

        min = np.inf
        for i in range(100):
            for j in range(100):
                if min > dist[i,j] and visited[i,j] != 1:
                    min = dist[i,j]
                    current = (i,j)
        showPath(img,current,start,parent)



def showPath(img,current,start,parent):
    new=np.copy(img)
    while current != start:
        var = int(parent[current][0]), int(parent[current][1])
        new[int(var[0]),int(var[1])] = 127
        current = (var[0],var[1])
    
    cv2.namedWindow('Window',cv2.WINDOW_NORMAL)
    cv2.imshow('Window',new)
    cv2.waitKey(1)

dijkstra(resized,start,end)

endtime = time.time()

print('Time taken for Dijkstra Algo :', endtime-starttime)
cv2.waitKey(0)
cv2.destroyAllWindows()
