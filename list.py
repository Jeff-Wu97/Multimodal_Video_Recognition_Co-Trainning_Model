import os
import random


rootpath=r"C:\Users\JGao9\Desktop\Ego15"#存帧的根目录
wroot="/home/s1924153"#list中的根目录，即数据集在谷歌云中存放的位置
ftr=open("train.list",'w')
fte=open("test.list","w")

classdir=os.listdir(rootpath)
classdir.sort()
count=0
for eachclass in classdir:
    classpath=os.path.join(rootpath,eachclass)
    
    videodir=os.listdir(classpath)
    videodir.sort()
    for video in videodir:
        randnum=random.randint(1,1000)
        videopath=os.path.join(classpath,video)
        wpath=[wroot,eachclass,video]
        wpath='/'.join(wpath)
        if (randnum>250):
            ftr.write(wpath+" "+str(count)+"\n")
        else:
            fte.write(wpath+" "+str(count)+"\n")
    count=count+1
ftr.close()
fte.close()
            
