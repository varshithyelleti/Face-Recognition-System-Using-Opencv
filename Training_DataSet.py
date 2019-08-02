
import os,cv2;
import numpy as np
from PIL import Image;

def getImagesAndLabels(path):
    
    #Path of the DataSet Folder
    ImagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #print("1",ImagePaths)
    #Creation of FaceSamples list
    FaceSamples=[]
    
    #Creation of IDs list
    ID=[]
    print(len(ImagePaths))
    
    for ImagePaths_2 in os.walk(path):
        SubDirectoryPath=os.path.split(ImagePaths_2[0])[-1]
        EntirePath=path+SubDirectoryPath+'/'
        print("2",ImagePaths_2)
    #Iterating through all the ImagePaths and loading the Ids and the images
        for I_Path in ImagePaths_2[2]:
            
            #Conversion of the Sample Frame into Grayscale
            PilImage=Image.open(EntirePath+I_Path).convert('L')
            
            #Conversion of the Sample Frame into Array for Metrics
            ImageNp=np.array(PilImage,'uint8')
            
            Id=int(os.path.split(ImagePaths[0])[-1])
            #Sample Frame is of Form 'ID.count'
            #Id=int(os.path.split(I_Path)[-1].split(".")[1])
            
            # Extraction 
            Faces=Detector.detectMultiScale(ImageNp)
            #If a face is there then append that in the list as well as Id of it
            for (x,y,w,h) in Faces:
                FaceSamples.append(ImageNp[y:y+h,x:x+w])
                ID.append(Id)
    return FaceSamples,ID
    
    return FaceSamples,ID
Recognizer = cv2.face.LBPHFaceRecognizer_create()
Detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
Employee_Faces,IDs = getImagesAndLabels('DataSet2/')

TrainedModel = Recognizer.train(Employee_Faces, np.array(IDs))
print(TrainedModel)
#color.write("Successfully Trained the Recognizer Model\n","STRING")
print("Successfully Trained the Recognizer Model")

Recognizer.save('trainer/trainer.yml')
