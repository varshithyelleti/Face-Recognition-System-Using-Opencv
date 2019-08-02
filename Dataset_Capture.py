
import cv2
import os
import sys
import PIL
#color=sys.stdout.shell
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Successfully Created the Path for the Employee")
        return True
    else:
        d=len(list(os.listdir(path)))
        print(d)
        if(d>30):
            print("DataSet for the Current Employee already Exists ")
            return False
        else:
            print("Successfully Created the Path for the Employee")
            return True

CodeExitFlag=0     
#Employee Details
Employee_Name =input('Enter Name : ')        
Employee_ID=input('Enter ID   : ')
EmployeePath='DataSet2/'+Employee_ID+"/"

CheckFlag=assure_path_exists(EmployeePath)
if(CheckFlag==True):
    fp=open('Employees_data.txt','a+')
    fp.write(Employee_ID+" "+Employee_Name+"\n")
    fp.close()
    
else:
    print("Creation of the DataSet Exited")
    CodeExitFlag=1
    
if(CodeExitFlag==0):
    
    #Initializing the Video for Capturing
    VideoMonitor = cv2.VideoCapture(0)
    
    #Object Detection usimg Haarcascade forntalface
    FaceDetection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Initialization of the Image sample count
    count = 0
    
    #Creation of the folder dataset if Not there
    
    while(True):
        #Capturing the Video Frame
        _, ImageFrame = VideoMonitor.read()
    
        # Conversion of the Frame into gray scale 
        gray = cv2.cvtColor(ImageFrame, cv2.COLOR_BGR2GRAY)
        Samples = FaceDetection.detectMultiScale(ImageFrame, 1.5, 5)
    
        for (x,y,w,h) in Samples:
    
            cv2.rectangle(ImageFrame, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            #print(ImageFrame.size())
            #Saving the Frame as 'face_id.count.jpg' format in the dataset folder
            cv2.imwrite(EmployeePath+"User." + str(Employee_ID) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('Frame', ImageFrame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif count>50:
            break
    
    VideoMonitor.release()
    cv2.destroyAllWindows()
