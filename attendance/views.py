from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
# %%
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime



# from PIL import ImageGrab

# Create your views here.
from django.templatetags.static import static
from django.conf import settings

from attendance_system.settings import BASE_DIR
from django.core.files.storage import FileSystemStorage
from django.conf.urls.static import static

def show_index(request):
    if request.method == 'POST':
        do_face_recognition()

    template_name = 'index.html'
    context = {}

    return render(request, template_name, context)    




    
def do_face_recognition():    

    path = settings.IMAGES_ROOT
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    # %%

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def markAttendance(name):
        attendance_path = settings.ATTENDANCE_ROOT
        with open(attendance_path, 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

    #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
    # def captureScreen(bbox=(300,300,690+300,530+300)):
    #     capScr = np.array(ImageGrab.grab(bbox))
    #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    #     return capScr

    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # img = captureScreen()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for faceLoc, face_encoding in zip(facesCurFrame, encodesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, face_encoding)

            name = 'Unkwown'
            faceDis = face_recognition.face_distance(encodeListKnown, face_encoding)
            # print(faceDis)

            matchIndex = np.argmin(faceDis)
            #   if matches[matchIndex]:
            #     name = classNames[matchIndex].upper()
            #   else: name = 'Unknown'
            # #print(name)
            #   y1,x2,y2,x1 = faceLoc
            #   y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            #   cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #   cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            #   cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #   markAttendance(name)
            if faceDis[matchIndex]:
                name = classNames[matchIndex]

            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

        # print(name)

        cv2.imshow('Webcam_facerecognition', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # %%

  
