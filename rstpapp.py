import numpy as np
import face_rec
import cv2
import urllib.request as u



url = 'rtsp://192.168.0.102:8080/h264_ulaw.sdp'

video = cv2.VideoCapture(url)




#loading sample pictures
f1 = face_rec.load_image_file('faces/modi.jpg')
f3 = face_rec.load_image_file('faces/trump.jpg')
f5 = face_rec.load_image_file('faces/kamal.jpg')


#learn how to recognise it
f1_encoding =  face_rec.face_encodings(f1)[0]
f3_encoding =  face_rec.face_encodings(f3)[0]
f5_encoding =  face_rec.face_encodings(f5)[0]

#array for known encodings
kfe = [
    f1_encoding,
    f3_encoding,
    f5_encoding
]

#array for known face names
kfn = [
    'Modi Ji',
    'Trump',
    'Kamal Haasan'
]



names = []
flag = True
floc = []
fe = []
count = 0
while(1):

    ret,frame = video.read() #grab frame by frame while(1)

    rframe = cv2.resize(frame,(0,0),fx=0.25,fy=0.25) #not needed ,
                                                    #just to make the process faster 

    rgbrframe = cv2.cvtColor(rframe,cv2.COLOR_BGR2RGB)#cv2 uses BGR color whereas,
                                #face_rec uses RGB , so reverse content

    if flag:
        floc = face_rec.face_locations(rgbrframe) # grab face from frame
        fe   = face_rec.face_encodings(rgbrframe,floc) # grab face encodings from frame 
        
        names = []
        for fenc in fe:
            matched_faces = face_rec.compare_faces(kfe,fenc)
            name = 'Unknown'

            fdist = face_rec.face_distance(kfe,fenc)
            best_match = np.argmin(fdist)
            if matched_faces[best_match]:
                name = kfn[best_match]

            names.append(name)
    flag = not flag

    # Display the results
    for (top, right, bottom, left), name in zip(floc, names):
        top *= 4    # resize image back again by *0.25
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)# Draw a box around the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (136, 227, 182), 1) #label the face

    

    h,w,l = frame.shape
    new_h = int(h/2)
    new_w = int(w/2)
    rzframe = cv2.resize(frame,(new_w,new_h))
    cv2.imshow('Video', frame)
    count += 1
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('processed ',count,'frames')
        video.release()
        cv2.destroyAllWindows()
        break

