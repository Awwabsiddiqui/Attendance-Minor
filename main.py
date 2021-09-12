import cv2
import numpy as np
import face_recognition

imgawab = face_recognition.load_image_file('images/awab.jpg')
imgawab=cv2.cvtColor(imgawab , cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('images/test.jpg')
imgtest=cv2.cvtColor(imgtest , cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgawab)[0]
encawab=face_recognition.face_encodings(imgawab)[0]
cv2.rectangle(imgawab , (faceloc[3] , faceloc[0]),(faceloc[1] , faceloc[2]) , (255,0,255),2)

faceloctest=face_recognition.face_locations(imgtest)[0]
enctest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest , (faceloctest[3] , faceloctest[0]),(faceloctest[1] , faceloctest[2]) , (255,0,255),2)

result =face_recognition.compare_faces([encawab] , enctest)
facedis=face_recognition.face_distance([encawab] , enctest)
print(result , facedis)

cv2.putText(imgtest , f'{result} {round(facedis[0] , 2)}' , (50,50) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,0,255) , 2)
cv2.imshow('awab' , imgawab)
cv2.imshow('test' , imgtest)
cv2.waitKey(0)