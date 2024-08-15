import cv2
from ultralytics import YOLO
import cvzone
import math
import time
model=YOLO("weights/best.pt")
cap=cv2.VideoCapture(0)

#cap=cv2.VideoCapture('traffic-sign-to-test.mp4')

#cap=cv2.imread('road463.png')
cap.set(3,1280)
cap.set(4,720)


classnames=["100km","120km","20km","30km",
        "50km","60km","70km","80km",
        "agıraraçsollamakyasa","araçgirişiyok","asgari80km","ağıraraçgirişiyasak",
        "ağıraraçsollamakserbest","bisikletliçıkabilir","dikkat","dur",
        "girişyok","gizlibuzlanma","hızsınırı yok","ilerivesağadönüş",
        "ilerivesoladönüş","ileriyön","kasisliyol","kavşak",
        "kayganyol","sağadönüş","sağdangiriş","sağdansollamakyasak",
        "soladönüş","soldangiriş","sollamakserbest","sollamakyasak",
        "traffic-sign","trafikışığıvar","vahşihayvan","virajlıyol",
        "yayageçidi","yolver","çalışmavar","çiftgirişliyol",
        "çocukçıkabilir","öncelikliyol"]


per_frame_time=0
new_frame_time=0

while True:
    new_frame_time=time.time()
    success,img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

            w,h=x1-x2,y1-y2
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf=math.ceil(box.conf[0]*100)/100
            cls=int(box.cls[0])
            cvzone.putTextRect(img,f'{classnames}{conf}{x1}',(max(0,x1),max(35,y1)),scale=1,thickness=1)

    #fps=1/(new_frame_time-per_frame_time)
    #per_frame_time=new_frame_time
    #print("fps:",fps)
    fps=cap.get(cv2.CAP_PROP_FPS)
    print("FPS:",fps)

    cv2.imshow("İmage",img)
    cv2.waitKey(1)

