import cv2
import sys
import torch
import numpy as np

#model = tf.keras.models.load_model("model_mask.h5")
#model2 = tf.keras.models.load_model("model_mask_2.h5")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # local model


####################### IMAGE ############################
# img = "feu_1.jpg"

# result = model(img)
# result.print()

# result.save()



########################## VIDEO/WEBCAM ########################

webcam = 0
video = 'fire.mp4'

video_capture = cv2.VideoCapture(video)

#dictionnaire
labels_dict = {
    'fire':0,
    'no fire':1
}

color = (0, 110, 127)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Model prediction 
    result = model(frame)
    label, confidence, coord = result.xyxy[0][:, -1], result.xyxy[0][:, -2], result.xyxy[0][:, :-2] 

    for i in range(len(result.xyxy[0])):
        x,y,w,h = coord[i]
        x = int(x.item())
        y = int(y.item())
        w = int(w.item())
        h = int(h.item())
        
        #get the label
        value = label[i]
        lab = None
        for k, val in labels_dict.items():
            if value.item() == val:
                lab = k

        #get the confidence
        conf = confidence[i]
        conf = round(float(conf.item()), 2)

        #drawing in the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), color)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
        cv2.putText(frame, f"{lab} : {conf}", (x + 10, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, .5, (0,0,0))

        cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()