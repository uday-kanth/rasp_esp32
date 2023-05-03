import cv2 


import time
from websocket import create_connection
import json
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
class_names = ["Bread","CardBoard","Metal Can","Plastic Bag"]
wei="C:/Users/91905/OneDrive/Desktop/vs code/rasp_esp32/yolov4.weights"
cf="C:/Users/91905/OneDrive/Desktop/vs code/rasp_esp32/yolov4.cfg"
net = cv2.dnn.readNet(wei,cf)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
ws = create_connection("wss://gtg3p8yh66.execute-api.us-east-1.amazonaws.com/production/")

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        # print(check) #prints true as long as the webcam is running
        # print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            # img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            # img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            # img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            # print("Converting RGB image to grayscale...")
            # gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            # print("Converted RGB image to grayscale...")
            # print("Resizing image to 28x28 scale...")
            # img_ = cv2.resize(gray,(28,28))
            # print("Resized...")
            # img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            print("Image saved!")

            img = cv2.imread("saved_img.jpg")
            classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            res=[]
            for (classid, score, box) in zip(classes, scores, boxes):
        
                label = "%s : %f" % (class_names[classid], score)
        
                res.append(label)
            print(res)
            mess= {
                    "action": "sendMessage",
                    "message": res[0]
                    }
            mess= json.dumps(mess)
            ws.send(mess)   
            print("Sent")
            print("Receiving...")
            result =  ws.recv()
            print("Received '%s'" % result)
            ws.close() 

            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
    