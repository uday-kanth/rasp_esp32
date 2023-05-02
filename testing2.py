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
img = cv2.imread("C:/Users/91905/OneDrive/Desktop/vs code/smartcampus/piyush.jpg")
classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
res=[]
for (classid, score, box) in zip(classes, scores, boxes):
        
        label = "%s : %f" % (class_names[classid], score)
        
        res.append(label)
print(res)
# ws = create_connection("wss://44n9516al9.execute-api.us-east-1.amazonaws.com/demo/")
# print(ws.recv())
# print("Sending 'Hello, World'...")
# # mess =  '{ "action":"sendMessage", "message":"hehehe"}'
# # mess= json.loads(mess)
# mess= {
#   "action": "sendMessage",
#   "message": res[0]
# }
# mess= json.dumps(mess)
# ws.send(mess)
# print("Sent")
# print("Receiving...")
# result =  ws.recv()
# print("Received '%s'" % result)
# ws.close()        