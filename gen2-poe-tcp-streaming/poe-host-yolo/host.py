import socket
import re
import cv2
import numpy as np

# Enter your own IP! After you run oak.py script, it will print the IP in the terminal
OAK_IP = "10.12.101.188"

labels =  [ "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

def get_frame(socket, size):
    bytes = socket.recv(4096)
    while True:
        read = 4096
        if size-len(bytes) < read:
            read = size-len(bytes)
        bytes += socket.recv(read)
        if size == len(bytes):
            return bytes

sock = socket.socket()
sock.connect((OAK_IP, 5000))

try:
    COLOR = (127,255,0)
    while True:
        header = str(sock.recv(32), encoding="ascii")
        chunks = re.split(' +', header)
        if chunks[0] == "IMG":
            print(f">{header}<")
            ts = float(chunks[1])
            imgSize = int(chunks[2])
            det_len = int(chunks[3])

            if 0 < det_len: # There are some detections
                det_str = str(sock.recv(det_len), encoding="ascii")
                print(f'dets >{det_str}<')

            img = get_frame(sock, imgSize)
            img_planar = np.frombuffer(img, dtype=np.uint8).reshape(3, 352, 640)
            img_interleaved = img_planar.transpose(1, 2, 0).copy()

            if 0 < det_len: # There are some detections
                dets = det_str.split("|")
                for det in dets:
                    det_section = det.split(";")
                    class_id = int(det_section[0])
                    confidence = float(det_section[1])
                    bbox = [
                        int(float(det_section[2]) * img_interleaved.shape[1]),
                        int(float(det_section[3]) * img_interleaved.shape[0]),
                        int(float(det_section[4]) * img_interleaved.shape[1]),
                        int(float(det_section[5]) * img_interleaved.shape[0])
                    ]
                    cv2.putText(img_interleaved, labels[class_id], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, COLOR)
                    cv2.putText(img_interleaved, f"{int(confidence)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, COLOR)
                    cv2.rectangle(img_interleaved, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR, 2)

            cv2.imshow("Img", img_interleaved)

        if cv2.waitKey(1) == ord('q'):
            break
except Exception as e:
    print("Error:", e)

sock.close()
