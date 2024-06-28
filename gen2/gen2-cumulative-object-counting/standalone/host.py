import socket
import re
import cv2
import numpy as np

# Enter your own IP!
OAK_IP = "192.168.112.100"

width = 640
height = 640

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
    while True:
        header = str(sock.recv(256), encoding="ascii")
        chunks = re.split(' +', header)
        if chunks[0] == "ABCDE":
            ts = float(chunks[1])
            imgSize = int(chunks[2])
            img = get_frame(sock, imgSize)
            buf = np.frombuffer(img, dtype=np.byte)
            tracklet_data = eval(header[48:256])
            counter = eval(header[32:48])
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            print(frame.shape)
            for tracklet in tracklet_data:
                scale = 640/300
                name = tracklet[0]
                x = int(tracklet[1]*scale)
                y = int(tracklet[2]*scale)
                print(name, x, y)
                cv2.putText(frame, name, (x - 10, y- 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        
        # Draw ROI line
        cv2.line(frame, (int(0.5*width), 0),
            (int(0.5*width), height), (0xFF, 0, 0), 5)
        
        # display count and status
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, f'Left: {counter[0]}; Right: {counter[1]}', (
            10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
            

            
        cv2.imshow('cumulative_object_counting', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
except Exception as e:
    print("Error host:", e)

sock.close()
