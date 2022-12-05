import socket
import re
import cv2
import numpy as np

# Enter your own IP!
OAK_IP = "192.168.34.116"

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
        header = str(sock.recv(32), encoding="ascii")
        chunks = re.split(' +', header)
        if chunks[0] == "ABCDE":
            # print(f">{header}<")
            ts = float(chunks[1])
            imgSize = int(chunks[2])
            img = get_frame(sock, imgSize)
            buf = np.frombuffer(img, dtype=np.byte)
            # print(buf.shape, buf.size)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            cv2.imshow("color", frame)
        if cv2.waitKey(1) == ord('q'):
            break
except Exception as e:
    print("Error:", e)

sock.close()
