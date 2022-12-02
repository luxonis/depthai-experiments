import socket
import re
import cv2
import numpy as np

def get_frame(socket, size):
    bytes = socket.recv(4096)
    while True:
        read = 4096
        if size-len(bytes) < read:
            read = size-len(bytes)
        bytes += socket.recv(read)
        if size == len(bytes):
            return bytes

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000))
server.listen()
print("connect")

print("Waiting for connection")
connection, client = server.accept()
try:
    print("Connected to client IP: {}".format(client))
    while True:
        header = str(connection.recv(32), encoding="ascii")
        chunks = re.split(' +', header)
        if chunks[0] == "ABCDE":
            # print(f">{header}<")
            ts = float(chunks[1])
            imgSize = int(chunks[2])
            img = get_frame(connection, imgSize)
            buf = np.frombuffer(img, dtype=np.byte)
            # print(buf.shape, buf.size)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            cv2.imshow("color", frame)
        if cv2.waitKey(1) == ord('q'):
            break
except Exception as e:
    print("Error:", e)

server.close()
