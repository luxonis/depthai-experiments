import socket
import re
import cv2
import numpy as np

# Enter your own IP!
OAK_IP = "169.254.1.222"

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

def send_lens_pos(socket, value):
    # Leave 28 bytes for other data user might want to send to the device, eg. exposure/iso setting
    header = f"{str(value).ljust(3)},{''.ljust(28)}" # 32 bytes in total.
    print(f"Setting manual focus to", value)
    socket.send(bytes(header, encoding='ascii'))

lensPos = 100

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

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('.') and lensPos < 255: # lensPos ++
            lensPos += 1
            send_lens_pos(sock, lensPos)
        elif key == ord(',') and 0 < lensPos: # lensPos --
            lensPos -= 1
            send_lens_pos(sock, lensPos)
        
except Exception as e:
    print("Error:", e)

sock.close()
