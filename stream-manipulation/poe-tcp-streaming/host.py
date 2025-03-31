import re
import socket

import cv2
import numpy as np
from utils.host_arguments import initialize_argparser

_, args = initialize_argparser()


def get_frame(socket, size):
    bytes = socket.recv(4096)
    while True:
        read = 4096
        if size - len(bytes) < read:
            read = size - len(bytes)
        bytes += socket.recv(read)
        if size == len(bytes):
            return bytes


def send_lens_pos(socket, value):
    # Leave 28 bytes for other data user might want to send to the device, eg. exposure/iso setting
    header = f"{str(value).ljust(3)},{''.ljust(28)}"  # 32 bytes in total.
    print("Setting manual focus to", value)
    socket.send(bytes(header, encoding="ascii"))


def send_autofocus(socket):
    header = f"AUT,{''.ljust(28)}"  # 32 bytes in total.
    print("Setting autofocus")
    socket.send(bytes(header, encoding="ascii"))


if args.mode == "client":
    sock = socket.socket()
    sock.connect((args.address, 9876))
    connection = sock
else:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 9876))
    sock.listen()
    print("Waiting for connection")
    connection, client = sock.accept()

lens_pos = 100

while True:
    try:
        header = str(connection.recv(32), encoding="ascii")
        chunks = re.split(" +", header)
        if chunks[0] == "ABCDE":
            ts = float(chunks[1])
            img_size = int(chunks[2])
            img = get_frame(connection, img_size)
            buf = np.frombuffer(img, dtype=np.byte)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            cv2.imshow("Color", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord(".") and lens_pos < 255:
            lens_pos += 1
            send_lens_pos(connection, lens_pos)
        elif key == ord(",") and 0 < lens_pos:
            lens_pos -= 1
            send_lens_pos(connection, lens_pos)
        elif key == ord("a"):
            send_autofocus(connection)
    except Exception as e:
        print("Error:", e)

sock.close()
