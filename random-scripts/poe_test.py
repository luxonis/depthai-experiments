import socket
import struct
import time
import select
import depthai as dai
from typing import List

device_infos: List[dai.DeviceInfo] = dai.Device.getAllAvailableDevices()
ip = None
for device_info in device_infos:
    if device_info.protocol == dai.XLinkProtocol.X_LINK_TCP_IP:
        ip = device_info.name
        break

if ip is None:
    raise Exception("No POE device found!")

print('Connecting to ', ip, '...')

DEVICE_COMMAND = 2 # Command to send to device
BROADCAST_PORT = 11491

SEND_MSG_TIMEOUT_SEC = 50
SEND_MSG_FREQ_SEC = 0.2

TEST_SPEED_PASS = 1000
TEST_FULL_DUPLEX_PASS = 1
TEST_BOOT_MODE_PASS = 3
TEST_MXID_LEN_PASS = 32

# message type
DEVICE_INFO_MSG = struct.pack('I', DEVICE_COMMAND)
socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
socket.setblocking(False)

# sockets from which we expect to read
inputs = [socket]
# sockets to which we expect to write
outputs = [socket]

is_timeout = False

def ok(ok: bool) -> str:
    return '(OK)' if ok else '(FAIL)'

prev_time = time.time()
while not is_timeout:
    ready_to_read, ready_to_write, in_error = select.select(inputs, outputs, inputs)
    if ready_to_read:
        data = socket.recvfrom(1024)

        # parse data
        parsed_data = struct.unpack('I32siii', data[0])
        command = parsed_data[0]
        mxid = parsed_data[1].decode('ascii')
        speed = parsed_data[2]
        full_duplex = parsed_data[3]
        boot_mode = parsed_data[4]

        print(f'mxid: {mxid}', ok(len(mxid) == TEST_MXID_LEN_PASS))
        print(f'speed: {speed}', ok(speed == TEST_SPEED_PASS))
        print(f'full duplex: {full_duplex}', ok(full_duplex == TEST_FULL_DUPLEX_PASS))
        print(f'boot mode: {boot_mode}', ok(boot_mode == TEST_BOOT_MODE_PASS))
        break

    # send message if test not passed
    time.sleep(SEND_MSG_FREQ_SEC)
    socket.sendto(DEVICE_INFO_MSG, (ip, BROADCAST_PORT))

    # check for timeout
    if (time.time() - prev_time) > SEND_MSG_TIMEOUT_SEC:
        raise Exception('Device Timeout!')




