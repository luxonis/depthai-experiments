
import depthai as dai
import numpy as np
import time


N = 50
DISCARD_N = 10 # Packets to discard (due to queue sizes, system booting up, etc.)
SIZE = 1000 * 1000 * 10 # 20MB

pipeline = dai.Pipeline()

script = pipeline.create(dai.node.Script)
script.setScript(f"""
import time

# Measure downlink first
sent_ts = []
buff = Buffer({SIZE})
for i in range({N}):
    node.io['xout'].send(buff)
    sent_ts.append(time.time())
    if i == {DISCARD_N-1}:
        # node.warn('{DISCARD_N-1}th buffer sent at' + str(time.time()))
        pass
    # node.warn('Sent buffer ' + str(i))
# node.warn('{N}th buffer sent at' + str(time.time()))
total_time = sent_ts[-1] - sent_ts[{DISCARD_N-1}]
total_bits = ({N-DISCARD_N}) * {SIZE} * 8
downlink = total_bits / total_time
downlink_mbps = downlink / (1000 * 1000)
# node.warn('Downlink ' + str(downlink_mbps) + ' mbps')

# Measure uplink
receive_ts = []
for i in range({N}):
    node.io['xin'].get()
    receive_ts.append(time.time())
    if i == {DISCARD_N-1}:
        # node.warn('{DISCARD_N-1}th buffer received at' + str(time.time()))
        pass
    # node.warn('Received buffer ' + str(i))
# node.warn('{N}th buffer received at' + str(time.time()))

total_time = receive_ts[-1] - receive_ts[{DISCARD_N-1}]
total_bits = ({N-DISCARD_N}) * {SIZE} * 8
uplink = total_bits / total_time
uplink_mbps = uplink / (1000 * 1000)
# node.warn('Uplink ' + str(uplink_mbps) + ' mbps')
""")

xin = pipeline.create(dai.node.XLinkIn)
xin.setNumFrames(2)
# For setMaxDataSize, *2 will improve uplink bandwidth (by about 5%-10% in this case) as there's an additional memcopy
# from incoming buffer (from USB/ETH) to the message.
xin.setMaxDataSize(SIZE * 2)
xin.setStreamName("xin")
xin.out.link(script.inputs['xin'])

xout = pipeline.create(dai.node.XLinkOut)
xout.input.setBlocking(True)
xout.input.setQueueSize(2)
xout.setStreamName("xout")
script.outputs['xout'].link(xout.input)

with dai.Device(pipeline) as device:
    device: dai.Device
    qin = device.getInputQueue("xin", 2, blocking=True)
    qout = device.getOutputQueue("xout", 2, blocking=True)

    # Downlink
    receive_ts = []
    for i in range(N):
        qout.get()
        receive_ts.append(time.time())
        # print('CCS',device.getLeonCssCpuUsage().average, 'MSS', device.getLeonMssCpuUsage().average)
        if i == DISCARD_N-1:
            # print(f'{DISCARD_N-1}th buffer received at {time.time()}')
            pass
    # print(f'{N}th buffer received at {time.time()}')

    total_time = receive_ts[-1] - receive_ts[DISCARD_N-1]
    total_bits = (N-DISCARD_N) * SIZE * 8
    downlink = total_bits / total_time
    print('Downlink {:.1f} mbps'.format(downlink/ (1000 * 1000)))

    buffer = dai.Buffer()
    buffer.setData(np.zeros(SIZE, dtype=np.uint8))
    sent_ts = []
    for i in range(N):
        qin.send(buffer)
        sent_ts.append(time.time())
        # print('CCS',device.getLeonCssCpuUsage().average, 'MSS', device.getLeonMssCpuUsage().average)
        if i == DISCARD_N-1:
            # print(f'{DISCARD_N}th buffer sent at {time.time()}')
            pass
        # print('Sending buffer', i)
    # print(f'{N}th buffer sent at {time.time()}')
    total_time = sent_ts[-1] - sent_ts[DISCARD_N-1]
    total_bits = (N-DISCARD_N) * SIZE * 8
    uplink = total_bits / total_time
    print('Uplink {:.1f} mbps'.format(uplink/ (1000 * 1000)))

    input("Press any key to continue...")

