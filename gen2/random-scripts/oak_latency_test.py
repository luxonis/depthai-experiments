
import depthai as dai
import threading
import time
import numpy as np

pipeline = dai.Pipeline()

N = 100

xin = pipeline.create(dai.node.XLinkIn)
xin.setMaxDataSize(10)
xin.setNumFrames(1)
xin.setStreamName("xin")

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("xout")
xout.input.setBlocking(True)
xout.input.setQueueSize(1)

xin.out.link(xout.input)

timestamps = []
def send_buff(q):
    for i in range(N):
        buffer = dai.Buffer()
        buffer.setData([1])
        q.send(buffer)
        print('Sending buffer', i)
        timestamps.append(time.time())
        time.sleep(0.2)

with dai.Device(pipeline) as device:

    qin = device.getInputQueue("xin", 1, blocking=True)
    qout = device.getOutputQueue("xout", 1, blocking=True)

    thread = threading.Thread(target=send_buff, args=(qin,))
    thread.start()

    latencies = np.array([])
    for i in range(N):
        buff: dai.Buffer = qout.get()
        latency = time.time() - timestamps[i]
        if i != 0: # Skip first buffer
            latency *= 1000 # to milliseconds
            latencies = np.append(latencies, latency)
            print('Got {}. buffer, latency {:.2f} ms'.format(i, latency))

    print()
    print('Average latency {:.2f} ms, Std: {:.1f}'.format(np.average(latencies), np.std(latencies)))

