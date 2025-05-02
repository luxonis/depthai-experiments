def get_server_script():
    return """
    import socket
    import time
    import threading

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 9876))
    server.listen()
    node.warn("Server up")

    def send_frame_thread(conn):
        try:
            while True:
                pck = node.io["frame"].get()
                data = pck.getData()
                ts = pck.getTimestamp()
                header = f"ABCDE " + str(ts.total_seconds()).ljust(18) + str(len(data)).ljust(8)
                conn.send(bytes(header, encoding='ascii'))
                conn.send(data)
        except Exception as e:
            node.warn("Client disconnected")

    def receive_msgs_thread(conn):
        try:
            node.warn("Receiving messages")
            while True:
                data = conn.recv(32)
                txt = str(data, encoding="ascii")
                vals = txt.split(',')
                ctrl = CameraControl()
                if vals[0] == "AUT":
                    ctrl.setAutoFocusMode(CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                    node.warn("Autofocus set")
                else:
                    ctrl.setManualFocus(int(vals[0]))
                    node.warn(f"Manual focus set to {int(vals[0])}")
                node.io['control'].send(ctrl)

        except Exception as e:
            node.warn(f"Client disconnected, {e}")

    while True:
        conn, client = server.accept()
        node.warn(f"Connected to client IP: {client}")
        threading.Thread(target=send_frame_thread, args=(conn,)).start()
        threading.Thread(target=receive_msgs_thread, args=(conn,)).start()
    """


def get_client_script(address):
    return f"""
    HOST_IP = "{address}"

    import socket
    import time
    import threading

    node.warn("Connecting to {address}")
    sock = socket.socket()
    sock.connect((HOST_IP, 9876))
    node.warn("Connected")

    def receive_msgs_thread(conn):
        node.warn("Receiving messages")
        while True:
            data = conn.recv(32)
            txt = str(data, encoding="ascii")
            vals = txt.split(',')
            ctrl = CameraControl()
            if vals[0] == "AUT":
                ctrl.setAutoFocusMode(CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                node.warn("Autofocus set")
            else:
                ctrl.setManualFocus(int(vals[0]))
                node.warn("Manual focus set to " + vals[0])
            node.io['control'].send(ctrl)

    threading.Thread(target=receive_msgs_thread, args=(sock,)).start()

    while True:
        pck = node.io["frame"].get()
        data = pck.getData()
        ts = pck.getTimestamp()
        header = f"ABCDE " + str(ts.total_seconds()).ljust(18) + str(len(data)).ljust(8)
        sock.send(bytes(header, encoding='ascii'))
        sock.send(data)
    """
