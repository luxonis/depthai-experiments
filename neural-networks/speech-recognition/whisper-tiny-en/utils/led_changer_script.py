import os

try:
    while True:
        color = node.inputs["color_in"].get()
        node.warn(f"{color}")
        if color is None:
            node.warn(
                "No unique LED color or multiple colors found; no LED changes made."
            )
        else:
            os_command = f"echo 1,{color[0]},{color[1]},{color[2]} > /dev/status_led"
            node.warn(f"Setting LED to {color}. Command: {os_command}")
            os.system(os_command)
except Exception as e:
    node.warn(str(e))
