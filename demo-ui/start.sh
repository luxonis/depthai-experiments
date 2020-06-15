#!/bin/bash

(cd api && nohup python3 main.py > app.log 2>&1 &)
sleep 1
sudo systemctl restart nginx
/usr/bin/chromium-browser --noerrdialogs --disable-infobars --kiosk http://localhost