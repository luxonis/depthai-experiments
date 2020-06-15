#!/bin/bash

# NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
export NVM_DIR="$HOME/.config/nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# NODE & yarn
nvm install v10.16.3
npm install -g yarn pm2

# WEB
sudo apt-get install -y nginx
(cd web && yarn && yarn build && sudo sh -c "uri='$uri' envsubst < nginx.conf.template > /etc/nginx/conf.d/demo.conf")

# API
(cd api && python3 -m pip install -r requirements.txt)

# DepthAI
git clone -b demo-ui https://github.com/luxonis/depthai.git
python3 -m pip install numpy opencv-python imutils concurrent_log_handler