# Checks if gdown is installed else install
if ! type "gdown" > /dev/null; then
    pip3 install gdown
fi

# Downloads with gdown
gdown https://drive.google.com/uc?id=1RBroAFc0tmfxgvrh7iXc2e1EK8TVzXkA
