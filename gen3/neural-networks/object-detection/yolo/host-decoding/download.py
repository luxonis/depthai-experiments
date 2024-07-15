import gdown
from zipfile import ZipFile
import os

# Download samples from Google Drive
url = "https://drive.google.com/uc?id={}".format("1kNqH7U07mXWKib8jQ1ULg4Mz03I6EtQk")
output = 'host-decoding.zip'
gdown.download(url, output, quiet=False)


with ZipFile(output, 'r') as zipObj:
   # Extract all the contents of zip file to ./vids/ directory
   zipObj.extractall(path ="models/")
   
os.remove("host-decoding.zip")