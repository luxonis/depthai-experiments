import gdown
from zipfile import ZipFile
import os

# Download samples from Google Drive
url = "https://drive.google.com/uc?id={}".format("1RdsMoSBAEbG_v64fMuiywnOUlQS95CU6")
output = 'car-detection.zip'
gdown.download(url, output, quiet=False)


with ZipFile(output, 'r') as zipObj:
   # Extract all the contents of zip file to ./vids/ directory
   zipObj.extractall(path ="models/")
   
os.remove("car-detection.zip")