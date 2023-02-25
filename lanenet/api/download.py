import gdown
from zipfile import ZipFile
import os

# Download samples from Google Drive
url = "https://drive.google.com/uc?id={}".format("1sAtURAZWk-RutjpmZrWDxIDxzf-mDxlu")
output = 'vids.zip'
gdown.download(url, output, quiet=False)


with ZipFile(output, 'r') as zipObj:
   # Extract all the contents of zip file to ./vids/ directory
   zipObj.extractall(path ="vids/")
   
os.remove("vids.zip")