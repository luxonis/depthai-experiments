import gdown
from zipfile import ZipFile
import os

# Download samples from Google Drive
output = 'vids.zip'
gdown.download(r"https://drive.google.com/uc?id=1VuXgMRBTKSEkWEjoaI_2_TBGAtT-rjjC", "vids.zip", quiet=False)

with ZipFile(output, 'r') as zipObj:
   # Extract all the contents of zip file to ./vids/ directory
   zipObj.extractall(path = "./vids/")

os.remove("vids.zip")
