import os
from zipfile import ZipFile

import gdown


def download_vids():
    # Download samples from Google Drive
    output = "vids.zip"
    gdown.download(
        r"https://drive.google.com/uc?id=1sAtURAZWk-RutjpmZrWDxIDxzf-mDxlu",
        "vids.zip",
        quiet=False,
    )

    with ZipFile(output, "r") as zipObj:
        # Extract all the contents of zip file to ./vids/ directory
        zipObj.extractall(path="./vids/")

    os.remove("vids.zip")


if __name__ == "__main__":
    download_vids()
