import gdown

def download_decoder():
    # Download decoder from Google Drive
    url = "https://drive.google.com/uc?id=1jYNvnseTL49SNRx9PDcbkZ9DwsY8up7n"
    gdown.download(url, output="onnx_decoder/")

if __name__ == "__main__":
    download_decoder()
