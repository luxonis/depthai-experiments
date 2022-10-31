import gdown

if __name__ == "__main__":

    # Download blob from Google Drive
    url = "https://drive.google.com/uc?id=1qZEaadgleeazz-Pe4I-yqdUqKr4qMCmv"

    gdown.download(url, output="models/")
