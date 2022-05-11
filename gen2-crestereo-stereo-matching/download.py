import gdown


if __name__ == "__main__":

    # Download blobs from Google Drive
    ids = [
        "11hH9s1LfQ6CvSjqZgfNOmr5MsZW56QHi",
        "1B3mOtAZ2CyR07TAHTzkwFlOKRyxTSasa"
    ]
    urls = [
        f"https://drive.google.com/uc?id={id}" for id in ids
    ]

    for url in urls:
        gdown.download(url, output="models/")