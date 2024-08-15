# QR Code detection

This demo uses [qrdet:nano-288x512](https://hub.luxonis.com/ai/models/d1183a0f-e9a0-4fa2-8437-f2f5b0181739?view=page) neural network to detect QR codes.


## Demo

![demo](https://user-images.githubusercontent.com/18037362/173070218-5a069728-f365-4fa1-869f-ef871b90a7f7.gif)

## Decoding

Inside the `host_qr_scanner.py` code you have an option (`DECODE=True`) to also decode the QR code detected. Decoding is performed on the host using the pyzbar library.

## Installation

```
python3 -m pip install -r requirements.txt
```

## Usage

Run the application

```
python3 main.py
```
