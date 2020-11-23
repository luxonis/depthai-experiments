import requests
from io import StringIO


class Version:
    v2019_3 = '2019.R3'
    v2020_1 = '2020.1'
    v2020_2 = '2020.2'
    v2020_3 = '2020.3'
    v2020_4 = '2020.4'
    v2021_1 = '2021.1'


url = "http://69.164.214.171:8084/compile"
params = {
    'version': Version.v2020_1
}
payload = {
    'name': 'mobilenet-ssd',
    'myriad_shaves': 7
}
config = StringIO("""
description: >-
  The `mobilenet-ssd` model is a Single-Shot multibox Detection (SSD) network
  intended to perform object detection. This model is implemented using the Caffe\*
  framework. For details about this model, check out the repository <https://github.com/chuanqi305/MobileNet-SSD>.

  The model input is a blob that consists of a single image of 1x3x300x300 in
  BGR order, also like the `densenet-121` model. The BGR mean values need to be
  subtracted as follows: [127.5, 127.5, 127.5] before passing the image blob into
  the network. In addition, values must be divided by 0.007843.

  The model output is a typical vector containing the tracked object data, as
  previously described.
documentation: https://github.com/openvinotoolkit/open_model_zoo/blob/efd238d02035f8a5417b7b1e25cd4c997d44351f/models/public/mobilenet-ssd/mobilenet-ssd.md
task_type: detection
files:
  - name: FP16/mobilenet-ssd.xml
    size: 177219
    sha256: ec3a3931faf1a4a5d70aa12576cc2a5f1b5b0d0be2517cc8f9c42f616fa10b2f
    source:
      $type: google_drive
      id: 11-PX4EDxAnhymbuvnyb91ptvZAW3oPOn
  - name: FP16/mobilenet-ssd.bin
    size: 11566946
    sha256: db075a98c8d3e4636bb4206048b3d20f164a5a3730a5aa6b6b0cdbbfd2607fab
    source:
      $type: google_drive
      id: 1pdC4eNWxyfewCJ7T0i9SXLHKt39gBDZV
framework: dldt
license: https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/LICENSE
""")
files = {
    'config': config,
}
response = requests.request("POST", url, data=payload, files=files, params=params)
print(response.status_code)
