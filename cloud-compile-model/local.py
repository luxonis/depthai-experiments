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
    'name': 'local-mobilenet-ssd',
    'myriad_shaves': 7
}
config = StringIO("""
task_type: detection
files:
  - name: FP16/local-mobilenet-ssd.xml
    source:
      $type: http
      url: $REQUEST/local-mobilenet-ssd.xml
  - name: FP16/local-mobilenet-ssd.bin
    source:
      $type: http
      url: $REQUEST/local-mobilenet-ssd.bin
framework: dldt
""")
files = {
    'config': config,
    'xml': open('./mobilenet-ssd/local-mobilenet-ssd.xml', 'rb'),
    'bin': open('./mobilenet-ssd/local-mobilenet-ssd.bin', 'rb')
}
response = requests.request("POST", url, data=payload, files=files, params=params)
print(response.status_code)
