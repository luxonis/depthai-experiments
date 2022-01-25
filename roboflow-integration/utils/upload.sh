wget https://picsum.photos/500/500 -O image.jpg

base64 image.jpg | curl -d @- \
"https://api.roboflow.com/dataset/oak-d-dataset/upload?\
api_key=vkIkZac3CXvp0RZ31B3f&&\
name=image123.jpg&\
split=train"

cat anno.xml | curl -d @- \
"https://api.roboflow.com/dataset/oak-d-dataset/annotate/zqW2a7lCRsJrp0wgwASm?\
api_key=vkIkZac3CXvp0RZ31B3f&\
name=image123.xml"
