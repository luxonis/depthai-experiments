FROM openvino/ubuntu18_dev

USER root
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python-dev python3-dev
USER openvino
WORKDIR /home/openvino
ADD reshape_openvino_model.py .

CMD source /opt/intel/openvino/bin/setupvars.sh && python3 reshape_openvino_model.py