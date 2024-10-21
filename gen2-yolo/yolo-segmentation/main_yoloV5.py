import cv2
import depthai as dai
import numpy as np
import argparse
import time
import random


def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def generate_random_color():
  return [random.randint(0, 255) for _ in range(3)]


coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class_colors = {cls: generate_random_color() for cls in coco_classes}

pixel_conf_threshold = 0.3


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("--blob", help="Select model's blob path for inference.", required=True, type=str)
  parser.add_argument("--conf", help="Confidence threshold.", default=0.3, type=float)
  parser.add_argument("--iou", help="IoU threshold.", default=0.5, type=float)
  args = parser.parse_args()

  nn_blob_path = args.blob
  confidence_threshold = args.conf
  iou_threshold = args.iou

  # Start defining a pipeline
  pipeline = dai.Pipeline()
  pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2022_1)

  # Define a neural network that will make predictions based on the source frames
  nn_node = pipeline.create(dai.node.NeuralNetwork)
  nn_node.setNumPoolFrames(4)
  nn_node.input.setBlocking(False)
  nn_node.setNumInferenceThreads(2)
  nn_node.setBlobPath(nn_blob_path)

  # Define camera source
  rgb_cam_node = pipeline.create(dai.node.ColorCamera)
  rgb_cam_node.setBoardSocket(dai.CameraBoardSocket.CAM_A)
  rgb_cam_node.setFps(30)
  rgb_cam_node.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
  rgb_cam_node.setPreviewSize(640, 640)
  rgb_cam_node.setInterleaved(False)
  rgb_cam_node.preview.link(nn_node.input)

  # Create outputs
  xout_rgb = pipeline.create(dai.node.XLinkOut)
  xout_rgb.setStreamName("rgb")
  xout_rgb.input.setBlocking(False)
  nn_node.passthrough.link(xout_rgb.input)

  xout_nn = pipeline.create(dai.node.XLinkOut)
  xout_nn.setStreamName("nn")
  xout_nn.input.setBlocking(False)
  nn_node.out.link(xout_nn.input)


  # Pipeline defined, now the device is assigned and pipeline is started
  with dai.Device() as device:
    device.startPipeline(pipeline)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=8, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=8, blocking=False)

    start_time = time.monotonic()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:

      rgb_out = q_rgb.get()
      nn_out = q_nn.get()

      frame = rgb_out.getCvFrame()
      frame_height, frame_width, c = frame.shape

      if nn_out is not None:
        layers = nn_out.getAllLayers()

        if not layer_info_printed:
          print("+++ Output layer info +++\n")
          for layer_nr, layer in enumerate(layers):
            print(f"Layer {layer_nr}")
            print(f"Name: {layer.name}")
            print(f"Order: {layer.order}")
            print(f"dataType: {layer.dataType}")
            #dims = layer.dims[::-1] # reverse dimensions
            print(f"dims: {layer.dims}\n")
          layer_info_printed = True
          print("+++++++++++++++++++++++++\n")

        # Get output0 data to parse detected bounding boxes and get mask weights
        output0 = nn_out.getLayerFp16(layers[0].name)
        output0 = np.array(output0)
        dims0 = layers[0].dims            # (1, 25200, 117)
        output0 = output0.reshape(dims0)
        output0 = output0.squeeze(0)      # (25200, 117)
        output0 = output0.transpose()     # (117, 25200)

        # Get output1 data to parse segmentation mask prototypes
        output1 = nn_out.getLayerFp16(layers[1].name)
        output1 = np.array(output1)
        dims1 = layers[1].dims            # (1, 32, 160, 160)
        output1 = output1.reshape(dims1)
        output1 = output1.squeeze(0)      # (32, 160, 160)
        mask_protos = output1.reshape(output1.shape[0], output1.shape[1]*output1.shape[2]) # (32, 25600)

        # Get main info from output0
        num_classes = output0.shape[0] - 5 - 32            # number of classes = Total number of 2nd dimension - 5 bbox. info. - 32 mask weights
        bounding_boxes = output0[:4, :]                    # bounding boxes coordinates format: (xc, yc, w, h)
        box_confidences = output0[4, :]                    # bounding box confidence format: (conf)
        class_confidences = output0[5:(num_classes+5), :]  # class confidences format: (class_0, class_1, ...)
        mask_weights = output0[(num_classes+5):, :]        # mask weights format: (mask_weight_0, mask_weight_1, ... , mask_weight_31)

        class_scores = np.max(class_confidences, axis=0)
        class_ids = np.argmax(class_confidences, axis=0)

        # Initial filtering based on box confidences
        filtered_indices = box_confidences > 0.0
        filtered_boxes = bounding_boxes[:, filtered_indices]
        filtered_box_scores = box_confidences[filtered_indices]
        filtered_class_scores = class_scores[filtered_indices]
        filtered_class_ids = class_ids[filtered_indices]
        filtered_mask_weights = mask_weights[:, filtered_indices]

        # Format bounding box coordinates
        x_center = filtered_boxes[0, :]
        y_center = filtered_boxes[1, :]
        box_width = filtered_boxes[2, :]
        box_height = filtered_boxes[3, :]

        x1 = x_center - box_width / 2.
        y1 = y_center - box_height / 2.
        x2 = x_center + box_width / 2.
        y2 = y_center + box_height / 2.

        # Apply NMS
        bboxes = np.stack([x1, y1, x2, y2], axis=1)
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), filtered_box_scores.tolist(), confidence_threshold, iou_threshold)

        final_boxes = [[int(v) for v in bboxes[i]] for i in indices]
        final_scores = [filtered_class_scores[i] for i in indices]
        final_class_ids = [filtered_class_ids[i] for i in indices]
        filtered_mask_weights_t = filtered_mask_weights.transpose()
        filtered_mask_weights = np.asarray([filtered_mask_weights_t[i] for i in indices]) # (N, 32)

        if filtered_mask_weights.shape[0] != 0:
          final_masks = filtered_mask_weights @ mask_protos # matrix multiplication

        for i in range(len(final_boxes)):
          # Get bounding box data
          x1_i, y1_i, x2_i, y2_i = final_boxes[i]
          score = final_scores[i]
          class_id = final_class_ids[i]
          class_name = coco_classes[class_id]

          # Clamp coordinates
          x1_i = np.clip(x1_i, 0, frame_width - 1)
          y1_i = np.clip(y1_i, 0, frame_height - 1)
          x2_i = np.clip(x2_i, 0, frame_width - 1)
          y2_i = np.clip(y2_i, 0, frame_height - 1)

          # Draw bounding box
          cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color=class_colors[class_name], thickness=1)

          # Get mask data
          mask_ph, mask_pw = output1.shape[1:]
          mask = final_masks[i].reshape(mask_ph, mask_pw)
          mask = sigmoid(mask)
          mask = (mask > pixel_conf_threshold).astype('uint8') * 255
          mask_x1 = round(x1_i / frame_width * mask_pw)
          mask_y1 = round(y1_i / frame_height * mask_ph)
          mask_x2 = round(x2_i / frame_width * mask_pw)
          mask_y2 = round(y2_i / frame_height * mask_ph)
          mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
          mask_img = mask.astype(np.uint8)

          if mask_img.shape:
            mask_img = cv2.resize(mask_img, (x2_i-x1_i,y2_i-y1_i), interpolation=cv2.INTER_LINEAR)
            mask = np.array(mask_img)

            # Get polygon data
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            polygon = np.array([[x1_i + c[0][0], y1_i + c[0][1]] for c in contours[0]], np.int32)

            # Fill polygon
            overlay = np.zeros_like(frame, dtype=np.uint8)
            #overlay[y1_i:y2_i, x1_i:x2_i][mask_img == 255] = (255, 255, 0)
            cv2.fillPoly(overlay, [polygon], color=class_colors[class_name])
            cv2.addWeighted(frame, 1.0, overlay, 0.5, 0, frame)

            # Draw polygon
            cv2.polylines(frame, [polygon], isClosed=False, color=(0, 0, 0), thickness=2)
            #cv2.drawContours(frame, [polygon], -1, (0,0,0), 2)

          # Draw detection label (class + confidence)
          label = f"{class_name}: {score:.2f}"
          cv2.putText(frame, label, (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw fps on frame
        cv2.putText(frame, "NN fps: {:.1f}".format(fps), (2, frame_height - 4), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

        # Show frame
        cv2.imshow('res', frame)

      # Compute fps
      counter += 1
      current_time = time.monotonic()
      if (current_time - start_time) > 1:
        fps = counter / (current_time - start_time)
        counter = 0
        start_time = current_time

      if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
