import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from depthai_sdk import OakCamera

# resize input to smaller size for faster inference
NN_WIDTH, NN_HEIGHT = 512, 256

# set max clusters for color output (number of lines + a few more due to errors in fast postprocessing)
MAX_CLUSTERS = 6


# perform postprocessing - clustering of line embeddings using DBSCAN
# note that this is not whole postprocessing, just a quick implementation to show what is possible
# if you want to use whole postprocessing please refer to the LaneNet paper: https://arxiv.org/abs/1802.05591
def cluster_outputs(binary_seg_ret, instance_seg_ret):
    # create mask from binary output
    mask = binary_seg_ret.copy()
    mask = mask.astype(bool)

    # mask out embeddings
    embeddings = instance_seg_ret.copy()
    embeddings = np.transpose(embeddings, (1, 0))
    embeddings_masked = embeddings[mask]

    # sort so same classes are sorted first each time and generate inverse sort
    # works only if new lanes are added on the right side
    idx = np.lexsort(np.transpose(embeddings_masked)[::-1])
    idx_inverse = np.empty_like(idx)
    idx_inverse[idx] = np.arange(idx.size)
    embeddings_masked = embeddings_masked[idx]

    # cluster embeddings with DBSCAN
    clustering = DBSCAN(eps=0.4, min_samples=500, algorithm="kd_tree").fit(embeddings_masked)

    # unsort so pixels match their positions again
    clustering_labels = clustering.labels_[idx_inverse]

    # create an array of masked clusters
    clusters = np.zeros((NN_WIDTH * NN_HEIGHT,))
    clusters[mask] = clustering_labels + 1
    clusters = clusters.reshape((NN_HEIGHT, NN_WIDTH))

    return clusters


# create overlay from cluster_outputs
def create_overlay(cluster_outputs):
    output = np.array(cluster_outputs) * (255 / MAX_CLUSTERS)  # multiply to get classes between 0 and 255
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    output_colors[output == 0] = [0, 0, 0]
    return output_colors


# merge 2 frames together
def show_output(overlay, frame):
    return cv2.addWeighted(frame, 1, overlay, 0.8, 0)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def callback(packet):
    nn_data = packet.img_detections

    # first layer is a binary segmentation mask of lanes
    out1 = np.array(nn_data.getLayerInt32("LaneNet/bisenetv2_backend/binary_seg/ArgMax/Squeeze"))
    out1 = out1.astype(np.uint8)

    # second layer is an array of embeddings of dimension 4 for each pixel in the input
    out2 = np.array(nn_data.getLayerFp16("LaneNet/bisenetv2_backend/instance_seg/pix_embedding_conv/Conv2D"))
    out2 = out2.reshape((4, NN_WIDTH * NN_HEIGHT))

    # cluster outputs
    clusters = cluster_outputs(out1, out2)

    overlay = create_overlay(clusters)
    overlay = cv2.resize(overlay, (packet.frame.shape[1], packet.frame.shape[0]))

    # add overlay
    combined_frame = show_output(overlay, packet.frame)
    cv2.imshow("Lane prediction", combined_frame)


with OakCamera(replay='cars-california-01') as oak:
    color = oak.create_camera('color', resolution='1080p')

    nn = oak.create_nn('models/lanenet_openvino_2021.4_6shave.blob', color)
    nn.config_nn(resize_mode='stretch')

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
