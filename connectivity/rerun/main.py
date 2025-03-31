import depthai as dai
import rerun as rr
from utils.arguments import initialize_argparser
from utils.host_rerun import Rerun

_, args = initialize_argparser()


resolution = (640, 400)

device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()


def main():
    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = rgb.requestOutput(
            resolution, fps=args.fps_limit, type=dai.ImgFrame.Type.NV12
        )

        left_out = None
        right_out = None
        pcl_out = None
        if args.left or args.pointcloud:
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            left_out = left.requestOutput(resolution, fps=args.fps_limit)
        if args.right or args.pointcloud:
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            right_out = right.requestOutput(resolution, fps=args.fps_limit)

        if args.pointcloud:
            stereo = pipeline.create(dai.node.StereoDepth).build(
                left=left_out,
                right=right_out,
                presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
            )

            img_align = pipeline.create(dai.node.ImageAlign)
            rgb_out.link(img_align.inputAlignTo)
            stereo.depth.link(img_align.input)

            pcl = pipeline.create(dai.node.PointCloud)
            img_align.outputAligned.link(pcl.inputDepth)
            pcl_out = pcl.outputPointCloud

        pipeline.create(Rerun).build(
            color=rgb_out,
            left=left_out if args.left else None,
            right=right_out if args.right else None,
            pointcloud=pcl_out if args.pointcloud else None,
        )

        print("Pipeline created.")
        pipeline.run()


if __name__ == "__main__":
    rr.init("Rerun")

    if args.serve is not None:
        rr.serve(open_browser=False, web_port=args.serve, server_memory_limit="2GB")
    else:
        rr.spawn(memory_limit="2GB")
    main()
