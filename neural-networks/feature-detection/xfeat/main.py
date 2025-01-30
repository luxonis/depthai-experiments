import depthai as dai
from utils.arguments import initialize_argparser
from stereo import stereo_mode
from mono import mono_mode

_, args = initialize_argparser()


model = args.model

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

model_description = dai.NNModelDescription(model)
platform = device.getPlatform().name
model_description.platform = platform
nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

parser = nn_archive.getConfig().model.heads[0].parser
if parser == "XFeatStereoParser":
    stereo_mode(device, nn_archive, visualizer, args.fps_limit)
elif parser == "XFeatMonoParser":
    mono_mode(device, nn_archive, visualizer, args.fps_limit)
else:
    raise ValueError(f"Unknown parser: {parser}")
