from pathlib import Path

from openvino.inference_engine import IENetwork, IECore
from argparse import ArgumentParser
import os, re


def read_args():
    parser = ArgumentParser(description='Take IR model files (xml and bin), dynamically reshape the model input and save reshapped model.')
    parser.add_argument('-m', '--model', help='Path to an .xml file', default=os.getenv('MODEL_PATH', None), required='MODEL_PATH' not in os.environ)
    parser.add_argument('-w', '--weights', help='Path to an .bin file', default=os.getenv('WEIGHTS_PATH', None), required='WEIGHTS_PATH' not in os.environ)
    parser.add_argument('-r','--reshape', help='new input shape (format = hxw)', type=str, default=os.getenv('RESHAPE', None), required='RESHAPE' not in os.environ)
    parser.add_argument('-s','--stride', help='ratio input/output (default=%(default)s)', type=int, default=os.getenv('STRIDE', 8))
    
    return parser.parse_args()

args = read_args()
print("Loading network files:\n\t{}\n\t{}".format(args.model, args.weights))
net = IENetwork(model=args.model, weights=args.weights)
for i in net.inputs.keys():
    print(f"Input blob: {i} - shape: {net.inputs[i].shape}")
for o in net.outputs.keys():
    print(f"Output blob: {o} - shape: {net.outputs[o].shape}")

if args.reshape is not None:
    m = re.match(r"(\d+)x(\d+)", args.reshape)
    if not m:
        print("Incorrect syntax for 'reshape' argument")
    else:
        h = int(m.group(1))
        w = int(m.group(2))
        h = h - h%args.stride
        w = w - w%args.stride
        print(f"Reshapping to {h}x{w}")
        for i in net.inputs.keys():
            n,c,_,_ = net.inputs[i].shape
            net.reshape({i: (n,c,h,w)})
            print(f"Input blob: {i} - new shape: {net.inputs[i].shape}")
        for o in net.outputs.keys():
            print(f"Output blob: {o} - new shape: {net.outputs[o].shape}")
        
        # Saving reshaped model in IR files
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        new_model_path = Path(args.model).parent / Path(f"{model_name}_{h}x{w}")

        print(f"Saving reshaped model in {new_model_path}")
        net.serialize(str(new_model_path.with_suffix(".xml")), str(new_model_path.with_suffix(".bin")))
