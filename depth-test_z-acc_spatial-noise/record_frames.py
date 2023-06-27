from depthai_sdk import OakCamera, RecordType
import argparse

def record_frames(path = './', fps=10, color_only=False):
    with OakCamera(args=False) as oak:
        streams = []
        rgb = oak.create_camera(source="color", fps=fps)
        streams.append(rgb)

        if not color_only:
            left = oak.create_camera(source="left", fps=fps)
            right = oak.create_camera(source="right", fps=fps)

            streams.append(left)
            streams.append(right)

        recorder = oak.record(streams, path, RecordType.VIDEO)
        oak.visualize(streams, scale=0.7)
        oak.start(blocking=True)

def replay_frames(path = './', fps=10, color_only=False):
    with OakCamera(replay=path, args=False) as oak:
        streams = []
        rgb = oak.create_camera(source="color", fps=fps)
        streams.append(rgb)
        if not color_only:
            left = oak.create_camera(source="left", fps=fps)
            right = oak.create_camera(source="right", fps=fps)
            stereo = oak.create_stereo(left=left, right=right)
            streams.append(left)
            streams.append(right)
            streams.append(stereo)
        oak.visualize(streams, scale=0.7)
        oak.start(blocking=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record and Replay frames from DepthAI OAK cameras.')
    subparsers = parser.add_subparsers(dest="command")

    record_parser = subparsers.add_parser('record', help='Record frames from the OAK cameras.')
    record_parser.add_argument('--path', default='./', help='Path where recorded frames will be stored')
    record_parser.add_argument('--fps', type=int, default=10, help='Frames per second for recording')
    record_parser.add_argument('--color-only', action='store_true', help='Record only color frames')

    replay_parser = subparsers.add_parser('replay', help='Replay frames from the OAK cameras.')
    replay_parser.add_argument('--path', default='./', help='Path from which frames will be replayed')
    replay_parser.add_argument('--fps', type=int, default=10, help='Frames per second for replaying')
    replay_parser.add_argument('--color-only', action='store_true', help='Replay only color frames')

    args = parser.parse_args()

    if args.command == 'record':
        record_frames(args.path, args.fps, args.color_only)
    elif args.command == 'replay':
        replay_frames(args.path, args.fps, args.color_only)
    else:
        parser.print_help()