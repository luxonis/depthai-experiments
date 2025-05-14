import argparse


def initialize_argparser():
    """Initialize the argument parser for the script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(
        help="Mode of the script.", required=True, dest="mode"
    )
    subparsers.add_parser("server", help="Run in server mode.")
    parser_client = subparsers.add_parser("client", help="Run in client mode.")
    parser_client.add_argument(
        "address", help="IP address of the host device.", type=str
    )

    args = parser.parse_args()

    return parser, args
