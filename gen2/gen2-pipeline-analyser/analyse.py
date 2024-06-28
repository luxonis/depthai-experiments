import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('file', metavar="FILE", type=Path, help="Path to pipeline JSON representation file to be used for analysis")
args = parser.parse_args()

if __name__ == "__main__":
    if not args.file.exists():
        raise RuntimeError("File does not exists")
    with args.file.open() as f:
        data = json.load(f)
    connected_nodes = {}
    for connection in data["connections"]:
        connected_nodes[connection["node1Id"]] = {
            **connected_nodes.get(connection["node1Id"], {}),
            connection["node2Id"]: connected_nodes.get(connection["node1Id"], {}).get(connection["node2Id"], []) + [connection]
        }

    def get_level(node_id, start_level=0):
        resolved_levels = [get_level(connected_id, start_level+1) for connected_id, connections in connected_nodes.items() if connected_id != node_id and node_id in connections]
        return max(resolved_levels) if len(resolved_levels) > 0 else start_level
    hierarchy = {}
    nodes = {}
    for node_id, node in dict(data["nodes"]).items():
        nodes[node_id] = node
        level = get_level(node_id)
        hierarchy[level] = {
            **hierarchy.get(level, {}),
            node_id: node
        }

    for level in sorted(hierarchy.keys()):
        print(f"=== LEVEL {level} ====")
        for node_id, node in hierarchy[level].items():
            print(node["name"])
            connected_to_str = " and ".join([
                f"\"{connection['node1Output']}\" to {nodes[connected_id]['name']} \"{connection['node2Input']}\""
                for connected_id in connected_nodes.get(node_id, [])
                for connection in connected_nodes.get(node_id, {}).get(connected_id, [])
            ])
            if len(connected_to_str) > 0:
                connected_to_str = "\tConnections: " + connected_to_str
            else:
                connected_to_str = "\tNo connections"
            print(connected_to_str)
            print(f"\tProperties: {nodes[node_id]['properties']}")
            print()

