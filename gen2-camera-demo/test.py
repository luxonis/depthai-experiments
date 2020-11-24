
import open3d as o3d
from pathlib import Path

print("Load a ply point cloud, print it, and render it")
curr_path = Path(__file__).parent.resolve()
print(curr_path)
Path("./pcl_dataset/depth").mkdir(parents=True, exist_ok=True)
Path("./pcl_dataset/rec_right").mkdir(parents=True, exist_ok=True)
Path("./pcl_dataset/ply").mkdir(parents=True, exist_ok=True)
# full_pth = str(curr_path) + '/pcl_dataset/ply/mesh_vista1.ply' 
full_pth = str(curr_path) + '/pcl_dataset/ply/rgbd_1.ply' 

mesh = o3d.io.read_triangle_mesh(full_pth)
# mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])

