"""
=====================
Transformation Editor
=====================

The transformation editor can be used to manipulate transformations.
"""

from pytransform3d.editor import TransformEditor
from pytransform3d.rotations import active_matrix_from_extrinsic_euler_xyx
import pytransform3d.rotations as pr
from pytransform3d.transform_manager import TransformManager
from pytransform3d.transformations import transform_from

tm = TransformManager()

# tm.add_transform(
#     "tree",
#     "world",
#     transform_from(
#         active_matrix_from_extrinsic_euler_xyx([0, 0.5, 0]), [0, 0, 0.5]  # R=active_matrix_from_extrinsic_euler_xyx(e):
#         # e:旋转x轴、y轴和x轴的角度 R:旋转矩阵 A2B=transform_from(R,p):从旋转和平移变换矩阵计算齐次变换矩阵
#         # pr.matrix_from_euler(e,i,j,k,extrinsic)
#     ),
# )

tm.add_transform(
    "tree",
    "world",
    transform_from(
        pr.matrix_from_euler([0, 0.5, 0],0,1,0, extrinsic=True), [0, 0, 0.5]  # R=active_matrix_from_extrinsic_euler_xyx(e):
        # e:旋转x轴、y轴和x轴的角度 R:旋转矩阵 A2B=transform_from(R,p):从旋转和平移变换矩阵计算齐次变换矩阵
        # pr.matrix_from_euler(e,i,j,k,extrinsic)
    ),
)
print(f"pr.matrix_from_euler([0, 0.5, 0],0,1,0, extrinsic=True):\n{pr.matrix_from_euler([0, 0.5, 0],0,1,0, extrinsic=True)}")
# tm.add_transform(
#     "car",
#     "world",
#     transform_from(
#         active_matrix_from_extrinsic_euler_xyx([0.5, 0, 0]), [0.5, 0, 0]
#     ),
# )
tm.add_transform(
    "car",
    "world",
    transform_from(
        pr.matrix_from_euler([0.5, 0, 0],0,1,0, extrinsic=True), [0.5, 0, 0]
    ),
)
print(f"pr.matrix_from_euler([0.5, 0, 0],0,1,0, extrinsic=True):\n{pr.matrix_from_euler([0.5, 0, 0],0,1,0, extrinsic=True)}")
print(f"tm:{tm}")
# Print transformations for debugging
print("tree to world transform:")
print(tm.get_transform("tree", "world"))
print("car to world transform:")
print(tm.get_transform("car", "world"))

# Initialize TransformEditor
try:
    te = TransformEditor(tm, "world", s=0.3)
    te.show()
except Exception as e:
    print(f"Error initializing TransformEditor: {e}")


te = TransformEditor(tm, "world", s=0.3)
te.show()
print("tree to world:")
print(te.transform_manager.get_transform("tree", "world"))
