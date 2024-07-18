import pybullet as p
import numpy as np
from PIL import Image
import os
import pybullet_data

CAM_DIST = 0.02  # in meters
SCALE=0.001
def render_and_capture(stl_filename, jpg_filename):
    # Connect to PyBullet
    p.connect(p.DIRECT)  # or p.GUI for a graphical version

    # Set additional search path to PyBullet's data path
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load a plane for reference
    # p.loadURDF('plane.urdf') 

    # Create a visual shape for the STL object
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_filename,
        meshScale=[SCALE, SCALE, SCALE],
        rgbaColor=[0.5, 0.5, 0.5, 1],  # Gray color
        specularColor=[1, 1, 1]  # Reflective
    )
    # Create a collision shape for the STL object
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_filename,
        meshScale=[SCALE, SCALE, SCALE]
    )

    # Create a multi-body for the STL object
    p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, 0, 0]
    )

    # Configure camera parameters (example parameters)
    width = 400
    height = 400
    fov = 60
    aspect = width / height
    near = 0.001
    far = 5.0
    camera_target = [0, 0, 0]
    camera_distance = CAM_DIST
    camera_yaw = 0.0
    camera_pitch = -90.0  # Pointing downwards

    # Compute view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target, camera_distance, camera_yaw, camera_pitch, 0, upAxisIndex=2)

    # Compute projection matrix
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Capture an image from the camera
    img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix)

    # Extract RGB image data
    rgb_img = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]

    # Convert to PIL image format
    pil_img = Image.fromarray(np.uint8(rgb_img))

    # Save the image to the specified JPEG filename
    pil_img.save(jpg_filename)

    # Disconnect from PyBullet
    p.disconnect()

# Example usage with STL file and JPEG filename
stl_file = '/Users/modeh/EAI2/91864A003_Black-Oxide Alloy Steel Socket Head Screw_sim_1.stl'
jpg_file = '/Users/modeh/EAI2/91864A003_Black-Oxide Alloy Steel Socket Head Screw_sim_1.jpg'
render_and_capture(stl_file, jpg_file)
