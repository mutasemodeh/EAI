import sys
import numpy as np
import time
import pybullet as p
import pybullet_data
# import pybullet_rendering

import random
import trimesh
import shutil
from stl import mesh  # Use `pip install numpy-stl` to install
from PIL import Image  # Pillow library for image handling
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from pathlib import Path
from numpy.random import default_rng
from scipy.spatial.transform import Rotation as R
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *



cube = None
WIDTH, HEIGHT = 1000, 1000  # Image dimensions
ZFAR=500 # clippling plane in meter
CAM_DIST= 0.05 # in meter
RND_TRANS=5 # +- 5mm in x and y
DRP_HEIGHT=0.02 # in meter
SCALE=0.001
SIZE_LIMIT=50 ## 50mm
def save_stl(cube, filename):

    """
    Save the transformed STL geometry to a file.

    Args:
    - cube: Transformed STL geometry object.
    - filename (str or Path): Path to save the STL file.
    """
    # Assuming cube is an instance of stl.mesh.Mesh
    cube.save(str(filename))

def load_stl(filename):
    try:
        your_mesh = mesh.Mesh.from_file(filename)
        return your_mesh
    except FileNotFoundError:
        print("Error: File not found")
        sys.exit(1)
  

def calculate_mesh_dimensions(mesh):
    """
    Calculate the dimensions (length, width, height) of a mesh.

    Parameters:
    - mesh (Trimesh object): The mesh object to calculate dimensions for.

    Returns:
    - tuple: (length, width, height)
    """
    # Calculate min and max coordinates of the mesh vertices
    min_xyz = np.min(mesh.vertices, axis=0)
    max_xyz = np.max(mesh.vertices, axis=0)
    
    # Calculate dimensions
    length = max_xyz[0] - min_xyz[0]
    width = max_xyz[1] - min_xyz[1]
    height = max_xyz[2] - min_xyz[2]

    return length, width, height

def step_to_stl(step_file, stl_file):
    """
    Convert a STEP file to STL format and ensure the STL file is correctly formatted.

    Parameters:
    - step_file (str): Path to the STEP file.
    - stl_file (str): Path to save the output STL file.
    """
    # Create a STEP reader instance
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)

    if status != 1:  # Check if the file was successfully read
        print(f"Error: Failed to read {step_file}")
        return False

    step_reader.TransferRoots()

    # Get the shape from STEP reader
    shape = step_reader.Shape()

    # Create a mesh of the shape
    mesh = BRepMesh_IncrementalMesh(shape, 0.01)

    # Export mesh to STL
    stl_writer = StlAPI_Writer()
    stl_writer.SetASCIIMode(True)  # Set to True for ASCII STL, False for binary STL
    temp_stl_file = 'temp_output.stl'
    stl_writer.Write(shape, temp_stl_file)

    # Use Trimesh to re-save the STL file to ensure it's correctly formatted
    trimesh_mesh = trimesh.load(temp_stl_file)

    # Calculate dimensions of the mesh
    dimensions = calculate_mesh_dimensions(trimesh_mesh)
    # Check if any dimension exceeds 1
    if any(dim > SIZE_LIMIT for dim in dimensions):
        print("Dimensions exceed. Skipping export.")
        return False

    # Export the mesh to the final STL file
    trimesh_mesh.export(stl_file, file_type='stl')

    return True

def convert_stl_to_jpeg(stl_variations_files_dir, jpeg_output_dir):
    # Ensure the JPEG output directory exists
    os.makedirs(jpeg_output_dir, exist_ok=True)

    # Define views and corresponding filenames
    views = {
        "xy": ([0, 0, CAM_DIST], [0, 1, 0]),
    }

    # Iterate through each subfolder in the stl_variations_files_dir
    for subdir in os.listdir(stl_variations_files_dir):
        subdir_path = os.path.join(stl_variations_files_dir, subdir)
        if os.path.isdir(subdir_path):
            # Iterate through all STL files in the subfolder
            for stl_file in os.listdir(subdir_path):
                if stl_file.endswith('.stl'):
                    stl_file_path = os.path.join(subdir_path, stl_file)

                    # Create a subfolder for each STL file's variations in the output directory
                    stl_output_dir = os.path.join(jpeg_output_dir, subdir)
                    os.makedirs(stl_output_dir, exist_ok=True)

                    # Counter for the number of images saved
                    num_images_saved = 0

                    # Render and save screenshots for each view
                    for view_name, (eye, up) in views.items():
                        jpeg_filename = os.path.join(stl_output_dir, f"{os.path.splitext(stl_file)[0]}_{view_name}.jpeg")
                        render_and_capture(stl_file_path, jpeg_filename)
                        num_images_saved += 1

                    print(f'Successfully converted {num_images_saved} images for {stl_file} into {stl_output_dir}')


def convert_step_to_stl(step_files_dir, stl_files_dir):
    """
    Converts STEP files in the given directory (including subdirectories) to STL files and saves them in the specified directory,
    maintaining the directory structure. Models exceeding the size threshold will be ignored.

    Args:
    - step_files_dir (str or Path): Path to the directory containing STEP files.
    - stl_files_dir (str or Path): Path to the directory where STL files will be saved.
    - size_threshold (float): The size threshold for models. Models exceeding this threshold will not be converted.
    """
    step_files_dir = Path(step_files_dir)
    stl_files_dir = Path(stl_files_dir)

    # Create STL_Files directory if it doesn't exist
    stl_files_dir.mkdir(parents=True, exist_ok=True)

    for step_file in step_files_dir.rglob('*.STEP'):
        print(f"Processing {step_file}...")

        # Determine the relative path from the step_files_dir to the step_file
        relative_path = step_file.relative_to(step_files_dir)

        # Determine the corresponding STL file path
        stl_file = stl_files_dir / relative_path.with_suffix('.stl')

        # Create necessary subdirectories in the target STL directory
        stl_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert STEP to STL file
        success = step_to_stl(str(step_file), str(stl_file))
        if not success:
            print(f"Conversion of {step_file} failed.")
            continue

    print("Conversion complete.")

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
    width = WIDTH
    height = HEIGHT
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


def calculate_weight(file_path, density, scale=0.001):
    """
    Calculate the weight (mass) of an STL object based on its volume and given density.
    
    Parameters:
    - file_path (str): Path to the STL file.
    - density (float): Density of the material in kg/m^3.
    - scale (float, optional): Scale of the STL object. Default is 0.001.
    
    Returns:
    - weight (float): Mass of the STL object in kilograms.
    """
    # Load the STL file
    mesh = trimesh.load(file_path)
    mesh.apply_scale(scale)  # Apply the scale if necessary
    
    # Calculate the volume of the mesh
    volume = mesh.volume
    
    # Calculate the mass (weight) based on volume and density
    weight = volume * density
    
    return weight

def simulate_physics_stl_file(stl_file, output_file, scale=0.001, simulation_steps=240, sleep_time=1./240., debug=False):
    """
    Simulate dropping an STL file in a PyBullet environment and save the final position as an STL file.
    
    Parameters:
    - stl_file (str): Path to the STL file.
    - output_file (str): Path to save the transformed STL file.
    - scale (float): Scale of the STL object.
    - simulation_steps (int): Number of simulation steps.
    - sleep_time (float): Time to sleep between simulation steps.
    - debug (bool): Whether to run in debug (GUI) mode or not.
    """
    # Connect to PyBullet simulation (GUI or DIRECT mode)
    if debug:
        physicsClient = p.connect(p.GUI)  # Use p.GUI to visualize
    else:
        physicsClient = p.connect(p.DIRECT)  # Use p.DIRECT for faster simulation without visualization

    # Set gravity
    p.setGravity(0, 0, -9.8)

    # Load the plane (base) for the objects to settle on
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")

    start_position = [0, 0, DRP_HEIGHT]  # 20mm height
    start_orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
    density_aluminum = 2700

    # Load the new STL object
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=stl_file, meshScale=[scale, scale, scale])
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=stl_file, meshScale=[scale, scale, scale])
    
    mass = calculate_weight(stl_file, density_aluminum, scale=scale)
    stl_object_id = p.createMultiBody(baseMass=mass,
                                      baseCollisionShapeIndex=collision_shape_id,
                                      baseVisualShapeIndex=visual_shape_id,
                                      basePosition=start_position,
                                      baseOrientation=start_orientation)


    # Adjust the camera to focus on the object if in debug mode
    if debug:
        camera_distance = CAM_DIST
        camera_target = [0, 0, 0]
        camera_yaw = 0  # No horizontal rotation
        camera_pitch = -90  # 90 degrees downward pitch
        camera_target = [0, 0, 0]  # Look at the origin
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)

    # Run the simulation for a while to let the object settle
    for _ in range(simulation_steps):
        p.stepSimulation()
        if debug:
            time.sleep(sleep_time)  # Slow down the simulation for visualization

    # Get the position and orientation of the object at the final state
    final_position_meters, final_orientation = p.getBasePositionAndOrientation(stl_object_id)
    mesh = trimesh.load(stl_file)

    trimesh_quaternion = (final_orientation[3],  # w
                        final_orientation[0],  # x
                        final_orientation[1],  # y
                        final_orientation[2])  # z

    final_position_mm = list(final_position_meters)
    final_position_mm[0] = 0
    final_position_mm[1] = 0
    final_position_mm[2] /= scale

    mesh.apply_transform(trimesh.transformations.quaternion_matrix(trimesh_quaternion))
    mesh.apply_transform(trimesh.transformations.translation_matrix(final_position_mm))

    mesh.export(output_file)

    # Disconnect from the simulation
    p.disconnect()

def generate_variation_stl(stl_folder, output_folder, num_simulations=5, num_variations=1):
    """
    Generate and save final STL files from simulations of STL files in a folder,
    along with additional random variations of XY translation and Z-axis rotation.
    
    Parameters:
    - stl_folder (str): Path to the folder containing STL files.
    - output_folder (str): Path to the folder to save the final and variation STL files.
    - num_simulations (int): Number of simulations to perform for each STL file.
    - num_variations (int): Number of random variations to generate for each simulation.
    """
    stl_folder = Path(stl_folder)
    output_folder = Path(output_folder)

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Iterate through STL files in the folder recursively
    for stl_file in stl_folder.rglob('*.stl'):
        print(f"Processing file: {stl_file}")

        # Determine the relative path from the stl_folder to the stl_file
        relative_path = stl_file.relative_to(stl_folder)

        # Determine the output directory maintaining the same structure
        file_output_folder = output_folder / relative_path.parent
        file_output_folder.mkdir(parents=True, exist_ok=True)

        # Perform N simulations for each STL file
        for sim_index in range(num_simulations):
            # Output file path for the simulation result
            simulation_output_file = file_output_folder / f"{stl_file.stem}_sim_{sim_index+1}.stl"
            
            # Perform simulation and save the final state directly as an STL file
            simulate_physics_stl_file(str(stl_file), str(simulation_output_file), debug=False)

            print(f"Saved transformed STL file for simulation {sim_index+1}: {simulation_output_file}")

            # Augumentation
            rng = default_rng()
            for var_index in range(num_variations):
                # Load the mesh from the simulation output
                mesh = trimesh.load_mesh(str(simulation_output_file))

                # Generate random rotation around Z axis
                theta = rng.uniform(0, 2 * np.pi)
                rotation = (0, 0, theta)  # Random rotation around Z-axis

                # Generate random translation in the XY plane
                translation = rng.uniform(-RND_TRANS, RND_TRANS, size=2)
                translation = np.append(translation, 0)  # No translation in Z
                
                # Apply transformations
                mesh.apply_transform(trimesh.transformations.translation_matrix(translation))
                mesh.apply_transform(trimesh.transformations.euler_matrix(*rotation))

                # Save transformed mesh to a new STL file for the variation
                variation_output_file = file_output_folder / f"{stl_file.stem}_sim_{sim_index+1}_var_{var_index+1}.stl"
                mesh.export(str(variation_output_file))

                print(f"Saved variation {var_index+1} for simulation {sim_index+1}: {variation_output_file}")

    print("Processing complete.")


# Main function
def main():
    step_files_dir = '/Users/modeh/EAI2/Type_Dataset/STEP'
    stl_files_dir = '/Users/modeh/EAI2/Type_Dataset/STL'
    stl_physics_files_dir = '/Users/modeh/EAI2/Type_Dataset/Physics_Augumented'
    JPEG_files_dir = '/Users/modeh/EAI2/Type_Dataset/JPEG'
    convert_step_to_stl(step_files_dir, stl_files_dir)
    generate_variation_stl(stl_files_dir, stl_physics_files_dir, num_simulations=1, num_variations=10)
    convert_stl_to_jpeg(stl_physics_files_dir, JPEG_files_dir)
    sys.exit(0)

if __name__ == "__main__":
    main()
