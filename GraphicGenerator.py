import sys
import numpy as np
import time
import pybullet as p
import pybullet_data
import random
import trimesh

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



 # Global variables
cube = None
WIDTH, HEIGHT = 4000, 4000  # Image dimensions
ZFAR=500 # clippling plane 
CAM_DIST= 200
RND_TRANS=50

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
    mesh = trimesh.load(temp_stl_file)
    mesh.export(stl_file, file_type='stl')

    print(f"Successfully converted {step_file} to {stl_file}")
    return True

def convert_stl_to_jpeg(stl_variations_files_dir, jpeg_output_dir):
    # Ensure the JPEG output directory exists
    os.makedirs(jpeg_output_dir, exist_ok=True)

    # Define views and corresponding filenames
    views = {
        "xy": ([0, 0, CAM_DIST], [0, 1, 0]),
    }

    # Initialize OpenGL for off-screen rendering
    fbo, texture = init_opengl(WIDTH, HEIGHT)

    # Iterate through each subfolder in the stl_variations_files_dir
    for subdir in os.listdir(stl_variations_files_dir):
        subdir_path = os.path.join(stl_variations_files_dir, subdir)
        if os.path.isdir(subdir_path):
            # Iterate through all STL files in the subfolder
            for stl_file in os.listdir(subdir_path):
                if stl_file.endswith('.stl'):
                    stl_file_path = os.path.join(subdir_path, stl_file)
                    global cube
                    cube = load_stl(stl_file_path)

                    # Create a subfolder for each STL file's variations in the output directory
                    stl_output_dir = os.path.join(jpeg_output_dir, subdir)
                    os.makedirs(stl_output_dir, exist_ok=True)

                    # Counter for the number of images saved
                    num_images_saved = 0

                    # Render and save screenshots for each view
                    for view_name, (eye, up) in views.items():
                        jpeg_filename = os.path.join(stl_output_dir, f"{os.path.splitext(stl_file)[0]}_{view_name}.jpeg")
                        render_and_save_screenshot(jpeg_filename, eye, up)
                        num_images_saved += 1

                    print(f'Successfully converted {num_images_saved} images for {stl_file} into {stl_output_dir}')

    # Cleanup OpenGL resources
    glDeleteTextures([texture])
    glDeleteFramebuffers(1, [fbo])

def convert_step_to_stl(step_files_dir, stl_files_dir):
    """
    Converts STEP files in the given directory to STL files and saves them in the specified directory.

    Args:
    - step_files_dir (str or Path): Path to the directory containing STEP files.
    - stl_files_dir (str or Path): Path to the directory where STL files will be saved.
    """
    step_files_dir = Path(step_files_dir)
    stl_files_dir = Path(stl_files_dir)

    # Create STL_Files directory if it doesn't exist
    stl_files_dir.mkdir(parents=True, exist_ok=True)

    for step_file in step_files_dir.glob('*.STEP'):
        print(f"Processing {step_file.name}...")
        
        # Generate STL file path in the same directory
        stl_file = stl_files_dir / (step_file.stem + '.stl')

        # Convert STEP to STL file
        success = step_to_stl(str(step_file), str(stl_file))
        if not success:
            print(f"Conversion of {step_file.name} failed.")
            continue

    print("Conversion complete.")


def calculate_model_dimensions(mesh_object):
    # Calculate min and max coordinates of the mesh vertices
    min_xyz = np.min(mesh_object.vectors, axis=(0, 1))
    max_xyz = np.max(mesh_object.vectors, axis=(0, 1))
    
    # Calculate center and size
    center = (min_xyz + max_xyz) / 2.0
    size = np.max(max_xyz - min_xyz)
    print(max_xyz)
    print(min_xyz)

    return center, size, min_xyz, max_xyz

def init_opengl(width, height):
    glutInit([])
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    
    # Create dummy window (not shown)
    glutCreateWindow(b'Off-screen Rendering')
    
    # Set viewport and projection matrix
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, ZFAR)
    glMatrixMode(GL_MODELVIEW)

    # Create framebuffer object (FBO) and bind it
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    
    # Create texture to render to
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    # Check if framebuffer is complete
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("Error: Framebuffer is not complete")
        sys.exit(1)
    
    # Set background color to white
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    
    # Enable lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # Set light parameters
    light_ambient = [0.4, 0.4, 0.4, 1.0]
    light_diffuse = [1.5, 1.5, 1.5, 1.0]  # Increase diffuse intensity for higher contrast
    light_specular = [1.0, 1.0, 1.0, 1.0]  # Keep specular highlight intense
    light_position = [1.0, 1.0, 1.0, 0.0]  # Directional light

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    
    return fbo, texture

def render_and_save_screenshot(filename, eye, up):
    global cube
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Set camera position
    center = [0, 0, 0]  # Camera always looks at the origin
    gluLookAt(*eye, *center, *up)
    # Enable depth testing and configure depth function
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    
    # Disable transparency (if not needed)
    glDisable(GL_BLEND)
    # Set material properties for gray color with reflectivity
    #http://devernay.free.fr/cours/opengl/materials.html
    mat_ambient = [0.19225, 0.19225, 0.19225, 1.0]
    mat_diffuse = [0.50754, 0.50754, 0.50754, 1.0]
    mat_specular = [0.508273, 0.508273, 0.508273, 1.0]
    mat_shininess = 0.4*128  # Adjusted shininess value for aluminum
    
    glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)
    
    # Render model
    glBegin(GL_TRIANGLES)
    for normal, triangle in zip(cube.normals, cube.vectors):
        glNormal3fv(normal)
        for vertex in triangle:
            glVertex3fv(vertex)
    glEnd()
    
    # Read pixels from framebuffer
    data = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
    
    # Create image and save as JPEG
    image = Image.frombytes("RGB", (WIDTH, HEIGHT), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL stores image upside down
    image.save(filename, "JPEG")

def apply_specific_transformations(mesh_object, translation=(0, 0, 0), rotation=(0, 0, 0)):
    """
    Apply specific transformations (translation and rotation) to a mesh object.
    
    Parameters:
    - mesh_object (trimesh.base.Trimesh): Input mesh object.
    - translation (tuple): XYZ translation values.
    - rotation (tuple): Euler angles for rotation around XYZ axes.
    
    Returns:
    - trimesh.base.Trimesh: Transformed mesh object.
    """
    # Check if the mesh object is a Trimesh instance
    if not isinstance(mesh_object, trimesh.Trimesh):
        raise ValueError("Input mesh_object must be a trimesh.Trimesh instance")

    # Apply translation
    translation_matrix = trimesh.transformations.translation_matrix(translation)
    mesh_object.apply_transform(translation_matrix)

    # Apply rotation
    rotation_matrix = trimesh.transformations.euler_matrix(*rotation)
    mesh_object.apply_transform(rotation_matrix)

    return mesh_object

    """
    Generate N variations of STL files with random XY translations and Z-axis rotations for each file in input_dir.
    
    Parameters:
    - input_dir (str): Path to the input directory containing STL files.
    - output_dir (str): Path to the output directory to save transformed STL files.
    - N (int): Number of random transformations to generate per STL file.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Random number generator
    rng = default_rng()
    
    # Iterate through all STL files in the input directory and its subdirectories
    for stl_file in input_dir.glob('**/*.stl'):
        # Load the STL file
        original_mesh = mesh.Mesh.from_file(stl_file)
        
        # Create a subfolder for each STL file's variations
        relative_path = stl_file.relative_to(input_dir)
        stl_output_dir = output_dir / relative_path.parent / (stl_file.stem + '_variations')
        stl_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(N):
            # Generate random rotation around Z axis
            theta = rng.uniform(0, 2 * np.pi)
            rotation = (0, 0, theta)  # Random rotation around Z-axis
            
            # Generate random translation in the X-Y plane
            translation = rng.uniform(-RND_TRANS, RND_TRANS, size=2)
            translation = np.append(translation, 0)  # No translation in Z
            
            # Apply transformations
            transformed_mesh = apply_specific_transformations(original_mesh, translation=translation, rotation=rotation)
            
            # Save the transformed mesh to a new STL file
            output_file_path = stl_output_dir / f'transformed_{i}.stl'
            transformed_mesh.save(str(output_file_path))
            
        print(f'{N} transformed STL files have been saved to {stl_output_dir}')


def simulate_drop_and_export_stl(stl_file_path, output_stl_path):
    # Initialize PyBullet
    p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    Scale = [0.001, 0.001, 0.001] 
    # Load the STL file as a PyBullet object
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=stl_file_path, meshScale=Scale)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=stl_file_path, meshScale=Scale)
    # Create a body with aluminum properties (assuming aluminum density and friction values)
    mass = 1.0
    bodyUniqueId = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId,
                                    basePosition=[0, 0, 2], baseOrientation=[0, 0, 0, 1],
                                    baseInertialFramePosition=[0, 0, 0], baseInertialFrameOrientation=[0, 0, 0, 1],
                                    useMaximalCoordinates=True)
    
    # Set aluminum-like properties (density and friction)
    p.changeDynamics(bodyUniqueId, -1, mass=mass, lateralFriction=0.5)
    
    # Run the simulation for a few seconds to let the object settle
    for _ in range(500):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Get the final position and orientation of the object
    pos, orn = p.getBasePositionAndOrientation(bodyUniqueId)
    
    # Export the final position and orientation as an STL file
    p.disconnect()  # Disconnect from PyBullet
    
    # Load the original STL mesh using stl library
    original_mesh = mesh.Mesh.from_file(stl_file_path)
    
    # Apply final position and orientation
    transform_matrix = p.getMatrixFromQuaternion(orn)
    rotation_matrix = transform_matrix[:3, :3]
    translation_vector = pos
    
    # Apply transformation to vertices
    transformed_vertices = (rotation_matrix @ original_mesh.vectors.T).T + translation_vector
    
    # Create a new mesh object with transformed vertices
    transformed_mesh = mesh.Mesh(transformed_vertices)
    
    # Save the transformed mesh to an STL file
    output_dir = os.path.dirname(output_stl_path)
    os.makedirs(output_dir, exist_ok=True)
    transformed_mesh.save(output_stl_path)
    
    print(f"Final position and orientation exported to {output_stl_path}")

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

def place_stl_object(file_path, scale=0.001):
    """
    Place an STL object in the simulation at a random position and orientation.
    
    Parameters:
    - file_path (str): Path to the STL file.
    - scale (float): Scale of the STL object.
    
    Returns:
    - body_id (int): ID of the created body.
    """
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=file_path, meshScale=[scale, scale, scale])
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=file_path, meshScale=[scale, scale, scale])
    
    start_position = [0, 0, 0.04]  # 20mm height
    start_orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
    
    # Density of aluminum (in kg/m^3)
    density_aluminum = 2700
    # Calculate mass based on volume and density
    mass = calculate_weight(file_path, density_aluminum, scale=scale)
    body_id = p.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=collision_shape_id,
                                baseVisualShapeIndex=visual_shape_id,
                                basePosition=start_position,
                                baseOrientation=start_orientation
                                )
    return body_id

def adjust_camera(body_id):
    """
    Adjust the camera to focus on the given body.
    
    Parameters:
    - body_id (int): ID of the body to focus on.
    """
    aabb_min, aabb_max = p.getAABB(body_id)
    aabb_center = [(a + b) * 0.5 for a, b in zip(aabb_min, aabb_max)]
    aabb_extent = [b - a for a, b in zip(aabb_min, aabb_max)]
    max_extent = max(aabb_extent)

    # Set the camera distance based on the bounding box extent
    camera_distance = max_extent * 2
    camera_target = aabb_center
    camera_position = [aabb_center[0], aabb_center[1] - camera_distance, aabb_center[2] + camera_distance]

    p.resetDebugVisualizerCamera(camera_distance, 0, -45, camera_target)

def simulate_stl_file(stl_file, scale=0.001, simulation_steps=240, sleep_time=1./240., debug=False):
    """
    Simulate dropping an STL file in a PyBullet environment and output the final position and orientation.
    
    Parameters:
    - stl_file (str): Path to the STL file.
    - scale (float): Scale of the STL object.
    - simulation_steps (int): Number of simulation steps.
    - sleep_time (float): Time to sleep between simulation steps.
    - debug (bool): Whether to run in debug (GUI) mode or not.
    
    Returns:
    - final_position (list): Final position of the object.
    - final_orientation (list): Final orientation of the object.
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

    # Load the new STL object
    stl_object_id = place_stl_object(stl_file, scale=scale)

    # Adjust the camera to focus on the object if in debug mode
    if debug:
        adjust_camera(stl_object_id)

    # Run the simulation for a while to let the object settle
    for _ in range(simulation_steps):
        p.stepSimulation()
        if debug:
            time.sleep(sleep_time)  # Slow down the simulation for visualization

    # Get the final position and orientation of the object
    final_position, final_orientation = p.getBasePositionAndOrientation(stl_object_id)
    print("Final position:", final_position)
    print("Final orientation:", final_orientation)

    # Disconnect from the simulation
    p.disconnect()

    return final_position, final_orientation

def generate_physics_stl(stl_folder, output_folder, num_simulations=5, num_variations=1):
    """
    Generate and save final STL files from simulations of STL files in a folder,
    along with additional random variations of XY translation and Z-axis rotation.
    
    Parameters:
    - stl_folder (str): Path to the folder containing STL files.
    - output_folder (str): Path to the folder to save the final and variation STL files.
    - num_simulations (int): Number of simulations to perform for each STL file.
    - num_variations (int): Number of random variations to generate for each simulation.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through STL files in the folder
    for file_name in os.listdir(stl_folder):
        if file_name.endswith('.stl'):
            stl_file = os.path.join(stl_folder, file_name)
            print(f"Processing file: {stl_file}")

            # Create a subdirectory for each STL file
            file_name_no_ext = os.path.splitext(file_name)[0]
            file_output_folder = os.path.join(output_folder, file_name_no_ext)
            os.makedirs(file_output_folder, exist_ok=True)

            # Perform N simulations for each STL file
            for sim_index in range(num_simulations):
                # Perform simulation
                final_position, final_orientation = simulate_stl_file(stl_file, debug=False)

                # Load STL mesh
                mesh = trimesh.load(stl_file)

                # Apply final position and orientation
                mesh.apply_transform(trimesh.transformations.translation_matrix(final_position))
                mesh.apply_transform(trimesh.transformations.quaternion_matrix(final_orientation))

                # Save transformed mesh to a new STL file for the simulation
                simulation_output_file = os.path.join(file_output_folder, f"{file_name_no_ext}_sim_{sim_index+1}.stl")
                mesh.export(simulation_output_file)

                print(f"Saved transformed STL file for simulation {sim_index+1}: {simulation_output_file}")

                # Generate M random variations
                rng = default_rng()
                for var_index in range(num_variations):
                    # Generate random rotation around Z axis
                    theta = rng.uniform(0, 2 * np.pi)
                    rotation = (0, 0, theta)  # Random rotation around Z-axis

                    # Generate random translation in the XY plane
                    translation = rng.uniform(-RND_TRANS, RND_TRANS, size=2)
                    translation = np.append(translation, 0)  # No translation in Z

                    # Apply transformations
                    transformed_mesh = apply_specific_transformations(mesh, translation=translation, rotation=rotation)

                    # Save transformed mesh to a new STL file for the variation
                    variation_output_file = os.path.join(file_output_folder, f"{file_name_no_ext}_sim_{sim_index+1}_var_{var_index+1}.stl")
                    transformed_mesh.export(variation_output_file)

                    print(f"Saved variation {var_index+1} for simulation {sim_index+1}: {variation_output_file}")



# Main function
def main():
    step_files_dir = '/Users/modeh/EAI2/Data/STEP_Files'
    stl_files_dir = '/Users/modeh/EAI2/Data/STL_Files'
    stl_physics_files_dir = '/Users/modeh/EAI2/Data/STL_Files Physics'  
    JPEG_files_dir = '/Users/modeh/EAI2/Data/JPEG_Files'  

    # convert_step_to_stl(step_files_dir, stl_files_dir)
    generate_physics_stl(stl_files_dir, stl_physics_files_dir, num_simulations=5, num_variations=0)
    convert_stl_to_jpeg(stl_physics_files_dir, JPEG_files_dir)

    # stl_file="/Users/modeh/EAI2/STL_Files/95462A030_Medium-Strength Steel Hex Nut.stl"
    # simulate_stl_file(stl_file, scale=0.001, simulation_steps=240*4, sleep_time=0.001, debug=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
