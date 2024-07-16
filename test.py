import bpy
import mathutils

def render_stl_image(stl_path, view_direction, output_path):
    # Clear existing objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Import STL file
    bpy.ops.import_mesh.stl(filepath=stl_path)

    # Set camera
    camera = bpy.data.objects['Camera']
    camera.location = (view_direction[0], view_direction[1], view_direction[2])
    camera.rotation_euler = (math.radians(90), 0, math.radians(90))
    camera.data.lens = 50
    bpy.context.scene.camera = camera

    # Set render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.fps = 24
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = output_path

    # Set background color
    bpy.data.worlds["World"].color = (1, 1, 1)

    # Render image
    bpy.ops.render.render(write_still=True)

# Example usage
stl_path = "/path/to/your/file.stl"
view_direction = (5, 5, 5)  # Example viewing direction
output_path = "/path/to/output/image.png"
render_stl_image(stl_path, view_direction, output_path)
