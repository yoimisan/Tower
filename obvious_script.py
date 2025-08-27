import bpy
import math
import sys
import os
from mathutils import Vector, Euler, Matrix
import bmesh
import yaml
import ast
import random
import numpy as np
from shapely.geometry import Polygon

COLORS = {
    "red": [1, 0, 0, 1],
    "green": [0, 1, 0, 1],
    "blue": [0, 0.25, 1, 1],
    "yellow": [1, 1, 0, 1],
    "purple": [1, 0, 1, 1],
    "cyan": [0, 1, 1, 1],
    "orange": [1, 0.5, 0, 1],
    "white": [1, 1, 1, 1],
    "gray": [0.5, 0.5, 0.5, 1],
    "black": [0, 0, 0, 1]
    }

MATERIALS = {
    'wood': {
        1: 0.05,
        2: 0.9,
        3: 2
    },
}

class Heightmap:
    def __init__(self):
        self.height = {}
        self.height_list = []
    
    def update_heightmap(self, position, size, rotation):
        """
        Update the heightmap with a block at a given position and size.
        """
        polygon = self.get_polygon(position, size, rotation)
        if position[2] != 0.75:
            pos_polygon = self.height[position[2]-size[2]/2]
            update = pos_polygon.difference(polygon)
            self.height[position[2]-size[2]/2] = update

        new_height = position[2] + size[2]/2
        if new_height not in self.height_list:
            self.height_list.append(new_height)
            self.height_list = sorted(self.height_list)
            self.height[new_height] = polygon
        else:
            current_multipolygon = self.height[new_height]
            if current_multipolygon.intersects(polygon):
                raise ValueError("polygons intersect!")
            
            new_multipoly = current_multipolygon.union(polygon)
            self.height[new_height] = new_multipoly

    def get_polygon(self, position, size, rotation):
        """
        Calculate the support area for a block on the heightmap.
        Only when the ratio is enough, the block can be placed.
        """
        l, w = size[0], size[1]
        angle = rotation[2]

        # Calculate the corners of the block in world coordinates
        corners = [
            (position[0] + l/2 * np.cos(angle) - w/2 * np.sin(angle),
             position[1] + l/2 * np.sin(angle) + w/2 * np.cos(angle)),
            (position[0] + l/2 * np.cos(angle) + w/2 * np.sin(angle),
             position[1] + l/2 * np.sin(angle) - w/2 * np.cos(angle)),
            (position[0] - l/2 * np.cos(angle) + w/2 * np.sin(angle),
             position[1] - l/2 * np.sin(angle) - w/2 * np.cos(angle)),
            (position[0] - l/2 * np.cos(angle) - w/2 * np.sin(angle),
             position[1] - l/2 * np.sin(angle) + w/2 * np.cos(angle))
        ]

        # Create a polygon from the corners
        polygon = Polygon(corners)
        return polygon

    def calculate_plane(self, degree, red_or_green, point=None):
        degree = math.radians(degree)
        if red_or_green == 'green':
            normal = (-math.sin(degree), 0, math.cos(degree))
            if not point:
                point = (-1.5, 0, 2.5)
        else:
            normal = (math.sin(degree), 0, math.cos(degree))
            if not point:
                point = (1.5, 0, 2.5)
        
        a = normal[0]
        b = 0
        c = normal[2]
        d = - a * point[0] - c * point[2]
        return a, b, c, d


    def generate_points_on_plane(self, size, degree, red_or_green, n_points=20, noise_level=2):
        a, b, c, d = self.calculate_plane(degree, red_or_green)

        x = np.random.uniform(PROJECTION_X[0], PROJECTION_X[1], n_points)
        y = np.random.uniform(PROJECTION_Y[0], PROJECTION_Y[1], n_points)
        
        z_plane = (-a*x - b*y - d) / c
        
        noise = np.random.normal(0, noise_level, n_points)
        
        X = np.column_stack([x, y, np.ones_like(x)])
        coefficients = np.linalg.lstsq(X, noise, rcond=None)[0]
        adjusted_noise = noise - X.dot(coefficients)
        
        z = z_plane + adjusted_noise

        processed_z = []
        for z_val in z:
            flag = False
            for i in range(len(self.height_list)):
                if i == 0:
                    h = self.height_list[i]
                    if z_val < h:
                        processed_z.append(self.height_list[i]+size[2]/2)
                        flag = True
                        break
                elif i<len(self.height_list):
                    h = self.height_list[i]
                    if z_val < h:
                        processed_z.append(self.height_list[i-1]+size[2]/2)
                        flag = True
                        break
            if flag == False:
                processed_z.append(self.height_list[-1]+size[2]/2)
        positions = [(float(x[i]), float(y[i]), float(processed_z[i])) for i in range(n_points)]
        return positions

    def get_valid_positions(self, size, rotation, flag, red_or_green):
        """Get all valid positions on the heightmap."""
        valid_counts = 0
        valid_positions = []
        while valid_counts < 80:
            if flag == 1:
                if red_or_green == 'green':
                    x = np.random.uniform(-1.5, 0)
                else:
                    x = np.random.uniform(0, 1.5)
                y = np.random.uniform(-0.75, 0.75)
                z = 0.75
                position = (x, y, z)
                new_polygon = self.get_polygon(position, size, rotation)
                valid_positions.append(position)
                valid_counts += 1
            else:
                positions = self.generate_points_on_plane(size, DEGREE, red_or_green)
                for position in positions:
                    new_polygon = self.get_polygon(position, size, rotation)
                    pos_multipoly = self.height[position[2]-size[2]/2]
                    if pos_multipoly.intersection(new_polygon).area >= INTERSECTION_THRESHOLD:
                    #if pos_multipoly.intersects(new_polygon):
                        valid_positions.append(position)
                        valid_counts += 1
        sorted_positions = sorted(valid_positions, key=lambda t:t[2])
        return sorted_positions

class CollisionDetector:
    def __init__(self):
        pass
    
    def get_block_vertices(self, position, size, rotation):
        """
        Get the 8 vertices of a block given its position, size, and rotation.
        """
        l, w, h = size
        half_l = l / 2
        half_w = w / 2
        half_h = h / 2
        
        # Create rotation matrix from Euler angles
        rotation_matrix = Euler(rotation, 'XYZ').to_matrix().to_4x4()
        
        # 8 vertices in local space
        local_vertices = [
            Vector(( half_l,  half_w,  half_h)),
            Vector(( half_l,  half_w, -half_h)),
            Vector(( half_l, -half_w,  half_h)),
            Vector(( half_l, -half_w, -half_h)),
            Vector((-half_l,  half_w,  half_h)),
            Vector((-half_l,  half_w, -half_h)),
            Vector((-half_l, -half_w,  half_h)),
            Vector((-half_l, -half_w, -half_h))
        ]
        
        # apply rotation and translation to get world coordinates
        world_vertices = []
        for vertex in local_vertices:
            rotated_vertex = rotation_matrix @ vertex

            world_vertex = Vector(position) + rotated_vertex
            world_vertices.append(world_vertex)
        
        return world_vertices
    
    def get_block_faces(self, vertices):
        """
        Get the faces of a block given its vertices.
        Each face is represented by a list of vertices.
        """
        faces = [
            [0, 1, 3, 2],  
            [4, 5, 7, 6],  
            [0, 4, 6, 2],  
            [1, 5, 7, 3],  
            [0, 1, 5, 4],  
            [2, 3, 7, 6]   
        ]
        
        return [[vertices[i] for i in face] for face in faces]
    
    def separating_axis_theorem(self, vertices1, vertices2):
        """
        Check for collision between two sets of vertices using the Separating Axis Theorem (SAT).
        Returns True if there is a collision, False otherwise.
        """
        # get all possible separating axes
        normals = self.get_all_separating_axes(vertices1, vertices2)
        
        # check each axis
        for normal in normals:
            min1, max1 = self.project_vertices(vertices1, normal)
            min2, max2 = self.project_vertices(vertices2, normal)
            
            if max1 <= min2 or max2 <= min1:
                return False
        # If no separating axis found, there is a collision
        return True
    
    def get_all_separating_axes(self, vertices1, vertices2):
        """
        Get all possible separating axes for two sets of vertices.
        This includes face normals and edge cross products.
        """
        faces1 = self.get_block_faces(vertices1)
        normals1 = [self.get_face_normal(face) for face in faces1]
        
        faces2 = self.get_block_faces(vertices2)
        normals2 = [self.get_face_normal(face) for face in faces2]
        
        edge_normals = []
        for i in range(len(faces1)):
            for j in range(len(faces2)):
                edges1 = self.get_face_edges(faces1[i])
                edges2 = self.get_face_edges(faces2[j])
                
                for edge1 in edges1:
                    for edge2 in edges2:
                        cross = edge1.cross(edge2)
                        if cross.length > 0.001:  # To avoid zero-length normals
                            edge_normals.append(cross.normalized())
        
        all_normals = normals1 + normals2 + edge_normals
        
        unique_normals = []
        seen = set()
        for normal in all_normals:
            # Round to avoid floating point precision issues
            key = (round(normal.x, 3), round(normal.y, 3), round(normal.z, 3))
            if key not in seen:
                seen.add(key)
                unique_normals.append(normal)
        
        return unique_normals
    
    def get_face_normal(self, face_vertices):
        """
        Calculate the normal vector of a face given its vertices.
        The face is defined by three vertices.
        """
        v0 = face_vertices[0]
        v1 = face_vertices[1]
        v2 = face_vertices[2]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        normal = edge1.cross(edge2).normalized()
        return normal
    
    def get_face_edges(self, face_vertices):
        """
        Get the edges of a face defined by its vertices.
        Each edge is represented as a vector from one vertex to the next.
        """
        edges = []
        n = len(face_vertices)
        for i in range(n):
            j = (i + 1) % n
            edge = face_vertices[j] - face_vertices[i]
            edges.append(edge)
        return edges
    
    def project_vertices(self, vertices, axis):
        """
        Project the vertices onto a given axis and return the min and max values.
        The axis should be a normalized vector.
        """
        min_val = float('inf')
        max_val = float('-inf')
        
        for vertex in vertices:
            projection = vertex.dot(axis)
            
            if projection < min_val:
                min_val = projection
            if projection > max_val:
                max_val = projection
        
        return min_val, max_val
    
    def check_block_collision(self, existing_blocks, new_position, new_size, new_rotation):
        """
        Check if a new block collides with existing blocks in the scene.
        """
        new_vertices = self.get_block_vertices(new_position, new_size, new_rotation)
        
        for block in existing_blocks:
                
            existing_vertices = self.get_block_vertices(
                block['position'], 
                block['size'], 
                block['rotation']
            )
            
            if self.separating_axis_theorem(new_vertices, existing_vertices):
                return True  
        
        return False  

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.anim.keyframe_clear_v3d(confirm=False)
    bpy.ops.object.delete(use_global=False)

    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        bpy.data.materials.remove(block)

    for block in bpy.data.images:
        bpy.data.images.remove(block)
    
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)
    
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)
    
    for block in bpy.data.actions:
        bpy.data.actions.remove(block)

    bpy.ops.ptcache.free_bake_all()
    if bpy.context.scene.rigidbody_world is not None:
        bpy.ops.rigidbody.world_remove()

    import gc
    gc.collect()

def setup_camera(cam_loc=(0, -20, 2), cam_rot=(1.5, 0, 0)):
    """
        Set up the camera.
        Args:
            cam_loc
            cam_rot
            video_len
            fps
    """
    cam_data = bpy.data.cameras.new('SceneCamera')
    cam_obj = bpy.data.objects.new('SceneCamera', cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = cam_loc
    cam_obj.rotation_euler = cam_rot
    bpy.context.scene.camera = cam_obj
    create_camera_animation(cam_obj)

def create_camera_animation(camera, target_loc=(0, 0, 2.5)):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=target_loc)
    empty = bpy.context.object
    empty.name = "CameraTarget"

    camera.parent = empty

    total_frames = VIDEO_LEN * FPS
    bpy.context.scene.frame_end = total_frames

    for frame in [1, total_frames]:
        bpy.context.scene.frame_set(frame)

        if frame == 1:
            empty.rotation_euler = (0, 0, 0)
        else:
            empty.rotation_euler = (0, 0, math.radians(360))

        empty.keyframe_insert(data_path="rotation_euler", frame=frame)

def setup_light(light_type='POINT'):
    """
    Set up point lights and the background light.
    """
    loc_list = []
    r = 8
    num_lights = 12
    angle_step = 2 * math.pi / num_lights
    for j in range(num_lights):
        angle = j * angle_step
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        loc_list.append((x, y, 2.5))
    num_lights_2 = 6
    angle_step_2 = 2 * math.pi / num_lights_2
    r2 = 5.5
    for j in range(num_lights_2):
        angle = j * angle_step_2
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        loc_list.append((x, y, 5))

    for i, loc in enumerate(loc_list):
        light_data = bpy.data.lights.new(name=f'SceneLight_{i}', type=light_type)
        light_data.energy = 300
        light_data.color = (1, 1, 1)
        light_obj = bpy.data.objects.new(name=f'SceneLight_{i}', object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc

    bpy.context.scene.world.use_nodes = True
    env = bpy.context.scene.world.node_tree.nodes['Background']
    env.inputs['Color'].default_value = (0, 0, 0, 1)

def create_material(obj, color, mat_name):
    """
    Set up block's material.
    Args:
        obj: a blender object (block)
        color: string
        mat_name: string
    """
    whole_name = mat_name + color
    mat_params = MATERIALS.get(mat_name)
    mat = bpy.data.materials.new(name=whole_name)
    mat.use_nodes = True

    mat.node_tree.nodes.clear()

    bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    for key, value in mat_params.items():
        bsdf.inputs[key].default_value = value
    bsdf.inputs[0].default_value = COLORS[color]

    output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'],output.inputs['Surface'])

    mat.node_tree.update_tag()
    bpy.context.view_layer.update()

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    #obj.visible_shadow = False
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False
    
    cycles_obj = obj.cycles
    cycles_obj.is_shadow_catcher = False
    #cycles_obj.case_shadow = False
    cycles_obj.diffuse_bounce = 0
    cycles_obj.glossy_bounce = 0
    cycles_obj.transmission_bounce = 0
    cycles_obj.transparent_bounce = 0

    obj.data.update_tag()
    bpy.context.view_layer.update()

def create_red_green_ground():
    """
    Create ground. Add material and physics. Set up render settings.
    """
    bpy.ops.mesh.primitive_circle_add(vertices=100, radius=20, fill_type='TRIFAN', location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "RedGreenGround"
    mesh = ground.data

    vcol_layer = mesh.vertex_colors.new(name="Col")
    for poly in mesh.polygons[:len(mesh.polygons)//2]:
        for i in poly.loop_indices:
            vcol_layer.data[i].color = COLORS['red']
    for poly in mesh.polygons[len(mesh.polygons)//2:]:
        for i in poly.loop_indices:
            vcol_layer.data[i].color = COLORS['green']

    mat = bpy.data.materials.new(name="RedGreenMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    vcol_node = nodes.new(type='ShaderNodeVertexColor')
    vcol_node.layer_name = "Col"

    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(vcol_node.outputs['Color'], output.inputs['Surface'])

    ground.visible_diffuse = False
    ground.visible_glossy = False
    ground.visible_transmission = False
    
    cycles_obj = ground.cycles
    cycles_obj.is_shadow_catcher = False
    cycles_obj.diffuse_bounce = 0
    cycles_obj.glossy_bounce = 0
    cycles_obj.transmission_bounce = 0
    cycles_obj.transparent_bounce = 0

    ground.data.materials.append(mat)
    bpy.ops.object.shade_smooth()

    bpy.context.view_layer.objects.active = ground
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'

def create_block_mesh(size):
    """
    Create a block mesh based on the size.
    Args:
        size: lenth, width and height
    """
    size_str = f"{size[0]:.1f}X{size[1]:.1f}X{size[2]:.1f}"
    mesh_name = f"BlockMesh_{size_str}"

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    
    scale_matrix = Matrix((
    (size[0], 0, 0, 0),
    (0, size[1], 0, 0),
    (0, 0, size[2], 0),
    (0, 0, 0, 1)
))
    bmesh.ops.transform(bm, matrix=scale_matrix, verts=bm.verts)
    
    mesh = bpy.data.meshes.new(mesh_name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh

def generate_a_block(block_data):
    """
    Generate a block based on block_data. Add material and physics.
    """
    index = block_data['index']
    color = block_data['color']
    mat_name = block_data['material']
    size = block_data['size']
    pos = block_data['position']
    rot = block_data['rotation']
    mesh = create_block_mesh(size)
    
    obj = bpy.data.objects.new(f"block_{index}", mesh)
    obj.location = Vector(pos)
    obj.rotation_euler = Euler(rot)
    create_material(obj, color, mat_name)
    bpy.context.scene.collection.objects.link(obj)

    #set_block_physics(obj)

def create_mesh(mesh_type, block_data=None):
    """
    Create object mesh. 
    Args:
        mesh_type: 'PLANE' or 'BLOCK'
        block_data: if 'BLOCK'
    """
    if mesh_type == 'PLANE':
        create_red_green_ground()
    elif mesh_type == 'BLOCK':
        generate_a_block(block_data)

def setup_render(resolution_x=800, resolution_y=800, samples=128):
    """
    Set up basic render settings.
    Args:
        index: scene index
        resolution_x
        resolution_y
        samples
    """
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.context.scene.cycles.samples = samples

    cycles = bpy.context.scene.cycles
    cycles.device = 'GPU'
    cycles.max_bounces = 0
    cycles.diffuse_bounces = 0
    cycles.glossy_bounces = 0
    cycles.transmission_bounces = 0
    cycles.transparent_max_bounces = 0

    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.use_transparent_shadows = False

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = VIDEO_LEN * FPS

    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'

    bpy.context.scene.render.fps = FPS

def get_block_position(existing_blocks, heightmap, collisiondetector, new_size, new_rot, red_or_green, flag=0):
    """
    Generate a block's position.
    Args:
        existing_blocks: dic
        heightmap
        collisiondetector
        size: size of the current block
        new_rot: rotation of the current block
        red_or_green: 'red' -> negative
        flag: if it's pedestal then flag equals to 1
    """
    valid_positions = heightmap.get_valid_positions(new_size, new_rot, flag, red_or_green)
    if not valid_positions:
        raise ValueError("No valid positions available for the block.")
    while valid_positions:
        if np.random.uniform(0.0, 1.0) < FATNESS:
            position = valid_positions[0]
        else:
            position = random.choice(valid_positions)
        valid_positions.remove(position)
        if not collisiondetector.check_block_collision(existing_blocks, position, new_size, new_rot):
            heightmap.update_heightmap(position, new_size, new_rot)
            return position
    raise ValueError("No valid position found for the block after checking all options.")

def generate_blocks_data(config, heightmap, collisiondetector, red_or_green):
    """
    Generate blocks data.
    Args:
        config: dictionary from yaml
        heightmap
        collisiondetector
        red_or_green: string 'red' or 'green'. If red, more x are negative.
    """
    blocks_data = []
    num_blocks = config['Scene']['num_blocks']#29
    ori_color_dic = config['Scene']['num_colors']#{"yellow": 13, "blue": 14, "white": 2}#7,9,1
    color_dic = {}
    for key, value in ori_color_dic.items():
        color_dic[key] = value
    ori_size_dic = config['Scene']['sizes']#{(0.5, 0.5, 1.5): 16, (1.5, 0.5, 0.5): 13}#8,9
    size_dic = {}
    for key, value in ori_size_dic.items():
        key_t = ast.literal_eval(key)
        size_dic[key_t] = value
    if ROT_DISCRETE == False:
        rot_range = config['Scene']['rot_range']#[0, 360]
        assert len(rot_range) == 2
        rot_range = [math.radians(rot_range[0]), math.radians(rot_range[1])]
    else:
        rot_range = config['Scene']['rot_range']#[0, 90, 180, 270]
        rot_range = [math.radians(rot_range[i]) for i in range(len(rot_range))]
    mat = config['Scene']['material']#'wood'
    
    ped_num = random.randint(2, 5)
    for i in range(num_blocks):
        if i < ped_num:
            if ROT_DISCRETE == False:
                new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
            else:
                new_rotation = (0, 0, random.choice(rot_range))
            new_position = get_block_position(blocks_data, heightmap, collisiondetector, (0.5, 0.5, 1.5), new_rotation, red_or_green, 1)
            block_data = {
                'index' : i,
                'color' : random.choice([key for key in color_dic.keys() if color_dic[key] > 0]),
                'material' : mat,
                'size' : (0.5, 0.5, 1.5),
                'position' : new_position,
                'rotation' : new_rotation
            }
        else:
            if i % 2 == 1 and size_dic[(0.5, 0.5, 1.5)] > 0:
                new_size = (0.5, 0.5, 1.5)
            else:
                new_size = (1.5, 0.5, 0.5)
            if ROT_DISCRETE == False:
                new_rotation = (0, 0, random.uniform(rot_range[0], rot_range[1]))
            else:
                new_rotation = (0, 0, random.choice(rot_range))
            new_position = get_block_position(blocks_data, heightmap, collisiondetector, new_size, new_rotation, red_or_green)
            block_data = {
                'index' : i,
                'color' : random.choice([key for key in color_dic.keys() if color_dic[key] > 0]),
                'material' : mat,
                'size' : new_size,
                'position' : new_position,
                'rotation' : new_rotation
            }
        color_dic[block_data['color']]-=1
        size_dic[block_data['size']]-=1
        blocks_data.append(block_data)
    return blocks_data, ped_num

def set_block_physics(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = 'ACTIVE'

def no_physics_render(index, config_num_colors):
    #num_blocks = config_num_colors['yellow'] + config_num_colors['blue'] + config_num_colors['white']
    #for i in range(num_blocks):
        #obj = bpy.data.objects[f'block_{i}']
        #obj.rigid_body.type = 'PASSIVE'
    bpy.context.scene.render.filepath = OUTPUT_PATH + f"/{index}_{config_num_colors['yellow']}_{config_num_colors['blue']}_{config_num_colors['white']}.mp4"
    bpy.ops.render.render(animation=True, write_still=True)

def physics_render(index, ped_num, config):
    """
    Bake and render.
    """
    if bpy.context.scene.rigidbody_world is None:
        raise ValueError("No rigidbody_world!")
    
    num_blocks = config['Scene']['num_blocks']
    for i in range(num_blocks):
        obj = bpy.data.objects[f'block_{i}']
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = 'ACTIVE'

    rigidbody_world = bpy.context.scene.rigidbody_world
    rigidbody_world.point_cache.frame_start = 1
    rigidbody_world.point_cache.frame_end = VIDEO_LEN * FPS

    bpy.ops.ptcache.bake_all(bake=True)
    
    positions = []
    for i in range(num_blocks):
        obj = bpy.data.objects[f'block_{i}']
        bpy.context.scene.frame_set(VIDEO_LEN * FPS)
        loc = obj.matrix_world.to_translation()
        positions.append(loc)
    tilt_color = get_final_tilt_color(positions, RED_OR_GREEN, ped_num)
    bpy.context.scene.render.filepath = OUTPUT_PATH + f"/{index}_p_{tilt_color}.mp4"
    
    bpy.context.scene.frame_set(1)
    bpy.ops.render.render(animation=True, write_still=True)

def get_final_tilt_color(block_positions, red_or_green, ped_num):
    """
    Args:
        block_positions: list, positions for all blocks after physics simulation
    """
    count = 0
    for p in block_positions:
        if red_or_green == 'green':
            if p[0] >= 0:
                count += 1
        else:
            if p[0] <= 0:
                count += 1
    if count >= (len(block_positions)-ped_num) // 2:
        return red_or_green
    else:
        print("reverse")
        if red_or_green == 'green':
            return 'red'
        else:
            return 'green'

def load_scene_config(yml_path='configs/config.yml'):
    """
    Load config and set up global variables.
    """
    with open(yml_path, 'r') as f:
        config = yaml.safe_load(f)

    global SEED, INTERSECTION_THRESHOLD, FATNESS, NUM_SCENES, RED_OR_GREEN, VIDEO_LEN, FPS
    global DEGREE, POINT
    global PROJECTION_X, PROJECTION_Y
    global ROT_DISCRETE
    global OUTPUT_PATH
    
    SEED = config['General'].get("SEED", 42) 
    INTERSECTION_THRESHOLD = config['General'].get("INTERSECTION_THRESHOLD", 0.01)
    FATNESS = config['General'].get("FATNESS", 0.5)
    
    NUM_SCENES = config['General'].get("NUM_SCENES", 1)
    RED_OR_GREEN = config['General'].get("RED_OR_GREEN")
    
    VIDEO_LEN = config['General'].get("VIDEO_LEN", 6)
    FPS = config['General'].get("FPS", 30)
    
    DEGREE = config['General'].get("DEGREE", 10)
    POINT = config['General'].get("POINT", None)
    
    PROJECTION_X = config['General'].get("PROJECTION_X", [-1.5, 2.5])
    PROJECTION_Y = config['General'].get("PROJECTION_Y", [-1.5, 1.5])#(-0.75, 0.75)

    ROT_DISCRETE = config['General'].get("ROT_DISCRETE", False)
    
    OUTPUT_PATH = config['General'].get("OUTPUT_PATH")
    
    random.seed(SEED)
    np.random.seed(SEED)
    return config

def main():
    config_path = 'D:/Desktop/University/Research/Intuitive_Physics/TowerTask/configs/config.yml'
    config = load_scene_config(config_path)
    config_num_colors = {}
    for key, value in config['Scene']['num_colors'].items():
        config_num_colors[key] = value
    for i in range(NUM_SCENES):
        clear_scene()
        
        heightmap = Heightmap()
        collisiondetector = CollisionDetector()
        blocks_data = []
        blocks_data, ped_num = generate_blocks_data(config, heightmap, collisiondetector, RED_OR_GREEN)
        
        setup_render()

        create_mesh('PLANE')

        for block_data in blocks_data:
            create_mesh('BLOCK', block_data)

        setup_camera()
        setup_light()
        
        no_physics_render(i, config_num_colors)
        physics_render(i, ped_num, config)
        print(f"Finish creating scene {i}.")

if __name__=="__main__":
    main()

