import bpy
import bmesh
import math
from mathutils import Vector, Euler, Matrix
from constants import COLORS, MATERIALS
import settings


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
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
    cam_data = bpy.data.cameras.new("SceneCamera")
    cam_obj = bpy.data.objects.new("SceneCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = cam_loc
    cam_obj.rotation_euler = cam_rot
    bpy.context.scene.camera = cam_obj
    create_camera_animation(cam_obj)


def create_camera_animation(camera, target_loc=(0, 0, 2.5)):
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=target_loc)
    empty = bpy.context.object
    empty.name = "CameraTarget"

    camera.parent = empty

    total_frames = settings.VIDEO_LEN * settings.FPS
    bpy.context.scene.frame_end = total_frames

    for frame in [1, total_frames]:
        bpy.context.scene.frame_set(frame)

        if frame == 1:
            empty.rotation_euler = (0, 0, 0)
        else:
            empty.rotation_euler = (0, 0, math.radians(360))

        empty.keyframe_insert(data_path="rotation_euler", frame=frame)


def setup_light(light_type="POINT"):
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
        light_data = bpy.data.lights.new(name=f"SceneLight_{i}", type=light_type)
        light_data.energy = 300
        light_data.color = (1, 1, 1)
        light_obj = bpy.data.objects.new(name=f"SceneLight_{i}", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc

    bpy.context.scene.world.use_nodes = True
    env = bpy.context.scene.world.node_tree.nodes["Background"]
    env.inputs["Color"].default_value = (0, 0, 0, 1)


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

    bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    for key, value in mat_params.items():
        bsdf.inputs[key].default_value = value
    bsdf.inputs[0].default_value = COLORS[color]

    output = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    mat.node_tree.update_tag()
    bpy.context.view_layer.update()

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # obj.visible_shadow = False
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_transmission = False

    cycles_obj = obj.cycles
    cycles_obj.is_shadow_catcher = False
    # cycles_obj.case_shadow = False
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
    bpy.ops.mesh.primitive_circle_add(
        vertices=100, radius=20, fill_type="TRIFAN", location=(0, 0, 0)
    )
    ground = bpy.context.object
    ground.name = "RedGreenGround"
    mesh = ground.data

    vcol_layer = mesh.vertex_colors.new(name="Col")
    for poly in mesh.polygons[: len(mesh.polygons) // 2]:
        for i in poly.loop_indices:
            vcol_layer.data[i].color = COLORS["red"]
    for poly in mesh.polygons[len(mesh.polygons) // 2 :]:
        for i in poly.loop_indices:
            vcol_layer.data[i].color = COLORS["green"]

    mat = bpy.data.materials.new(name="RedGreenMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    vcol_node = nodes.new(type="ShaderNodeVertexColor")
    vcol_node.layer_name = "Col"

    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(vcol_node.outputs["Color"], output.inputs["Surface"])

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
    ground.rigid_body.type = "PASSIVE"


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

    scale_matrix = Matrix(
        ((size[0], 0, 0, 0), (0, size[1], 0, 0), (0, 0, size[2], 0), (0, 0, 0, 1))
    )
    bmesh.ops.transform(bm, matrix=scale_matrix, verts=bm.verts)

    mesh = bpy.data.meshes.new(mesh_name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def generate_a_block(block_data):
    """
    Generate a block based on block_data. Add material and physics.
    """
    index = block_data["index"]
    color = block_data["color"]
    mat_name = block_data["material"]
    size = block_data["size"]
    pos = block_data["position"]
    rot = block_data["rotation"]
    mesh = create_block_mesh(size)

    obj = bpy.data.objects.new(f"block_{index}", mesh)
    obj.location = Vector(pos)
    obj.rotation_euler = Euler(rot)
    create_material(obj, color, mat_name)
    bpy.context.scene.collection.objects.link(obj)

    # set_block_physics(obj)


def create_mesh(mesh_type, block_data=None):
    """
    Create object mesh.
    Args:
        mesh_type: 'PLANE' or 'BLOCK'
        block_data: if 'BLOCK'
    """
    if mesh_type == "PLANE":
        create_red_green_ground()
    elif mesh_type == "BLOCK":
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
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.context.scene.cycles.samples = samples

    cycles = bpy.context.scene.cycles
    cycles.device = "GPU"
    cycles.max_bounces = 0
    cycles.diffuse_bounces = 0
    cycles.glossy_bounces = 0
    cycles.transmission_bounces = 0
    cycles.transparent_max_bounces = 0

    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.use_transparent_shadows = False

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = settings.VIDEO_LEN * settings.FPS

    bpy.context.scene.render.image_settings.file_format = "FFMPEG"
    bpy.context.scene.render.ffmpeg.format = "MPEG4"

    bpy.context.scene.render.fps = settings.FPS


def set_block_physics(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = "ACTIVE"


def get_ground_color_for_location(loc):
    """
    根据方块在世界坐标中的位置，射线检测到地面多边形，
    再根据多边形索引判断其位于红区还是绿区。
    返回 'red' / 'green' 或 None（如果没击中地面）。
    """
    ground = bpy.data.objects.get("RedGreenGround")
    if ground is None:
        return None

    # 使用 ground 自身的 ray_cast，避免被其他物体（塔块）挡住
    # 需要把射线转换到 ground 的局部坐标系
    origin_world = Vector((loc.x, loc.y, loc.z + 10.0))
    direction_world = Vector((0.0, 0.0, -1.0))

    inv_mat = ground.matrix_world.inverted()
    origin_local = inv_mat @ origin_world
    direction_local = (inv_mat.to_3x3() @ direction_world).normalized()

    success, hit_loc, hit_normal, face_index = ground.ray_cast(
        origin_local, direction_local
    )

    if not success:
        return None

    mesh = ground.data
    num_polys = len(mesh.polygons)
    if num_polys == 0:
        return None

    # 直接读取该多边形上的顶点颜色，以真实地面着色为准
    vcols = getattr(mesh, "vertex_colors", None)
    if not vcols or "Col" not in vcols:
        return None

    vcol_layer = vcols["Col"]
    poly = mesh.polygons[face_index]
    if not poly.loop_indices:
        return None

    # 该多边形所有 loop 的颜色应该一致，这里取第一个即可
    loop_index = poly.loop_indices[0]
    col = vcol_layer.data[loop_index].color  # (r, g, b, a)
    r, g = col[0], col[1]

    if r > g:
        return "red"
    elif g > r:
        return "green"
    else:
        return None


def no_physics_render(index, config_num_colors):
    # num_blocks = config_num_colors['yellow'] + config_num_colors['blue'] + config_num_colors['white']
    # for i in range(num_blocks):
    # obj = bpy.data.objects[f'block_{i}']
    # obj.rigid_body.type = 'PASSIVE'
    bpy.context.scene.render.filepath = settings.OUTPUT_PATH + f"/{index}.mp4"
    bpy.ops.render.render(animation=True, write_still=True)


def physics_render(index, ped_num, config):
    """
    Bake and render.
    """
    if bpy.context.scene.rigidbody_world is None:
        raise ValueError("No rigidbody_world!")

    num_blocks = config["Scene"]["num_blocks"]
    for i in range(num_blocks):
        obj = bpy.data.objects[f"block_{i}"]
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = "ACTIVE"

    rigidbody_world = bpy.context.scene.rigidbody_world
    rigidbody_world.point_cache.frame_start = 1
    rigidbody_world.point_cache.frame_end = settings.VIDEO_LEN * settings.FPS

    bpy.ops.ptcache.bake_all(bake=True)

    # 在最后一帧获取每个方块的位置，并根据其正下方地面的颜色
    # 判断属于 red 区还是 green 区（忽略前 ped_num 个底座方块）。
    colors = []
    bpy.context.scene.frame_set(settings.VIDEO_LEN * settings.FPS)

    for i in range(num_blocks):
        obj = bpy.data.objects[f"block_{i}"]
        loc = obj.matrix_world.to_translation()

        # 跳过底座方块
        if i < ped_num:
            continue

        c = get_ground_color_for_location(loc)
        if c is not None:
            colors.append(c)

    if not colors:
        # 一个方块都没打到地面：使用特殊标签 'none'
        tilt_color = "none"
    else:
        num_green = sum(1 for c in colors if c == "green")
        num_red = len(colors) - num_green

        if num_green > 0 and num_red > 0:
            # red 和 green 同时存在：使用特殊标签 'both'
            tilt_color = "both"
        elif num_green > 0:
            tilt_color = "green"
        elif num_red > 0:
            tilt_color = "red"
        else:
            # 理论上不会到这里，兜底成 'none'
            tilt_color = "none"

    # 在控制台打印当前场景的预测结果
    print(f"[Scene {index}] predicted tilt_color = {tilt_color}")

    scene = bpy.context.scene

    # 如果需要保存最后一帧图像
    if settings.SAVE_LAST_FRAME_IMAGE:
        last_frame = settings.VIDEO_LEN * settings.FPS

        # 备份当前渲染设置
        render = scene.render
        prev_filepath = render.filepath
        prev_file_format = render.image_settings.file_format
        prev_ffmpeg_format = getattr(render, "ffmpeg", None)

        scene.frame_set(last_frame)
        render.image_settings.file_format = "PNG"
        render.filepath = settings.OUTPUT_PATH + f"/{index}_p_{tilt_color}.png"
        bpy.ops.render.render(animation=False, write_still=True)

        # 恢复原设置
        render.image_settings.file_format = prev_file_format
        render.filepath = prev_filepath
        if prev_ffmpeg_format is not None:
            render.ffmpeg.format = prev_ffmpeg_format.format

    # 如果配置关闭视频渲染，则只预测（以及可选地保存单帧图像）
    if not settings.RENDER_VIDEO:
        return tilt_color

    # 渲染整段视频
    scene.render.filepath = settings.OUTPUT_PATH + f"/{index}_p_{tilt_color}.mp4"
    scene.frame_set(1)
    bpy.ops.render.render(animation=True, write_still=True)
