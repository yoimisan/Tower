import bpy
import bmesh
import math
import os
from mathutils import Vector, Euler, Matrix
from constants import COLORS, MATERIALS
import settings
import numpy as np
import tempfile

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


def setup_camera(cam_loc=(0, -8, 4), cam_rot=(math.radians(60), 0, 0)):
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
    # 生成数据集时使用“固定视角”：相机朝向和位置在整个序列中保持不变，
    # 仅通过 create_camera_animation 设置时间轴范围，不再让相机绕塔旋转。
    create_camera_animation(cam_obj)


def create_camera_animation(camera, target_loc=(0, 0, 2.5)):
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=target_loc)
    empty = bpy.context.object
    empty.name = "CameraTarget"

    total_frames = settings.VIDEO_LEN * settings.FPS
    bpy.context.scene.frame_end = total_frames

    # 固定视角：不再给相机添加环绕动画，只设置时间轴范围。
    # 如需恢复绕塔旋转，只需重新将 camera.parent = empty 并设置关键帧。


def setup_light(light_type="SUN"):
    """
    使用单一方向光源（SUN）+ 适当的环境光，保证阴影方向统一且具有一定高光和层次。
    """
    # 主光源：模拟太阳光，固定方向，产生清晰的高光和阴影
    sun_data = bpy.data.lights.new(name="SunLight", type="SUN")
    # 略微降低直射光强度，减轻阴影对比度
    sun_data.energy = 6.0
    # 增大半影角，让阴影边缘更柔和、不那么“实”
    sun_data.angle = math.radians(5.0)

    sun_obj = bpy.data.objects.new(name="SunLight", object_data=sun_data)
    bpy.context.collection.objects.link(sun_obj)
    # 把光源抬得更高，并稍微靠后一点，让阴影更规整、更接近俯视光
    sun_obj.location = (10.0, -15.0, 30.0)
    # 更接近从正上方斜射下来的光线，减小地面上的拉长阴影
    sun_obj.rotation_euler = (math.radians(70.0), 0.0, math.radians(35.0))

    # 略高一点的环境光，进一步抬高阴影区域亮度，让阴影不那么“死黑”
    bpy.context.scene.world.use_nodes = True
    world = bpy.context.scene.world
    env = world.node_tree.nodes.get("Background")
    if env is not None:
        env.inputs["Color"].default_value = (0.09, 0.09, 0.09, 1.0)
        env.inputs["Strength"].default_value = 1.5


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
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    for key, value in mat_params.items():
        bsdf.inputs[key].default_value = value

    # 如果有与材质同名的贴图文件，则使用贴图作为 Base Color；
    # 否则退回到纯颜色（COLORS[color]）作为 Base Color。
    tex_node = None
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.dirname(script_dir)
        tex_path = os.path.join(project_root, "texture", f"{mat_name}.png")

        if os.path.exists(tex_path):
            img = bpy.data.images.load(tex_path)

            # 纹理坐标 + Mapping
            tex_coord = nodes.new(type="ShaderNodeTexCoord")
            tex_coord.location = (-800, 0)

            mapping = nodes.new(type="ShaderNodeMapping")
            mapping.location = (-500, 0)

            # 对于地面（WoodGround），只使用一整张贴图，且避免世界原点落在贴图的拼接交点上
            if obj.name == "WoodGround":
                # 半径约为 20，XY ∈ [-20, 20]，用 1/40 把它线性压缩到宽度 1
                s = 1.0 / 40.0
                mapping.inputs["Scale"].default_value[0] = s
                mapping.inputs["Scale"].default_value[1] = s
                mapping.inputs["Scale"].default_value[2] = 1.0
                # 把世界原点 (0,0) 映射到贴图内部 0.25,0.25 位置（而不是四等分的交点 0.5,0.5）
                mapping.inputs["Location"].default_value[0] = 0.25
                mapping.inputs["Location"].value[1] = 0.25
                mapping.inputs["Location"].value[2] = 0.0
                # 关闭重复平铺：超出 0~1 的区域使用边缘颜色，避免出现额外交界
                try:
                    tex_node.extension = "EXTEND"
                except Exception:
                    pass
            else:
                # 其他物体仍然使用适度重复的贴图细节
                mapping.inputs["Scale"].default_value[0] = 2.0
                mapping.inputs["Scale"].default_value[1] = 2.0
                mapping.inputs["Scale"].default_value[2] = 1.0

            tex_node = nodes.new(type="ShaderNodeTexImage")
            tex_node.location = (-200, 0)
            tex_node.image = img
            # 使用 Box 投影，可以减少在立方体上的拉伸和条纹感
            try:
                tex_node.projection = "BOX"
                tex_node.projection_blend = 0.25
            except Exception:
                pass

            # 使用 Object 坐标做“体积式”投影，比 Generated 更不容易出现方向性条纹
            links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
            links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
    except Exception:
        # 贴图加载失败时，静默回退到纯颜色
        tex_node = None

    if tex_node is not None:
        # 贴图作为 Base Color
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = COLORS[color]

    # 略微降低粗糙度并增强高光，让表面更有光泽感
    try:
        if "Roughness" in bsdf.inputs:
            rough = bsdf.inputs["Roughness"].default_value
            bsdf.inputs["Roughness"].default_value = max(0.2, float(rough) * 0.8)
        if "Specular IOR Level" in bsdf.inputs:
            spec = bsdf.inputs["Specular IOR Level"].default_value
            bsdf.inputs["Specular IOR Level"].default_value = min(
                1.0, float(spec) + 0.1
            )
    except Exception:
        pass

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    mat.node_tree.update_tag()
    bpy.context.view_layer.update()

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # 确保物体参与漫反射和高光、阴影计算
    try:
        obj.visible_diffuse = True
        obj.visible_glossy = True
        obj.visible_transmission = True
        obj.visible_shadow = True
        cycles_obj = obj.cycles
        cycles_obj.is_shadow_catcher = False
    except Exception:
        pass

    obj.data.update_tag()
    bpy.context.view_layer.update()


def create_ground():
    """
    创建用于物理模拟的斜坡地面和一整块木质地板。
    仅保留地面，不再在四周生成围墙。
    """
    bpy.ops.mesh.primitive_circle_add(
        vertices=100, radius=20, fill_type="TRIFAN", location=(0, 0, 0)
    )
    ground = bpy.context.object
    ground.name = "PhysicsGround"
    mesh = ground.data

    # 根据配置中的 DEGREE 给地面加一个倾斜角（固定朝 +x 方向抬起）。
    tilt_rad = math.radians(settings.DEGREE)
    ground.rotation_euler = (0.0, -tilt_rad, 0.0)

    bpy.context.view_layer.objects.active = ground
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = "PASSIVE"

    # 让物理地面只用于物理，不参与渲染
    ground.hide_render = True
    ground.hide_viewport = True

    # === 可见的“桌面”地板，使用塑料贴图 ===
    # 复制一份网格，作为真正渲染出来的桌面（大块贴图）
    wood_mesh = mesh.copy()
    wood_mesh.name = "WoodGroundMesh"
    wood_ground = bpy.data.objects.new("WoodGround", wood_mesh)
    bpy.context.scene.collection.objects.link(wood_ground)
    wood_ground.location = ground.location
    wood_ground.rotation_euler = ground.rotation_euler
    # 使用 plastic 材质，会自动优先加载 texture/plastic.png 作为大贴图
    create_material(wood_ground, "white", "plastic")
    bpy.context.view_layer.objects.active = wood_ground
    bpy.ops.object.shade_smooth()


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
        create_ground()
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
    # 开启适度的光线反弹和高光反射，提高整体立体感与材质细节
    cycles.max_bounces = 6
    cycles.diffuse_bounces = 2
    cycles.glossy_bounces = 3
    cycles.transmission_bounces = 2
    cycles.transparent_max_bounces = 4

    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.use_transparent_shadows = True

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = settings.VIDEO_LEN * settings.FPS

    # 默认仍然配置为视频输出；是否真的生成 mp4 取决于 General.RENDER_VIDEO。
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"
    bpy.context.scene.render.ffmpeg.format = "MPEG4"

    bpy.context.scene.render.fps = settings.FPS


def set_block_physics(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = "ACTIVE"


def is_block_hitting_ground(obj, distance_epsilon: float = 0.05) -> bool:
    """
    判断一个方块是否“真正接触到地面”。

    早期版本只要从方块上方向下打射线击中 PhysicsGround 就视为命中，
    这样无论方块离地多高，都会被算作“击中地面”，导致所有场景都被判为坍塌。

    这里改为：
        1. 仍然从方块上方向下对 PhysicsGround 做 ray_cast，得到地面交点 z_hit
        2. 计算方块世界空间包围盒的最低点 z_min
        3. 当 (z_min - z_hit) <= distance_epsilon 时，认为方块已经落到地面附近
    """
    ground = bpy.data.objects.get("PhysicsGround")
    if ground is None:
        return False

    # 计算方块世界空间 AABB 的最低 z
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    z_min = min(v.z for v in bbox_world)

    # 从方块正上方向下发射一条射线，求它与 PhysicsGround 的交点高度
    loc = obj.matrix_world.to_translation()
    origin_world = Vector((loc.x, loc.y, loc.z + 10.0))
    direction_world = Vector((0.0, 0.0, -1.0))

    inv_mat = ground.matrix_world.inverted()
    origin_local = inv_mat @ origin_world
    direction_local = (inv_mat.to_3x3() @ direction_world).normalized()

    success, hit_loc_local, _, _ = ground.ray_cast(origin_local, direction_local)
    if not success:
        return False

    hit_loc_world = ground.matrix_world @ hit_loc_local

    # 当方块底部已经非常接近地面（或略有穿插）时，认为它“砸到地面”
    return (z_min - hit_loc_world.z) <= distance_epsilon


def no_physics_render(index, config_num_colors):
    # num_blocks = config_num_colors['yellow'] + config_num_colors['blue'] + config_num_colors['white']
    # for i in range(num_blocks):
    # obj = bpy.data.objects[f'block_{i}']
    # obj.rigid_body.type = 'PASSIVE'
    bpy.context.scene.render.filepath = settings.OUTPUT_PATH + f"/{index}.mp4"
    bpy.ops.render.render(animation=True, write_still=True)

def simple_render(index=0, scene_dir = "./predict_cache/"):
    # 1. 备份原有渲染设置（避免影响其他函数）
    render = bpy.context.scene.render
    prev_filepath = render.filepath
    prev_file_format = render.image_settings.file_format
    prev_ffmpeg_format = None
    if hasattr(render, "ffmpeg"):
        prev_ffmpeg_format = render.ffmpeg.format

    # 3. 创建临时文件（自动删除，无残留）
    temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(temp_fd)  # 关闭文件句柄，让Blender可以写入

    # try:
        # 4. 配置渲染参数，写入临时文件
    bpy.context.scene.frame_set(1)  # 固定渲染第1帧（初始静态状态）
    render.image_settings.file_format = "PNG"
    render.filepath = temp_path  # 指向临时文件路径

    # 5. 执行后台渲染（必须 write_still=True，才能生成有效文件）
    # 后台模式下 animation=False 表示单帧渲染
    bpy.ops.render.render(animation=False, write_still=True)

    # 6. 从临时文件读取数据，转换为numpy数组
    from PIL import Image  # 需确保安装Pillow：pip install pillow
    with Image.open(temp_path) as img:
        img_rgb = img.convert("RGB")  # 转换为RGB，去除Alpha通道
        pixels_rgb = np.array(img_rgb, dtype=np.uint8)


    # finally:
    #     # 7. 恢复原有渲染设置（确保不影响后续函数）
    render.filepath = prev_filepath
    render.image_settings.file_format = prev_file_format
    if prev_ffmpeg_format is not None and hasattr(render, "ffmpeg"):
        render.ffmpeg.format = prev_ffmpeg_format
    
    return pixels_rgb


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

    scene = bpy.context.scene

    # 每个样本（场景）使用一个独立的子文件夹：<OUTPUT_PATH>/<index>/
    # 其中包含：
    #   - meta.json
    #   - frame_0001.png, frame_0002.png, ...（如果启用 SAVE_ALL_FRAMES_IMAGES）
    #   - f_init.png（可选）
    #   - p_<state>.png / p_<state>.mp4（可选）
    scene_dir = os.path.join(settings.OUTPUT_PATH, f"{index}")
    os.makedirs(scene_dir, exist_ok=True)

    # 可选：在物理模拟前渲染第一帧静态图（初始状态）
    if settings.SAVE_FIRST_FRAME_IMAGE:
        render = scene.render
        prev_filepath = render.filepath
        prev_file_format = render.image_settings.file_format
        prev_ffmpeg_format = getattr(render, "ffmpeg", None)

        scene.frame_set(1)
        render.image_settings.file_format = "PNG"
        render.filepath = os.path.join(scene_dir, "f_init.png")
        bpy.ops.render.render(animation=False, write_still=True)

        # 恢复原设置
        render.image_settings.file_format = prev_file_format
        render.filepath = prev_filepath
        if prev_ffmpeg_format is not None:
            render.ffmpeg.format = prev_ffmpeg_format.format

    bpy.ops.ptcache.bake_all(bake=True)

    # === 记录整段物理过程中的方块状态序列 ===
    total_frames = settings.VIDEO_LEN * settings.FPS
    state_sequence = []

    # 倒塌过程统计：每一帧有多少非底座方块已经“砸到地面”
    per_frame_hit_counts = []

    for frame in range(1, total_frames + 1):
        bpy.context.scene.frame_set(frame)
        frame_states = []
        hit_count_this_frame = 0

        for i in range(num_blocks):
            obj = bpy.data.objects[f"block_{i}"]
            loc = obj.matrix_world.to_translation()
            rot = obj.matrix_world.to_euler()

            # 仅对非底座方块进行“是否砸到地面”的检测
            if i >= ped_num and is_block_hitting_ground(obj):
                hit_count_this_frame += 1

            frame_states.append(
                {
                    "index": i,
                    "location": [float(loc.x), float(loc.y), float(loc.z)],
                    "rotation_euler": [float(rot.x), float(rot.y), float(rot.z)],
                }
            )

        state_sequence.append(frame_states)
        per_frame_hit_counts.append(hit_count_this_frame)

    # 在最后一帧，根据命中地面的非底座方块数量判断“倒塌 / 未倒塌”
    final_hit_count = per_frame_hit_counts[-1] if per_frame_hit_counts else 0

    if final_hit_count == 0:
        collapse_state = "stable"  # 未倒塌
    else:
        collapse_state = "collapsed"  # 发生倒塌

    # 在控制台打印当前场景的二分类结果
    print(
        f"[Scene {index}] collapse_state = {collapse_state}, "
        f"hit_ground_blocks_final = {final_hit_count}"
    )

    # 将方块状态序列与倒塌过程信息保存为 JSON（每个场景一个文件）
    # 这里只需要 json，os 已在文件顶部全局导入，避免在函数内部重新 import os
    # 造成 Python 把 os 当作局部变量，从而在前面使用 os.path.join 时触发 UnboundLocalError。
    import json

    meta = {
        "scene_index": int(index),
        "num_blocks": int(num_blocks),
        "pedestal_blocks": int(ped_num),
        "video_len": float(settings.VIDEO_LEN),
        "fps": int(settings.FPS),
        "collapse_state": collapse_state,
        # 每一帧有多少非底座方块已经击中地面，可视作“倒塌过程序列”
        "per_frame_hit_counts": per_frame_hit_counts,
        # 方块坐标与姿态的完整时间序列
        "state_sequence": state_sequence,
    }

    meta_path = os.path.join(scene_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # 如果需要导出整段模拟过程中的所有帧图像
    if settings.SAVE_ALL_FRAMES_IMAGES:
        render = scene.render
        prev_filepath = render.filepath
        prev_file_format = render.image_settings.file_format
        prev_ffmpeg_format = getattr(render, "ffmpeg", None)

        # 以 PNG 序列形式导出动画，文件名形如：<scene_dir>/frame_0001.png
        render.image_settings.file_format = "PNG"
        render.filepath = os.path.join(scene_dir, "frame_")
        bpy.ops.render.render(animation=True, write_still=True)

        # 恢复原设置
        render.image_settings.file_format = prev_file_format
        render.filepath = prev_filepath
        if prev_ffmpeg_format is not None:
            render.ffmpeg.format = prev_ffmpeg_format.format

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
        render.filepath = os.path.join(scene_dir, f"p_{collapse_state}.png")
        bpy.ops.render.render(animation=False, write_still=True)

        # 恢复原设置
        render.image_settings.file_format = prev_file_format
        render.filepath = prev_filepath
        if prev_ffmpeg_format is not None:
            render.ffmpeg.format = prev_ffmpeg_format.format

    # 如果配置关闭视频渲染，则只预测（以及可选地保存单帧图像）
    if not settings.RENDER_VIDEO:
        return collapse_state

    # 渲染整段视频
    scene.render.filepath = os.path.join(scene_dir, f"p_{collapse_state}.mp4")
    scene.frame_set(1)
    bpy.ops.render.render(animation=True, write_still=True)
