"""
Living Tales — Amber Cipher Station Diorama

Run inside Blender:
    blender --background --python blender/mystery/generate_station.py

Generates a low-poly Victorian train station diorama for the Amber Cipher mystery.
Sits on a circular base, lit with moody gas-lamp lighting.
"""
import bpy
import math
from mathutils import Vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def mat(name, color):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 1.0
    spec_key = "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    bsdf.inputs[spec_key].default_value = 0.0
    return m


def box(name, size, location, material, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    obj.rotation_euler = rotation
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    obj.data.materials.append(material)
    for p in obj.data.polygons:
        p.use_smooth = False
    return obj


def cylinder(name, radius, depth, location, material, vertices=8):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    for p in obj.data.polygons:
        p.use_smooth = False
    return obj


def sphere(name, radius, location, material, subdivisions=1):
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivisions, radius=radius, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    for p in obj.data.polygons:
        p.use_smooth = False
    return obj


# ---------------------------------------------------------------------------
# Station Elements
# ---------------------------------------------------------------------------

def generate_station():
    clean_scene()

    # Materials - Victorian moody palette
    mat_platform = mat("Platform", (0.45, 0.42, 0.38, 1))       # grey stone
    mat_brick = mat("Brick", (0.55, 0.35, 0.28, 1))             # red brick
    mat_wood = mat("Wood", (0.45, 0.35, 0.25, 1))               # dark wood
    mat_roof = mat("Roof", (0.30, 0.28, 0.25, 1))               # slate
    mat_glass = mat("Glass", (0.6, 0.65, 0.7, 1))               # foggy glass
    mat_iron = mat("Iron", (0.3, 0.3, 0.32, 1))                 # wrought iron
    mat_track = mat("Track", (0.35, 0.33, 0.30, 1))             # steel
    mat_gravel = mat("Gravel", (0.5, 0.48, 0.43, 1))            # track bed
    mat_lamp = mat("Lamp", (1.0, 0.9, 0.6, 1))                  # warm glow
    mat_bench = mat("Bench", (0.40, 0.32, 0.22, 1))             # dark bench
    mat_fog = mat("Fog", (0.75, 0.78, 0.82, 0.3))               # translucent fog
    mat_satchel = mat("Satchel", (0.50, 0.35, 0.20, 1))         # brown leather
    mat_base = mat("Base", (0.25, 0.23, 0.20, 1))               # dark base

    # --- Circular diorama base ---
    cylinder("DioramaBase", 3.0, 0.15, (0, 0, -0.075), mat_base, vertices=32)

    # --- Platform ---
    platform = box("Platform", (2.5, 1.2, 0.15), (0, 0, 0.075), mat_platform)

    # Platform edge (slightly raised)
    edge = box("PlatformEdge", (2.5, 0.05, 0.05), (0, -0.6, 0.175), mat_platform)

    # --- Tracks ---
    # Two rails
    rail_l = box("RailL", (3.0, 0.02, 0.03), (0, -1.2, 0.015), mat_track)
    rail_r = box("RailR", (3.0, 0.02, 0.03), (0, -0.9, 0.015), mat_track)
    # Gravel bed
    gravel = box("Gravel", (3.0, 0.5, 0.02), (0, -1.05, 0.01), mat_gravel)
    # Sleepers
    for i in range(-12, 13, 2):
        sleeper = box(f"Sleeper_{i}", (0.04, 0.4, 0.02), (i * 0.2, -1.05, 0.02), mat_wood)

    # --- Station building ---
    # Main wall
    wall_main = box("StationWall", (1.5, 0.12, 0.8), (0, 0.55, 0.5), mat_brick)
    # Roof
    roof = box("StationRoof", (1.7, 0.25, 0.05), (0, 0.55, 0.95), mat_roof)
    # Awning over platform
    awning = box("Awning", (2.0, 0.6, 0.03), (0, 0.2, 0.75), mat_roof)
    # Awning supports (iron pillars)
    for x in [-0.8, 0, 0.8]:
        pillar = cylinder(f"Pillar_{x}", 0.02, 0.6, (x, -0.1, 0.45), mat_iron)

    # Window
    window = box("Window", (0.25, 0.02, 0.3), (-0.4, 0.48, 0.55), mat_glass)
    # Door
    door = box("Door", (0.2, 0.02, 0.5), (0.3, 0.48, 0.35), mat_wood)

    # --- Office interior hint (visible through window) ---
    desk = box("Desk", (0.2, 0.15, 0.1), (-0.4, 0.65, 0.2), mat_wood)
    # Telegraph (tiny box on desk)
    telegraph = box("Telegraph", (0.05, 0.04, 0.04), (-0.35, 0.62, 0.28), mat_iron)

    # --- Benches ---
    bench1 = box("Bench1", (0.4, 0.1, 0.05), (-0.7, 0.1, 0.175), mat_bench)
    bench_legs1a = box("BenchLeg1a", (0.02, 0.02, 0.12), (-0.85, 0.1, 0.11), mat_iron)
    bench_legs1b = box("BenchLeg1b", (0.02, 0.02, 0.12), (-0.55, 0.1, 0.11), mat_iron)

    bench2 = box("Bench2", (0.4, 0.1, 0.05), (0.7, 0.1, 0.175), mat_bench)

    # --- Gas lamps ---
    for x in [-1.0, 0, 1.0]:
        # Lamp post
        post = cylinder(f"LampPost_{x}", 0.015, 0.6, (x, -0.3, 0.45), mat_iron)
        # Lamp head
        lamp = sphere(f"Lamp_{x}", 0.04, (x, -0.3, 0.78), mat_lamp)
        # Point light for glow
        bpy.ops.object.light_add(type='POINT', location=(x, -0.3, 0.78))
        light = bpy.context.active_object
        light.name = f"LampLight_{x}"
        light.data.energy = 5.0
        light.data.color = (1.0, 0.85, 0.5)
        light.data.shadow_soft_size = 0.5

    # --- Signal box (far end) ---
    signal_box = box("SignalBox", (0.3, 0.2, 0.4), (1.8, 0.3, 0.35), mat_brick)
    signal_roof = box("SignalRoof", (0.35, 0.25, 0.03), (1.8, 0.3, 0.58), mat_roof)

    # --- Satchel on bench ---
    satchel = box("Satchel", (0.08, 0.06, 0.05), (-0.65, 0.1, 0.22), mat_satchel)

    # --- Fog planes ---
    for y_offset in [-0.8, -1.5]:
        fog = box(f"Fog_{y_offset}", (3.0, 0.01, 0.3), (0, y_offset, 0.3), mat_fog)

    # --- Clock ---
    clock_face = cylinder("ClockFace", 0.08, 0.02, (0, 0.48, 0.82), mat_glass, vertices=12)
    clock_hand = box("ClockHand", (0.005, 0.005, 0.06), (0, 0.47, 0.82), mat_iron)
    clock_hand.rotation_euler = (0, 0, math.radians(-65))  # 11:12

    # --- Camera ---
    bpy.ops.object.camera_add(location=(2.5, -3.5, 2.5))
    cam = bpy.context.active_object
    cam.name = "StationCamera"
    cam.rotation_euler = (math.radians(55), 0, math.radians(25))
    bpy.context.scene.camera = cam

    # --- Ambient light ---
    bpy.ops.object.light_add(type='SUN', location=(2, -2, 4))
    sun = bpy.context.active_object
    sun.name = "MoonLight"
    sun.data.energy = 0.8  # dim, moody
    sun.data.color = (0.7, 0.75, 0.85)  # cold blue moonlight
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))

    # --- Render settings ---
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    world = bpy.context.scene.world
    if world and world.node_tree:
        world.node_tree.nodes["Background"].inputs[0].default_value = (0.12, 0.13, 0.18, 1)

    total_polys = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
    print(f"Station generated: {total_polys} polygons")


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "unity", "Assets", "Models", "Mysteries")
    os.makedirs(out_dir, exist_ok=True)

    generate_station()

    # Export
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(out_dir, "amber_cipher_station.glb"),
        export_format='GLB',
        use_selection=False,
    )
    print("Exported: amber_cipher_station.glb")
