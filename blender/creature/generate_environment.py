"""
Living Tales — Creature Environment Generator

Run inside Blender:
    blender --background --python blender/creature/generate_environment.py

Generates low-poly environment pieces for the creature game:
- Den room (interior with window, door, blanket nest)
- Kitchen corner (bowls, food bag)
- Garden (fence, grass, flowers, gate)
- Porch (swing, step)

Each piece exports as a separate .glb for modular Unity assembly.
"""
import bpy
import bmesh
import math
from mathutils import Vector


# ---------------------------------------------------------------------------
# Cleanup & materials
# ---------------------------------------------------------------------------

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def mat(name, color):
    """Create flat material."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 1.0
    spec_key = "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    bsdf.inputs[spec_key].default_value = 0.0
    return m


def box(name, size, location, material):
    """Create a low-poly box."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(scale=True)
    obj.data.materials.append(material)
    for p in obj.data.polygons:
        p.use_smooth = False
    return obj


def plane(name, size, location, material):
    """Create a flat plane."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def cylinder(name, radius, depth, location, material, vertices=8):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    for p in obj.data.polygons:
        p.use_smooth = False
    return obj


# ---------------------------------------------------------------------------
# Den Room
# ---------------------------------------------------------------------------

def generate_den_room():
    """Generate the creature's cozy den room."""
    # Materials
    mat_floor = mat("Floor", (0.65, 0.55, 0.42, 1))       # warm wood
    mat_wall = mat("Wall", (0.88, 0.85, 0.78, 1))          # cream plaster
    mat_blanket = mat("Blanket", (0.55, 0.35, 0.30, 1))    # rust red
    mat_blanket2 = mat("Blanket2", (0.70, 0.60, 0.45, 1))  # cream wool
    mat_window = mat("Window", (0.75, 0.85, 0.95, 1))      # sky blue
    mat_frame = mat("Frame", (0.45, 0.35, 0.28, 1))        # dark wood

    # Floor
    floor = plane("DenFloor", 4.0, (0, 0, 0), mat_floor)

    # Walls (3 sides, open front)
    back_wall = box("BackWall", (2.0, 0.05, 1.0), (0, 2.0, 1.0), mat_wall)
    left_wall = box("LeftWall", (0.05, 2.0, 1.0), (-2.0, 0, 1.0), mat_wall)
    right_wall = box("RightWall", (0.05, 2.0, 1.0), (2.0, 0, 1.0), mat_wall)

    # Window in back wall
    window = box("Window", (0.5, 0.02, 0.4), (0, 1.98, 1.3), mat_window)
    window_frame = box("WindowFrame", (0.55, 0.03, 0.45), (0, 1.97, 1.3), mat_frame)

    # Door frame in right wall
    door_frame = box("DoorFrame", (0.03, 0.4, 0.8), (1.98, -0.8, 0.8), mat_frame)

    # Blanket nest in corner
    blanket1 = box("Blanket1", (0.5, 0.4, 0.05), (-1.3, 1.3, 0.05), mat_blanket)
    blanket1.rotation_euler = (0, 0, 0.1)
    blanket2 = box("Blanket2", (0.4, 0.35, 0.04), (-1.2, 1.4, 0.08), mat_blanket2)
    blanket2.rotation_euler = (0.05, 0, -0.15)

    # Rug in center
    mat_rug = mat("Rug", (0.5, 0.42, 0.35, 1))
    rug = plane("Rug", 1.5, (0, 0, 0.01), mat_rug)

    print("Den room generated")
    return [floor, back_wall, left_wall, right_wall, window, window_frame,
            door_frame, blanket1, blanket2, rug]


# ---------------------------------------------------------------------------
# Kitchen Corner
# ---------------------------------------------------------------------------

def generate_kitchen():
    """Generate kitchen corner with bowls."""
    mat_ceramic = mat("Ceramic", (0.82, 0.78, 0.72, 1))   # off-white
    mat_water = mat("Water", (0.6, 0.75, 0.85, 1))         # light blue
    mat_food = mat("Food", (0.55, 0.40, 0.25, 1))          # brown kibble
    mat_mat_floor = mat("BowlMat", (0.5, 0.45, 0.38, 1))  # brown mat
    mat_bag = mat("FoodBag", (0.65, 0.58, 0.42, 1))        # paper bag

    # Bowl mat
    bowl_mat = plane("BowlMat", 0.5, (1.5, 1.5, 0.01), mat_mat_floor)

    # Food bowl (low-poly cylinder with hollow)
    food_bowl = cylinder("FoodBowl", 0.12, 0.06, (1.4, 1.5, 0.04), mat_ceramic)
    food = cylinder("Food", 0.10, 0.03, (1.4, 1.5, 0.06), mat_food)

    # Water bowl
    water_bowl = cylinder("WaterBowl", 0.12, 0.06, (1.6, 1.5, 0.04), mat_ceramic)
    water = cylinder("Water", 0.10, 0.02, (1.6, 1.5, 0.055), mat_water)

    # Food bag
    food_bag = box("FoodBag", (0.15, 0.10, 0.25), (1.7, 1.7, 0.25), mat_bag)
    food_bag.rotation_euler = (0, 0, 0.2)

    print("Kitchen generated")
    return [bowl_mat, food_bowl, food, water_bowl, water, food_bag]


# ---------------------------------------------------------------------------
# Garden
# ---------------------------------------------------------------------------

def generate_garden():
    """Generate garden with fence, grass, and flowers."""
    mat_grass = mat("Grass", (0.45, 0.65, 0.35, 1))       # green
    mat_fence = mat("Fence", (0.6, 0.52, 0.38, 1))        # wood
    mat_flower_r = mat("FlowerRed", (0.85, 0.3, 0.3, 1))  # red
    mat_flower_y = mat("FlowerYel", (0.95, 0.85, 0.3, 1)) # yellow
    mat_flower_p = mat("FlowerPur", (0.6, 0.35, 0.7, 1))  # purple
    mat_stem = mat("Stem", (0.35, 0.55, 0.25, 1))         # dark green
    mat_stone = mat("Stone", (0.65, 0.63, 0.58, 1))       # grey
    mat_gate = mat("Gate", (0.5, 0.42, 0.30, 1))          # darker wood

    # Grass ground
    grass = plane("GardenGrass", 6.0, (0, -4, 0), mat_grass)

    # Fence posts
    fence_parts = []
    for i in range(-6, 7, 2):
        post = box(f"FencePost_{i}", (0.04, 0.04, 0.3), (i * 0.5, -1.5, 0.3), mat_fence)
        fence_parts.append(post)

    # Fence rails
    rail_top = box("FenceRailTop", (3.0, 0.02, 0.02), (0, -1.5, 0.5), mat_fence)
    rail_bot = box("FenceRailBot", (3.0, 0.02, 0.02), (0, -1.5, 0.2), mat_fence)
    fence_parts.extend([rail_top, rail_bot])

    # Gate
    gate = box("Gate", (0.3, 0.03, 0.4), (0, -1.5, 0.35), mat_gate)
    fence_parts.append(gate)

    # Flowers (simple: cone on cylinder)
    flowers = []
    positions = [(-1.0, -2.5), (-0.5, -2.8), (0.3, -2.3), (0.8, -2.7), (1.2, -2.4)]
    colors = [mat_flower_r, mat_flower_y, mat_flower_p, mat_flower_r, mat_flower_y]

    for i, ((x, y), c) in enumerate(zip(positions, colors)):
        stem = cylinder(f"Stem_{i}", 0.01, 0.15, (x, y, 0.075), mat_stem)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.04, location=(x, y, 0.17))
        flower = bpy.context.active_object
        flower.name = f"Flower_{i}"
        flower.data.materials.append(c)
        for p in flower.data.polygons:
            p.use_smooth = False
        flowers.extend([stem, flower])

    # Stepping stones
    stones = []
    for i, (x, y) in enumerate([(0, -0.5), (0.15, -1.0), (-0.1, -1.5)]):
        stone = cylinder(f"Stone_{i}", 0.12, 0.03, (x, y, 0.015), mat_stone, vertices=6)
        stones.append(stone)

    print("Garden generated")
    return [grass] + fence_parts + flowers + stones


# ---------------------------------------------------------------------------
# Porch
# ---------------------------------------------------------------------------

def generate_porch():
    """Generate porch with step and swing."""
    mat_wood = mat("PorchWood", (0.55, 0.45, 0.35, 1))
    mat_cushion = mat("Cushion", (0.7, 0.55, 0.40, 1))
    mat_chain = mat("Chain", (0.5, 0.48, 0.45, 1))

    # Porch floor
    porch_floor = box("PorchFloor", (1.5, 0.8, 0.05), (0, -0.3, 0.15), mat_wood)

    # Step
    step = box("PorchStep", (1.0, 0.25, 0.08), (0, -0.8, 0.08), mat_wood)

    # Swing frame
    post_l = box("SwingPostL", (0.04, 0.04, 0.6), (-0.5, -0.1, 0.6), mat_wood)
    post_r = box("SwingPostR", (0.04, 0.04, 0.6), (0.5, -0.1, 0.6), mat_wood)
    beam = box("SwingBeam", (0.55, 0.04, 0.04), (0, -0.1, 0.9), mat_wood)

    # Swing seat
    seat = box("SwingSeat", (0.35, 0.2, 0.02), (0, -0.1, 0.4), mat_wood)
    cushion = box("SwingCushion", (0.30, 0.18, 0.03), (0, -0.1, 0.43), mat_cushion)

    # Chains
    chain_l = box("ChainL", (0.01, 0.01, 0.25), (-0.15, -0.1, 0.65), mat_chain)
    chain_r = box("ChainR", (0.01, 0.01, 0.25), (0.15, -0.1, 0.65), mat_chain)

    print("Porch generated")
    return [porch_floor, step, post_l, post_r, beam, seat, cushion, chain_l, chain_r]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_environments():
    """Generate all environment pieces and export."""
    clean_scene()

    print("=== Generating Living Tales Environments ===")

    den_parts = generate_den_room()
    kitchen_parts = generate_kitchen()
    garden_parts = generate_garden()
    porch_parts = generate_porch()

    # Add lighting
    bpy.ops.object.light_add(type='SUN', location=(3, -2, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.data.color = (1.0, 0.95, 0.85)
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))

    # Camera
    bpy.ops.object.camera_add(location=(3, -5, 3))
    cam = bpy.context.active_object
    cam.rotation_euler = (math.radians(65), 0, math.radians(15))
    bpy.context.scene.camera = cam

    # Render settings
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    total_polys = sum(
        len(obj.data.polygons)
        for obj in bpy.data.objects
        if obj.type == 'MESH'
    )
    print(f"\nTotal environment polygons: {total_polys}")
    print("Done!")


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "unity", "Assets", "Models", "Creature")
    os.makedirs(out_dir, exist_ok=True)

    generate_all_environments()

    # Export
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(out_dir, "environment.glb"),
        export_format='GLB',
        use_selection=False,
    )
    print(f"Exported: {os.path.join(out_dir, 'environment.glb')}")

    # Export
    bpy.ops.export_scene.gltf(
        filepath="//creature_environment.glb",
        export_format='GLB',
        use_selection=False,
    )
    print("Exported: creature_environment.glb")
