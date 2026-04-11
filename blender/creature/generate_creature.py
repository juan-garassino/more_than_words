"""
Living Tales — Low-Poly Creature Generator

Run inside Blender:
    blender --background --python blender/creature/generate_creature.py

Generates a low-poly creature with armature, ready for animation.
~500 triangles, warm flat-shaded materials, rigged with bones for
ears, tail, eyes, jaw, and legs.
"""
import bpy
import bmesh
import math
from mathutils import Vector, Matrix


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def clean_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def make_flat_material(name, color):
    """Create a flat-shaded material with given RGBA color."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 1.0
    # Blender 3.x uses "Specular", 4.x uses "Specular IOR Level"
    spec_key = "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    bsdf.inputs[spec_key].default_value = 0.0
    return mat


# ---------------------------------------------------------------------------
# Mesh creation helpers
# ---------------------------------------------------------------------------

def create_ico_sphere(name, radius, subdivisions, location, material):
    """Create a low-poly ico sphere."""
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivisions,
        radius=radius,
        location=location,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name
    obj.data.materials.append(material)
    # Flat shading
    for poly in obj.data.polygons:
        poly.use_smooth = False
    return obj


def create_cylinder(name, radius, depth, location, rotation, material):
    """Create a low-poly cylinder (for legs)."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=6,
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name
    obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = False
    return obj


def create_cone(name, radius, depth, location, rotation, material):
    """Create a low-poly cone (for ears)."""
    bpy.ops.mesh.primitive_cone_add(
        vertices=4,
        radius1=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name
    obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = False
    return obj


# ---------------------------------------------------------------------------
# Creature assembly
# ---------------------------------------------------------------------------

def generate_creature():
    """Generate the complete low-poly creature."""
    clean_scene()

    # Materials
    mat_body = make_flat_material("Body", (0.76, 0.65, 0.52, 1.0))      # warm cream
    mat_belly = make_flat_material("Belly", (0.85, 0.78, 0.68, 1.0))    # lighter belly
    mat_nose = make_flat_material("Nose", (0.25, 0.18, 0.15, 1.0))      # dark brown
    mat_eye = make_flat_material("Eye", (0.1, 0.08, 0.06, 1.0))         # near black
    mat_eye_highlight = make_flat_material("EyeHighlight", (1, 1, 1, 1)) # white dot
    mat_ear_inner = make_flat_material("EarInner", (0.85, 0.65, 0.62, 1.0))  # pink
    mat_paw = make_flat_material("Paw", (0.6, 0.5, 0.4, 1.0))          # darker paws

    # --- Body (main blob) ---
    body = create_ico_sphere("Body", radius=0.5, subdivisions=2, location=(0, 0, 0.6), material=mat_body)
    # Slightly elongate body
    body.scale = (0.6, 0.45, 0.4)
    bpy.ops.object.transform_apply(scale=True)

    # --- Head ---
    head = create_ico_sphere("Head", radius=0.3, subdivisions=2, location=(0.55, 0, 0.75), material=mat_body)

    # --- Snout ---
    snout = create_ico_sphere("Snout", radius=0.12, subdivisions=1, location=(0.8, 0, 0.7), material=mat_body)
    snout.scale = (1.2, 0.8, 0.7)
    bpy.ops.object.transform_apply(scale=True)

    # --- Nose ---
    nose = create_ico_sphere("Nose", radius=0.04, subdivisions=1, location=(0.92, 0, 0.72), material=mat_nose)

    # --- Eyes ---
    eye_l = create_ico_sphere("EyeL", radius=0.06, subdivisions=1, location=(0.72, 0.12, 0.85), material=mat_eye)
    eye_r = create_ico_sphere("EyeR", radius=0.06, subdivisions=1, location=(0.72, -0.12, 0.85), material=mat_eye)

    # Eye highlights
    hl_l = create_ico_sphere("HighlightL", radius=0.02, subdivisions=0, location=(0.75, 0.14, 0.87), material=mat_eye_highlight)
    hl_r = create_ico_sphere("HighlightR", radius=0.02, subdivisions=0, location=(0.75, -0.10, 0.87), material=mat_eye_highlight)

    # --- Ears ---
    ear_l = create_cone("EarL", radius=0.08, depth=0.2, location=(0.5, 0.2, 1.05),
                        rotation=(0, -0.3, 0.2), material=mat_ear_inner)
    ear_r = create_cone("EarR", radius=0.08, depth=0.2, location=(0.5, -0.2, 1.05),
                        rotation=(0, -0.3, -0.2), material=mat_ear_inner)

    # --- Legs ---
    leg_fl = create_cylinder("LegFL", radius=0.06, depth=0.35, location=(0.3, 0.2, 0.18),
                             rotation=(0, 0, 0), material=mat_paw)
    leg_fr = create_cylinder("LegFR", radius=0.06, depth=0.35, location=(0.3, -0.2, 0.18),
                             rotation=(0, 0, 0), material=mat_paw)
    leg_bl = create_cylinder("LegBL", radius=0.06, depth=0.35, location=(-0.3, 0.2, 0.18),
                             rotation=(0, 0, 0), material=mat_paw)
    leg_br = create_cylinder("LegBR", radius=0.06, depth=0.35, location=(-0.3, -0.2, 0.18),
                             rotation=(0, 0, 0), material=mat_paw)

    # --- Tail (3 segments) ---
    tail1 = create_cylinder("Tail1", radius=0.04, depth=0.15, location=(-0.55, 0, 0.65),
                            rotation=(0, math.radians(70), 0), material=mat_body)
    tail2 = create_cylinder("Tail2", radius=0.03, depth=0.12, location=(-0.65, 0, 0.75),
                            rotation=(0, math.radians(50), 0), material=mat_body)
    tail3 = create_ico_sphere("TailTip", radius=0.035, subdivisions=1, location=(-0.72, 0, 0.83),
                              material=mat_body)

    # --- Join all meshes ---
    all_parts = [body, head, snout, nose, eye_l, eye_r, hl_l, hl_r,
                 ear_l, ear_r, leg_fl, leg_fr, leg_bl, leg_br,
                 tail1, tail2, tail3]

    # Select all parts
    bpy.ops.object.select_all(action='DESELECT')
    for obj in all_parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()

    creature = bpy.context.active_object
    creature.name = "Creature"

    # --- Create Armature ---
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.armature_add(location=(0, 0, 0))
    armature_obj = bpy.context.active_object
    armature_obj.name = "CreatureArmature"
    armature = armature_obj.data
    armature.name = "CreatureArmature"

    bpy.ops.object.mode_set(mode='EDIT')

    # Root bone
    root = armature.edit_bones["Bone"]
    root.name = "Root"
    root.head = Vector((0, 0, 0))
    root.tail = Vector((0, 0, 0.6))

    # Spine
    spine = armature.edit_bones.new("Spine")
    spine.head = Vector((0, 0, 0.6))
    spine.tail = Vector((0.3, 0, 0.65))
    spine.parent = root

    # Head bone
    head_bone = armature.edit_bones.new("Head")
    head_bone.head = Vector((0.3, 0, 0.65))
    head_bone.tail = Vector((0.55, 0, 0.75))
    head_bone.parent = spine

    # Jaw
    jaw = armature.edit_bones.new("Jaw")
    jaw.head = Vector((0.7, 0, 0.7))
    jaw.tail = Vector((0.85, 0, 0.65))
    jaw.parent = head_bone

    # Ears
    ear_l_bone = armature.edit_bones.new("EarL")
    ear_l_bone.head = Vector((0.5, 0.2, 0.95))
    ear_l_bone.tail = Vector((0.5, 0.2, 1.1))
    ear_l_bone.parent = head_bone

    ear_r_bone = armature.edit_bones.new("EarR")
    ear_r_bone.head = Vector((0.5, -0.2, 0.95))
    ear_r_bone.tail = Vector((0.5, -0.2, 1.1))
    ear_r_bone.parent = head_bone

    # Tail chain
    tail_bone1 = armature.edit_bones.new("Tail1")
    tail_bone1.head = Vector((-0.45, 0, 0.6))
    tail_bone1.tail = Vector((-0.58, 0, 0.7))
    tail_bone1.parent = root

    tail_bone2 = armature.edit_bones.new("Tail2")
    tail_bone2.head = Vector((-0.58, 0, 0.7))
    tail_bone2.tail = Vector((-0.68, 0, 0.8))
    tail_bone2.parent = tail_bone1

    tail_bone3 = armature.edit_bones.new("TailTip")
    tail_bone3.head = Vector((-0.68, 0, 0.8))
    tail_bone3.tail = Vector((-0.75, 0, 0.88))
    tail_bone3.parent = tail_bone2

    # Legs
    for name, x, y in [("LegFL", 0.3, 0.2), ("LegFR", 0.3, -0.2),
                        ("LegBL", -0.3, 0.2), ("LegBR", -0.3, -0.2)]:
        bone = armature.edit_bones.new(name)
        bone.head = Vector((x, y, 0.4))
        bone.tail = Vector((x, y, 0.0))
        bone.parent = root

    bpy.ops.object.mode_set(mode='OBJECT')

    # --- Parent mesh to armature ---
    bpy.ops.object.select_all(action='DESELECT')
    creature.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    # --- Set flat shading on final mesh ---
    bpy.ops.object.select_all(action='DESELECT')
    creature.select_set(True)
    bpy.context.view_layer.objects.active = creature
    bpy.ops.object.shade_flat()

    # --- Camera ---
    bpy.ops.object.camera_add(location=(1.5, -1.5, 1.0))
    camera = bpy.context.active_object
    camera.name = "CreatureCamera"
    camera.rotation_euler = (math.radians(65), 0, math.radians(45))
    bpy.context.scene.camera = camera

    # --- Light ---
    bpy.ops.object.light_add(type='SUN', location=(2, -1, 3))
    light = bpy.context.active_object
    light.name = "KeyLight"
    light.data.energy = 3.0
    light.data.color = (1.0, 0.95, 0.85)  # warm

    # --- Render settings ---
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.85, 0.9, 0.95, 1)

    print(f"Creature generated: {len(creature.data.polygons)} polygons")
    print(f"Armature bones: {len(armature.bones)}")

    return creature, armature_obj


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_glb(filepath="creature.glb"):
    """Export the creature as GLB for Unity."""
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        use_selection=False,
        export_animations=True,
    )
    print(f"Exported: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "unity", "Assets", "Models", "Creature")
    os.makedirs(out_dir, exist_ok=True)

    creature, armature = generate_creature()

    # Export
    export_glb(os.path.join(out_dir, "creature.glb"))
