"""
Living Tales — Creature Animation Generator

Run inside Blender AFTER generate_creature.py:
    blender creature.blend --python blender/creature/generate_animations.py

Or run standalone (generates creature first):
    blender --background --python blender/creature/generate_animations.py

Creates NLA animation strips mapped to Living Tales token categories.
Each animation is a short keyframed sequence on the creature armature.
"""
import bpy
import math
from mathutils import Vector, Euler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_armature():
    """Find the creature armature in the scene."""
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and 'Creature' in obj.name:
            return obj
    return None


def clear_all_actions():
    """Remove existing actions."""
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)


def set_bone_keyframe(armature, bone_name, frame, location=None, rotation=None, scale=None):
    """Set a keyframe on a pose bone."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bone = armature.pose.bones.get(bone_name)
    if not bone:
        return

    bpy.context.scene.frame_set(frame)

    if location is not None:
        bone.location = Vector(location)
        bone.keyframe_insert(data_path="location", frame=frame)

    if rotation is not None:
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = Euler(rotation)
        bone.keyframe_insert(data_path="rotation_euler", frame=frame)

    if scale is not None:
        bone.scale = Vector(scale)
        bone.keyframe_insert(data_path="scale", frame=frame)

    bpy.ops.object.mode_set(mode='OBJECT')


def create_action(armature, name, frame_start, frame_end):
    """Create a new action and assign it to the armature."""
    action = bpy.data.actions.new(name=name)
    armature.animation_data_create()
    armature.animation_data.action = action
    return action


def push_to_nla(armature, action_name):
    """Push current action to NLA strip."""
    if armature.animation_data and armature.animation_data.action:
        track = armature.animation_data.nla_tracks.new()
        track.name = action_name
        track.strips.new(action_name, 1, armature.animation_data.action)
        armature.animation_data.action = None


def reset_pose(armature):
    """Reset all pose bones to rest position."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.rot_clear()
    bpy.ops.pose.loc_clear()
    bpy.ops.pose.scale_clear()
    bpy.ops.object.mode_set(mode='OBJECT')


# ---------------------------------------------------------------------------
# Animation definitions
# ---------------------------------------------------------------------------

def anim_idle_breathe(armature):
    """Gentle breathing loop. 60 frames = 2.5s at 24fps."""
    name = "idle_breathe"
    create_action(armature, name, 1, 60)

    # Subtle body rise/fall
    set_bone_keyframe(armature, "Spine", 1, location=(0, 0, 0))
    set_bone_keyframe(armature, "Spine", 15, location=(0, 0, 0.008))
    set_bone_keyframe(armature, "Spine", 30, location=(0, 0, 0))
    set_bone_keyframe(armature, "Spine", 45, location=(0, 0, 0.008))
    set_bone_keyframe(armature, "Spine", 60, location=(0, 0, 0))

    # Slight ear twitch
    set_bone_keyframe(armature, "EarL", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarL", 20, rotation=(0.05, 0, 0))
    set_bone_keyframe(armature, "EarL", 40, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarL", 60, rotation=(0, 0, 0))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_wag_tail(armature):
    """Happy tail wag. 24 frames = 1s."""
    name = "wag_tail"
    create_action(armature, name, 1, 24)

    for f in range(1, 25, 3):
        angle = math.sin(f * 1.5) * 0.4
        set_bone_keyframe(armature, "Tail1", f, rotation=(0, 0, angle))
        set_bone_keyframe(armature, "Tail2", f, rotation=(0, 0, angle * 1.3))
        set_bone_keyframe(armature, "TailTip", f, rotation=(0, 0, angle * 1.5))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_ears_droop(armature):
    """Sad ears drooping. 24 frames = 1s."""
    name = "ears_droop"
    create_action(armature, name, 1, 24)

    set_bone_keyframe(armature, "EarL", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarR", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarL", 12, rotation=(0, 0.4, -0.2))
    set_bone_keyframe(armature, "EarR", 12, rotation=(0, 0.4, 0.2))
    set_bone_keyframe(armature, "EarL", 24, rotation=(0, 0.5, -0.3))
    set_bone_keyframe(armature, "EarR", 24, rotation=(0, 0.5, 0.3))

    # Head drops slightly
    set_bone_keyframe(armature, "Head", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "Head", 24, rotation=(0.15, 0, 0))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_eat_bowl(armature):
    """Eating from bowl. 48 frames = 2s."""
    name = "eat_bowl"
    create_action(armature, name, 1, 48)

    # Head dips down to bowl level
    set_bone_keyframe(armature, "Head", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "Head", 10, rotation=(0.4, 0, 0))

    # Repeated eating motion (head bobs)
    for f in range(10, 45, 5):
        up = 0.35 if f % 10 < 5 else 0.45
        set_bone_keyframe(armature, "Head", f, rotation=(up, 0, 0))

    # Head comes back up
    set_bone_keyframe(armature, "Head", 48, rotation=(0, 0, 0))

    # Tail wags while eating
    for f in range(1, 49, 4):
        angle = math.sin(f * 1.2) * 0.25
        set_bone_keyframe(armature, "Tail1", f, rotation=(0, 0, angle))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_look_at_bowl(armature):
    """Looking longingly at bowl. 36 frames = 1.5s."""
    name = "look_at_bowl"
    create_action(armature, name, 1, 36)

    # Turn head slightly toward bowl direction
    set_bone_keyframe(armature, "Head", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "Head", 12, rotation=(0.1, 0, 0.15))
    set_bone_keyframe(armature, "Head", 24, rotation=(0.1, 0, 0.15))
    set_bone_keyframe(armature, "Head", 36, rotation=(0.1, 0, 0.15))

    # Ears perk forward
    set_bone_keyframe(armature, "EarL", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarR", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarL", 12, rotation=(-0.2, 0, 0))
    set_bone_keyframe(armature, "EarR", 12, rotation=(-0.2, 0, 0))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_play_bounce(armature):
    """Playful bouncing. 48 frames = 2s."""
    name = "play_bounce"
    create_action(armature, name, 1, 48)

    # Body bounces up and down
    for f in range(1, 49, 6):
        height = abs(math.sin(f * 0.5)) * 0.06
        set_bone_keyframe(armature, "Root", f, location=(0, 0, height))

    # Front legs play bow at start
    set_bone_keyframe(armature, "LegFL", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "LegFR", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "LegFL", 8, rotation=(0.3, 0, 0))
    set_bone_keyframe(armature, "LegFR", 8, rotation=(0.3, 0, 0))
    set_bone_keyframe(armature, "LegFL", 16, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "LegFR", 16, rotation=(0, 0, 0))

    # Tail goes wild
    for f in range(1, 49, 3):
        angle = math.sin(f * 2.0) * 0.5
        set_bone_keyframe(armature, "Tail1", f, rotation=(0, 0, angle))
        set_bone_keyframe(armature, "Tail2", f, rotation=(0, 0, angle * 1.5))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_yawn_stretch(armature):
    """Yawning and stretching. 48 frames = 2s."""
    name = "yawn_stretch"
    create_action(armature, name, 1, 48)

    # Jaw opens wide
    set_bone_keyframe(armature, "Jaw", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "Jaw", 12, rotation=(0.3, 0, 0))
    set_bone_keyframe(armature, "Jaw", 24, rotation=(0.35, 0, 0))
    set_bone_keyframe(armature, "Jaw", 36, rotation=(0.1, 0, 0))
    set_bone_keyframe(armature, "Jaw", 48, rotation=(0, 0, 0))

    # Front legs stretch forward
    set_bone_keyframe(armature, "LegFL", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "LegFR", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "LegFL", 20, rotation=(-0.3, 0, 0))
    set_bone_keyframe(armature, "LegFR", 20, rotation=(-0.3, 0, 0))
    set_bone_keyframe(armature, "LegFL", 48, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "LegFR", 48, rotation=(0, 0, 0))

    # Body stretches
    set_bone_keyframe(armature, "Spine", 1, location=(0, 0, 0))
    set_bone_keyframe(armature, "Spine", 20, location=(0.03, 0, -0.02))
    set_bone_keyframe(armature, "Spine", 48, location=(0, 0, 0))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_shiver(armature):
    """Shivering from cold/illness. 36 frames = 1.5s."""
    name = "shiver"
    create_action(armature, name, 1, 36)

    # Rapid small rotations on body
    for f in range(1, 37, 2):
        shake = math.sin(f * 5) * 0.015
        set_bone_keyframe(armature, "Spine", f, rotation=(0, 0, shake))
        set_bone_keyframe(armature, "Head", f, rotation=(0, 0, shake * 0.5))

    # Ears flat
    set_bone_keyframe(armature, "EarL", 1, rotation=(0, 0.3, -0.2))
    set_bone_keyframe(armature, "EarR", 1, rotation=(0, 0.3, 0.2))
    set_bone_keyframe(armature, "EarL", 36, rotation=(0, 0.3, -0.2))
    set_bone_keyframe(armature, "EarR", 36, rotation=(0, 0.3, 0.2))

    # Tail tucked
    set_bone_keyframe(armature, "Tail1", 1, rotation=(-0.3, 0, 0))
    set_bone_keyframe(armature, "Tail1", 36, rotation=(-0.3, 0, 0))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_lean_against(armature):
    """Leaning against player's leg. 48 frames = 2s."""
    name = "lean_against"
    create_action(armature, name, 1, 48)

    # Body tilts to one side
    set_bone_keyframe(armature, "Root", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "Root", 16, rotation=(0, 0.15, 0))
    set_bone_keyframe(armature, "Root", 48, rotation=(0, 0.15, 0))

    # Head rests
    set_bone_keyframe(armature, "Head", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "Head", 20, rotation=(0.1, 0.1, 0))
    set_bone_keyframe(armature, "Head", 48, rotation=(0.1, 0.1, 0))

    # Eyes close (scale down)
    # Tail slow wag
    for f in range(16, 49, 8):
        angle = math.sin(f * 0.5) * 0.15
        set_bone_keyframe(armature, "Tail1", f, rotation=(0, 0, angle))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_sleep_curl(armature):
    """Curled up sleeping. 72 frames = 3s loop."""
    name = "sleep_curl"
    create_action(armature, name, 1, 72)

    # Body low to ground
    set_bone_keyframe(armature, "Root", 1, location=(0, 0, -0.08))
    set_bone_keyframe(armature, "Root", 72, location=(0, 0, -0.08))

    # Legs tucked
    for leg in ["LegFL", "LegFR", "LegBL", "LegBR"]:
        set_bone_keyframe(armature, leg, 1, rotation=(0.5, 0, 0))
        set_bone_keyframe(armature, leg, 72, rotation=(0.5, 0, 0))

    # Head resting
    set_bone_keyframe(armature, "Head", 1, rotation=(0.2, 0, 0.1))
    set_bone_keyframe(armature, "Head", 72, rotation=(0.2, 0, 0.1))

    # Gentle breathing
    set_bone_keyframe(armature, "Spine", 1, location=(0, 0, 0))
    set_bone_keyframe(armature, "Spine", 18, location=(0, 0, 0.005))
    set_bone_keyframe(armature, "Spine", 36, location=(0, 0, 0))
    set_bone_keyframe(armature, "Spine", 54, location=(0, 0, 0.005))
    set_bone_keyframe(armature, "Spine", 72, location=(0, 0, 0))

    # Tail wrapped around body
    set_bone_keyframe(armature, "Tail1", 1, rotation=(0, -0.3, 0.4))
    set_bone_keyframe(armature, "Tail2", 1, rotation=(0, -0.2, 0.5))
    set_bone_keyframe(armature, "Tail1", 72, rotation=(0, -0.3, 0.4))
    set_bone_keyframe(armature, "Tail2", 72, rotation=(0, -0.2, 0.5))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_startle(armature):
    """Startled jump. 24 frames = 1s."""
    name = "startle"
    create_action(armature, name, 1, 24)

    # Quick jump up
    set_bone_keyframe(armature, "Root", 1, location=(0, 0, 0))
    set_bone_keyframe(armature, "Root", 4, location=(0, 0, 0.06))
    set_bone_keyframe(armature, "Root", 10, location=(0, 0, 0))
    set_bone_keyframe(armature, "Root", 24, location=(0, 0, 0))

    # Ears snap up
    set_bone_keyframe(armature, "EarL", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarR", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarL", 4, rotation=(-0.4, 0, 0))
    set_bone_keyframe(armature, "EarR", 4, rotation=(-0.4, 0, 0))
    set_bone_keyframe(armature, "EarL", 24, rotation=(-0.1, 0, 0))
    set_bone_keyframe(armature, "EarR", 24, rotation=(-0.1, 0, 0))

    # Tail puffs up (scale)
    set_bone_keyframe(armature, "TailTip", 1, scale=(1, 1, 1))
    set_bone_keyframe(armature, "TailTip", 4, scale=(1.5, 1.5, 1.5))
    set_bone_keyframe(armature, "TailTip", 24, scale=(1, 1, 1))

    push_to_nla(armature, name)
    reset_pose(armature)


def anim_look_at_door(armature):
    """Looking toward door/visitor. 36 frames = 1.5s."""
    name = "look_at_door"
    create_action(armature, name, 1, 36)

    # Head turns toward door
    set_bone_keyframe(armature, "Head", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "Head", 10, rotation=(-0.1, 0.3, 0))
    set_bone_keyframe(armature, "Head", 36, rotation=(-0.1, 0.3, 0))

    # Ears perk forward
    set_bone_keyframe(armature, "EarL", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarR", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "EarL", 10, rotation=(-0.3, 0, 0.1))
    set_bone_keyframe(armature, "EarR", 10, rotation=(-0.3, 0, -0.1))
    set_bone_keyframe(armature, "EarL", 36, rotation=(-0.3, 0, 0.1))
    set_bone_keyframe(armature, "EarR", 36, rotation=(-0.3, 0, -0.1))

    # Tail up alert
    set_bone_keyframe(armature, "Tail1", 1, rotation=(0, 0, 0))
    set_bone_keyframe(armature, "Tail1", 10, rotation=(0.2, 0, 0))
    set_bone_keyframe(armature, "Tail1", 36, rotation=(0.2, 0, 0))

    push_to_nla(armature, name)
    reset_pose(armature)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_animations():
    """Generate all creature animations."""
    armature = get_armature()
    if not armature:
        # Try generating creature first
        import importlib
        import sys
        sys.path.insert(0, ".")
        from blender.creature.generate_creature import generate_creature
        _, armature = generate_creature()

    clear_all_actions()

    print("Generating animations...")

    animations = [
        ("idle_breathe", anim_idle_breathe),
        ("wag_tail", anim_wag_tail),
        ("ears_droop", anim_ears_droop),
        ("eat_bowl", anim_eat_bowl),
        ("look_at_bowl", anim_look_at_bowl),
        ("play_bounce", anim_play_bounce),
        ("yawn_stretch", anim_yawn_stretch),
        ("shiver", anim_shiver),
        ("lean_against", anim_lean_against),
        ("sleep_curl", anim_sleep_curl),
        ("startle", anim_startle),
        ("look_at_door", anim_look_at_door),
    ]

    for name, func in animations:
        func(armature)
        print(f"  Created: {name}")

    print(f"Total: {len(animations)} animations, {len(bpy.data.actions)} actions")
    return armature


if __name__ == "__main__":
    armature = generate_all_animations()

    # Export with animations
    bpy.ops.export_scene.gltf(
        filepath="//creature_animated.glb",
        export_format='GLB',
        use_selection=False,
        export_animations=True,
        export_nla_strips=True,
    )
    print("Exported: creature_animated.glb")
