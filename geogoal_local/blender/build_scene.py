"""
build_scene.py – Blender script that builds a 3D football-pitch scene
from tracked match data and renders it to MP4.

Run via:
    blender --background --python geogoal_local/blender/build_scene.py -- \
        --data output/match_data.json --out output/render.mp4 \
        --blend output/scene.blend --fps 25
"""

import sys
import os
import json
import argparse
import math

try:
    import bpy
    import mathutils
except ImportError:
    print("[blender] ERROR: bpy not available. Run this script inside Blender:")
    print("  blender --background --python build_scene.py -- --data ... --out ...")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Parse CLI args (everything after "--")
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build Blender football scene")
    parser.add_argument("--data", required=True, help="Path to match_data.json")
    parser.add_argument("--out", required=True, help="Output MP4 path")
    parser.add_argument("--blend", required=True, help="Output .blend path")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    parser.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"],
                        help="Render engine")
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEAM_COLORS = {
    1:  (0.8, 0.1, 0.1, 1.0),   # home  – red
    2:  (0.1, 0.2, 0.8, 1.0),   # away  – blue
    0:  (0.9, 0.9, 0.1, 1.0),   # ref   – yellow
    -1: (0.5, 0.5, 0.5, 1.0),   # unknown – gray
}

def make_material(name, color, emission=False, emission_strength=2.5,
                  roughness=0.5):
    """Create and return a simple Principled-BSDF or Emission material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    if emission:
        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = color
        emit.inputs["Strength"].default_value = emission_strength
        emit.location = (0, 0)
        links.new(emit.outputs["Emission"], output.inputs["Surface"])
    else:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.location = (0, 0)
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat

# ---------------------------------------------------------------------------
# 2. Clear default scene
# ---------------------------------------------------------------------------

def clear_scene():
    print("[blender] Clearing default scene …")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block_coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights,
                       bpy.data.materials, bpy.data.curves, bpy.data.fonts):
        for block in block_coll:
            block_coll.remove(block)

# ---------------------------------------------------------------------------
# 3. Create the pitch (ground plane)
# ---------------------------------------------------------------------------

def create_pitch(length=105.0, width=68.0):
    print("[blender] Creating pitch …")
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    pitch = bpy.context.active_object
    pitch.name = "Pitch"
    pitch.scale = (length, width, 1)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("Grass")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Wave texture for mowing-stripe pattern
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-600, 0)

    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'X'
    wave.inputs["Scale"].default_value = 18.0
    wave.inputs["Distortion"].default_value = 0.0
    wave.inputs["Detail"].default_value = 0.0
    wave.location = (-400, 0)
    links.new(tex_coord.outputs["Object"], wave.inputs["Vector"])

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-100, 0)
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.04, 0.16, 0.02, 1.0)
    ramp.color_ramp.elements[1].position = 0.55
    ramp.color_ramp.elements[1].color = (0.06, 0.24, 0.03, 1.0)
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    pitch.data.materials.append(mat)
    return pitch

# ---------------------------------------------------------------------------
# 4. Draw pitch lines
# ---------------------------------------------------------------------------

def _white_line_material():
    """Shared emissive white material for all lines."""
    name = "PitchLine"
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    return make_material(name, (1, 1, 1, 1), emission=True,
                         emission_strength=2.5)

def _add_rect_line(name, x_min, x_max, y_min, y_max, z=0.01,
                   thickness=0.12):
    """Rectangle outline as four curve segments."""
    corners = [
        (x_min, y_min), (x_max, y_min),
        (x_max, y_max), (x_min, y_max),
    ]
    for i in range(4):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % 4]
        _add_line_segment(f"{name}_{i}", ax, ay, bx, by, z, thickness)

def _add_line_segment(name, x1, y1, x2, y2, z=0.01, thickness=0.12):
    curve = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = thickness / 2
    curve.bevel_resolution = 2

    spline = curve.splines.new('POLY')
    spline.points.add(1)
    spline.points[0].co = (x1, y1, z, 1)
    spline.points[1].co = (x2, y2, z, 1)

    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_white_line_material())
    return obj

def _add_circle_line(name, cx, cy, radius, z=0.01, thickness=0.12,
                     segments=64, arc_start=0.0, arc_end=2 * math.pi):
    """Full or partial circle as a curve."""
    curve = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = thickness / 2
    curve.bevel_resolution = 2

    n = segments
    spline = curve.splines.new('POLY')
    spline.points.add(n)  # n+1 points total
    for i in range(n + 1):
        t = arc_start + (arc_end - arc_start) * i / n
        x = cx + radius * math.cos(t)
        y = cy + radius * math.sin(t)
        spline.points[i].co = (x, y, z, 1)

    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_white_line_material())
    return obj

def _add_spot(name, cx, cy, z=0.01, radius=0.2):
    bpy.ops.mesh.primitive_circle_add(
        vertices=16, radius=radius, fill_type='NGON',
        location=(cx, cy, z))
    spot = bpy.context.active_object
    spot.name = name
    spot.data.materials.append(_white_line_material())
    return spot

def create_pitch_lines(length=105.0, width=68.0):
    print("[blender] Drawing pitch lines …")
    hl = length / 2   # 52.5
    hw = width / 2    # 34.0
    z = 0.01

    # Outer boundary
    _add_rect_line("Boundary", -hl, hl, -hw, hw, z)

    # Halfway line
    _add_line_segment("HalfwayLine", 0, -hw, 0, hw, z)

    # Center circle (r=9.15)
    _add_circle_line("CenterCircle", 0, 0, 9.15, z)

    # Center spot
    _add_spot("CenterSpot", 0, 0, z, 0.25)

    # Penalty areas (16.5m deep, 40.32m wide → ±20.16)
    pa_depth = 16.5
    pa_hw = 20.16
    _add_rect_line("PenaltyAreaL", -hl, -hl + pa_depth, -pa_hw, pa_hw, z)
    _add_rect_line("PenaltyAreaR", hl - pa_depth, hl, -pa_hw, pa_hw, z)

    # Goal areas (5.5m deep, 18.32m wide → ±9.16)
    ga_depth = 5.5
    ga_hw = 9.16
    _add_rect_line("GoalAreaL", -hl, -hl + ga_depth, -ga_hw, ga_hw, z)
    _add_rect_line("GoalAreaR", hl - ga_depth, hl, -ga_hw, ga_hw, z)

    # Penalty spots (11m from goal line → ±41.5 from center)
    ps_x = hl - 11.0  # 41.5
    _add_spot("PenaltySpotL", -ps_x, 0, z, 0.2)
    _add_spot("PenaltySpotR", ps_x, 0, z, 0.2)

    # Penalty arcs (partial circle r=9.15 at penalty spot, outside the area)
    # Left arc: center at (-41.5, 0), keep part with x > -36
    arc_half = math.acos(pa_depth / 9.15)  # ≈ angle where arc meets area edge
    _add_circle_line("PenaltyArcL", -ps_x, 0, 9.15, z, 0.12, 32,
                     -arc_half, arc_half)
    _add_circle_line("PenaltyArcR", ps_x, 0, 9.15, z, 0.12, 32,
                     math.pi - arc_half, math.pi + arc_half)

# ---------------------------------------------------------------------------
# 5. Stadium lighting
# ---------------------------------------------------------------------------

def create_lighting():
    print("[blender] Setting up stadium lighting …")

    # Sun lamp
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 80))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 4.0
    sun.data.color = (1.0, 0.95, 0.9)
    sun.rotation_euler = (math.radians(10), 0, 0)

    # Four corner floodlights
    corners = [
        ( 60,  40, 45),
        (-60,  40, 45),
        ( 60, -40, 45),
        (-60, -40, 45),
    ]
    for i, pos in enumerate(corners):
        bpy.ops.object.light_add(type='AREA', location=pos)
        light = bpy.context.active_object
        light.name = f"Floodlight_{i}"
        light.data.energy = 1500
        light.data.color = (1.0, 0.95, 0.85)
        light.data.size = 10.0
        light.data.use_shadow = True
        # Point toward center
        direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector(pos)
        light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # World ambient
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.01, 0.01, 0.02, 1.0)
        bg.inputs["Strength"].default_value = 0.3

# ---------------------------------------------------------------------------
# 6. Create player objects
# ---------------------------------------------------------------------------

def _team_material(team_id):
    name = f"Team_{team_id}"
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    color = TEAM_COLORS.get(team_id, TEAM_COLORS[-1])
    return make_material(name, color, roughness=0.6)

def create_player(player_id, team_id):
    """Build one player representation and return the root object."""
    mat = _team_material(team_id)

    # Body cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.5, depth=1.8, location=(0, 0, 0.9))
    body = bpy.context.active_object
    body.name = f"Player_{player_id}"
    body.data.materials.append(mat)

    # Direction cone on top
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.35, radius2=0, depth=0.4,
        location=(0, 0, 2.0))
    cone = bpy.context.active_object
    cone.name = f"Player_{player_id}_cone"
    cone.data.materials.append(mat)
    cone.parent = body

    # Text label
    bpy.ops.object.text_add(location=(0, 0, 2.4))
    label = bpy.context.active_object
    label.name = f"Player_{player_id}_label"
    label.data.body = str(player_id)
    label.data.size = 0.6
    label.data.align_x = 'CENTER'
    label.data.align_y = 'CENTER'
    label_mat = make_material(f"Label_{player_id}", (1, 1, 1, 1),
                              emission=True, emission_strength=3.0)
    label.data.materials.append(label_mat)
    label.parent = body

    # Billboard constraint — label always faces camera
    bpy.context.view_layer.objects.active = label
    bpy.ops.object.constraint_add(type='TRACK_TO')
    cam = bpy.data.objects.get("Camera")
    if cam:
        label.constraints["Track To"].target = cam
        label.constraints["Track To"].track_axis = 'TRACK_Z'
        label.constraints["Track To"].up_axis = 'UP_Y'

    return body

# ---------------------------------------------------------------------------
# 7. Create ball object
# ---------------------------------------------------------------------------

def create_ball():
    print("[blender] Creating ball …")
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.22, segments=24, ring_count=16,
        location=(0, 0, 0.22))
    ball = bpy.context.active_object
    ball.name = "Ball"
    mat = make_material("BallMat", (1.0, 0.95, 0.85, 1.0), roughness=0.3)
    ball.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ball

# ---------------------------------------------------------------------------
# 8. Animate positions
# ---------------------------------------------------------------------------

def set_linear_interpolation(obj):
    """Set all existing keyframes on *obj* to LINEAR interpolation."""
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

def animate(data, player_objects, ball_obj, fps):
    frames = data.get("frames", [])
    if not frames:
        print("[blender] No frames to animate.")
        return 1

    print(f"[blender] Animating {len(player_objects)} players + ball "
          f"over {len(frames)} data-frames …")

    max_frame = 1
    for fdata in frames:
        ts_ms = fdata.get("timestampMs", 0)
        frame_num = max(1, round(ts_ms / 1000.0 * fps))
        if frame_num > max_frame:
            max_frame = frame_num

        # Players
        for p in fdata.get("players", []):
            pid = p["playerId"]
            obj = player_objects.get(pid)
            if obj is None:
                continue
            bx = p["x"] - 52.5
            by = p["y"] - 34.0
            obj.location = (bx, by, 0)
            obj.keyframe_insert(data_path="location", frame=frame_num)

        # Ball
        ball_data = fdata.get("ball")
        if ball_data and ball_obj:
            bx = ball_data["x"] - 52.5
            by = ball_data["y"] - 34.0
            ball_obj.location = (bx, by, 0.22)
            ball_obj.keyframe_insert(data_path="location", frame=frame_num)

    # Set all to linear interpolation
    for obj in player_objects.values():
        set_linear_interpolation(obj)
    if ball_obj:
        set_linear_interpolation(ball_obj)

    print(f"[blender] Animation spans frames 1–{max_frame}")
    return max_frame

# ---------------------------------------------------------------------------
# 9. Camera setup
# ---------------------------------------------------------------------------

def create_camera(max_frame):
    print("[blender] Setting up camera …")

    # Orbit empty at origin
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    orbit_empty = bpy.context.active_object
    orbit_empty.name = "CameraOrbit"

    # Track target
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    track_target = bpy.context.active_object
    track_target.name = "CameraTarget"

    # Camera
    bpy.ops.object.camera_add(location=(0, -70, 50))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.data.lens = 35
    cam.data.clip_end = 500
    bpy.context.scene.camera = cam

    # Parent to orbit empty
    cam.parent = orbit_empty

    # Track-To constraint
    bpy.ops.object.constraint_add(type='TRACK_TO')
    cam.constraints["Track To"].target = track_target
    cam.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    cam.constraints["Track To"].up_axis = 'UP_Y'

    # Subtle orbit: 30° over the whole animation
    orbit_empty.rotation_euler = (0, 0, 0)
    orbit_empty.keyframe_insert(data_path="rotation_euler", frame=1)
    orbit_empty.rotation_euler = (0, 0, math.radians(30))
    orbit_empty.keyframe_insert(data_path="rotation_euler", frame=max_frame)
    set_linear_interpolation(orbit_empty)

    return cam

# ---------------------------------------------------------------------------
# 10. Render settings
# ---------------------------------------------------------------------------

def configure_render(engine, fps, max_frame, output_path):
    print(f"[blender] Configuring render: {engine}, {fps} fps, "
          f"frames 1–{max_frame}")

    scene = bpy.context.scene
    if engine == "CYCLES":
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 64
    else:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        # Fallback for Blender < 4.0 where the id was BLENDER_EEVEE
        if scene.render.engine != 'BLENDER_EEVEE_NEXT':
            try:
                scene.render.engine = 'BLENDER_EEVEE'
            except Exception:
                pass
        try:
            scene.eevee.use_soft_shadows = True
            scene.eevee.taa_render_samples = 64
        except AttributeError:
            pass

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = max_frame

    scene.render.filepath = os.path.abspath(output_path)
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load data
    print(f"[blender] Loading data from {args.data} …")
    with open(args.data, "r") as f:
        data = json.load(f)

    fps = args.fps
    frames = data.get("frames", [])
    pitch_info = data.get("pitch", {})
    pitch_length = pitch_info.get("length_m", 105.0)
    pitch_width = pitch_info.get("width_m", 68.0)

    # 2. Clear
    clear_scene()

    # 3. Pitch
    create_pitch(pitch_length, pitch_width)

    # 4. Lines
    create_pitch_lines(pitch_length, pitch_width)

    # 5. Lighting
    create_lighting()

    # 9. Camera first (so player labels can track it)
    # We compute max_frame early to set up orbit keyframes
    if frames:
        max_frame = max(
            1,
            max(round(f.get("timestampMs", 0) / 1000.0 * fps) for f in frames)
        )
    else:
        max_frame = 1
    cam = create_camera(max_frame)

    # 6. Players
    print("[blender] Creating player objects …")
    # Collect unique players and their team
    player_teams = {}
    for fdata in frames:
        for p in fdata.get("players", []):
            pid = p["playerId"]
            if pid not in player_teams:
                player_teams[pid] = p.get("teamId", -1)

    player_objects = {}
    for pid, tid in player_teams.items():
        player_objects[pid] = create_player(pid, tid)

    print(f"[blender] Created {len(player_objects)} player(s)")

    # 7. Ball
    ball_obj = create_ball()

    # 8. Animate
    max_frame = animate(data, player_objects, ball_obj, fps)

    # 10. Render settings
    output_path = args.out
    blend_path = args.blend
    configure_render(args.engine, fps, max_frame, output_path)

    # 11. Save .blend
    os.makedirs(os.path.dirname(os.path.abspath(blend_path)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blend_path))
    print(f"[blender] Saved {blend_path}")

    # Render animation
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    bpy.ops.render.render(animation=True)
    print(f"[blender] Rendered to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[blender] FATAL: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
