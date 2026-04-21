# softform.py
# SoftForm - Blender Add-on for soft furniture mesh deformations
# Author: Generated
# Blender: 5.0+
# All UI labels in Dutch, code comments in English

bl_info = {
    "name": "SoftForm",
    "author": "Generated",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "View3D > N-Panel > SoftForm",
    "description": "Pas stofachtige vervormingen toe voor meubelmodellen",
    "category": "Mesh",
}

import bpy
import bmesh
import json
import math
import random
import traceback
from mathutils import Vector, noise as mnoise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SF_PREFIX = "SF_Zone_"
SF_META_KEY = "softform_zones"
SF_ORIG_KEY = "softform_orig_positions"
ZONE_COLORS = [
    (0.9, 0.2, 0.2, 1.0),
    (0.2, 0.7, 0.9, 1.0),
    (0.2, 0.9, 0.4, 1.0),
    (0.9, 0.7, 0.1, 1.0),
    (0.7, 0.2, 0.9, 1.0),
    (0.9, 0.5, 0.1, 1.0),
    (0.1, 0.9, 0.8, 1.0),
    (0.9, 0.1, 0.6, 1.0),
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_next_zone_index(obj):
    """Return the next available SF_ zone index for an object."""
    existing = [vg.name for vg in obj.vertex_groups if vg.name.startswith(SF_PREFIX)]
    idx = 1
    while f"{SF_PREFIX}{idx}" in existing:
        idx += 1
    return idx


def get_zone_color(index):
    return ZONE_COLORS[(index - 1) % len(ZONE_COLORS)]


def load_zone_meta(obj):
    """Load zone metadata dict from object custom property."""
    raw = obj.get(SF_META_KEY, None)
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def save_zone_meta(obj, meta):
    obj[SF_META_KEY] = json.dumps(meta)


def get_next_logical_zone_id(sf):
    """Return a new logical zone id (shared by multiple objects)."""
    max_idx = 0
    for z in sf.zones:
        zid = getattr(z, "zone_id", "")
        if not zid.startswith("zone_"):
            continue
        try:
            max_idx = max(max_idx, int(zid.split("_")[-1]))
        except Exception:
            pass
    return f"zone_{max_idx + 1}"


def load_zone_object_map(zone):
    """Read {object_name: vertex_group_name} mapping from zone property."""
    raw = getattr(zone, "object_vgroups_json", "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def save_zone_object_map(zone, object_map):
    """Persist {object_name: vertex_group_name} mapping on the zone."""
    zone.object_vgroups_json = json.dumps(object_map)


def iter_zone_object_groups(context, zone):
    """Yield tuples (obj, vg_name) for all objects linked to this logical zone."""
    object_map = load_zone_object_map(zone)

    # Backward compatibility for old scenes that only had one vgroup_name.
    if not object_map and zone.vgroup_name:
        for obj in context.scene.objects:
            if obj.type != 'MESH':
                continue
            if obj.vertex_groups.get(zone.vgroup_name):
                yield obj, zone.vgroup_name
        return

    for obj_name, vg_name in object_map.items():
        obj = context.scene.objects.get(obj_name)
        if obj is None or obj.type != 'MESH':
            continue
        if obj.vertex_groups.get(vg_name) is None:
            continue
        yield obj, vg_name


def create_zone_entry(sf, logical_zone_id, zone_name, color, object_map, fallback_vgroup=""):
    """Create one logical zone entry in UI and store object->vertex-group mapping."""
    zone = sf.zones.add()
    zone.zone_id = logical_zone_id
    zone.zone_name = zone_name
    zone.vgroup_name = fallback_vgroup  # legacy compatibility only
    zone.color = color
    save_zone_object_map(zone, object_map)
    sf.active_zone_index = len(sf.zones) - 1
    return zone


def gather_multi_object_face_selection(context):
    """Collect selected vertex indices per object from multi-object edit mode."""
    selected_by_object = {}
    objs = getattr(context, "objects_in_mode_unique_data", [])
    for obj in objs:
        if obj.type != 'MESH' or obj.mode != 'EDIT':
            continue
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        # We convert selected faces to vertices so the vertex group stores the same region.
        selected_verts = {v.index for f in bm.faces if f.select for v in f.verts}
        if selected_verts:
            selected_by_object[obj] = sorted(selected_verts)
    return selected_by_object


def get_protected_vert_indices(bm):
    """Return set of vertex indices that should not be displaced (open/seam edges)."""
    protected = set()
    for e in bm.edges:
        is_seam = e.seam
        is_boundary = not e.is_manifold
        if is_seam or is_boundary:
            for v in e.verts:
                protected.add(v.index)
    return protected


def store_original_positions(obj):
    """Store current vertex positions as a custom property JSON blob."""
    mesh = obj.data
    positions = {str(v.index): list(v.co) for v in mesh.vertices}
    obj[SF_ORIG_KEY] = json.dumps(positions)


def restore_original_positions(obj):
    """Restore vertex positions from stored custom property."""
    raw = obj.get(SF_ORIG_KEY, None)
    if raw is None:
        return False
    try:
        positions = json.loads(raw)
        mesh = obj.data
        for v in mesh.vertices:
            key = str(v.index)
            if key in positions:
                v.co = Vector(positions[key])
        mesh.update()
        return True
    except Exception as e:
        print(f"[SoftForm] restore_original_positions error: {e}")
        return False


def reproject_box_uv(bm, original_positions):
    """Re-project UVs using box (triplanar) mapping from original positions."""
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        return
    for face in bm.faces:
        n = face.normal
        ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
        for loop in face.loops:
            orig_co = original_positions.get(loop.vert.index)
            if orig_co is None:
                continue
            if az >= ax and az >= ay:
                loop[uv_layer].uv = (orig_co.x, orig_co.y)
            elif ax >= ay:
                loop[uv_layer].uv = (orig_co.y, orig_co.z)
            else:
                loop[uv_layer].uv = (orig_co.x, orig_co.z)


def get_vgroup_weights(bm, vgroup_index):
    """Return dict {vert_index: weight} for a vertex group."""
    deform_layer = bm.verts.layers.deform.active
    weights = {}
    if deform_layer is None:
        return weights
    for v in bm.verts:
        w = v[deform_layer].get(vgroup_index, 0.0)
        if w > 0.0:
            weights[v.index] = w
    return weights


# ---------------------------------------------------------------------------
# Deformation functions
# ---------------------------------------------------------------------------

def apply_inflate(bm, vgroup_index, params, weight_mul, protect_set, preserve_uv, orig_positions):
    deform_layer = bm.verts.layers.deform.active
    if deform_layer is None:
        return
    direction = params.inflate_direction  # -1 flatten, +1 inflate
    intensity = params.inflate_intensity * weight_mul
    falloff = params.inflate_falloff

    for v in bm.verts:
        if v.index in protect_set:
            continue
        w = v[deform_layer].get(vgroup_index, 0.0)
        if w <= 0.0:
            continue
        # Apply falloff to weight
        if falloff == 'SHARP':
            fw = w ** 3
        elif falloff == 'LINEAR':
            fw = w
        else:  # SMOOTH
            fw = w * w * (3 - 2 * w)
        disp = v.normal * intensity * direction * fw
        v.co += disp

    if preserve_uv:
        reproject_box_uv(bm, orig_positions)


def apply_waves(bm, vgroup_index, params, weight_mul, protect_set, preserve_uv, orig_positions):
    deform_layer = bm.verts.layers.deform.active
    if deform_layer is None:
        return
    angle_rad = math.radians(params.waves_direction)
    wave_dir = Vector((math.cos(angle_rad), math.sin(angle_rad), 0.0))
    amplitude = params.waves_amplitude * weight_mul
    wavelength = max(params.waves_wavelength, 0.001)
    phase = params.waves_phase * 2 * math.pi
    randomness = params.waves_randomness

    for v in bm.verts:
        if v.index in protect_set:
            continue
        w = v[deform_layer].get(vgroup_index, 0.0)
        if w <= 0.0:
            continue
        proj = v.co.dot(wave_dir)
        rand_offset = (random.random() - 0.5) * randomness * amplitude
        disp = math.sin((2 * math.pi * proj / wavelength) + phase) * amplitude * w + rand_offset * w
        v.co += v.normal * disp

    if preserve_uv:
        reproject_box_uv(bm, orig_positions)


def apply_folds(bm, vgroup_index, params, weight_mul, protect_set, preserve_uv, orig_positions):
    deform_layer = bm.verts.layers.deform.active
    if deform_layer is None:
        return
    angle_rad = math.radians(params.folds_direction)
    fold_dir = Vector((math.cos(angle_rad), math.sin(angle_rad), 0.0))
    intensity = params.folds_intensity * weight_mul
    length = max(params.folds_length, 0.001)
    freq = params.folds_frequency
    rand = params.folds_randomness
    profile = params.folds_profile

    for v in bm.verts:
        if v.index in protect_set:
            continue
        w = v[deform_layer].get(vgroup_index, 0.0)
        if w <= 0.0:
            continue
        proj = v.co.dot(fold_dir)
        t = (2 * math.pi * proj * freq / length)
        rand_val = (random.random() - 0.5) * rand
        if profile == 'V':
            shape = abs(math.sin(t)) * intensity
        elif profile == 'U':
            raw = math.sin(t)
            shape = (raw * raw) * intensity
        else:  # ASYM
            raw = math.sin(t)
            shape = raw * intensity if raw > 0 else raw * intensity * 0.3
        shape *= w
        shape += rand_val * intensity * w
        v.co += v.normal * (-shape)

    if preserve_uv:
        reproject_box_uv(bm, orig_positions)


def apply_bumps(bm, vgroup_index, params, weight_mul, protect_set, preserve_uv, orig_positions):
    deform_layer = bm.verts.layers.deform.active
    if deform_layer is None:
        return
    intensity = params.bumps_intensity * weight_mul
    freq = params.bumps_frequency
    randomness = params.bumps_randomness
    mode = params.bumps_direction
    bump_form = params.bumps_form

    for v in bm.verts:
        if v.index in protect_set:
            continue
        w = v[deform_layer].get(vgroup_index, 0.0)
        if w <= 0.0:
            continue
        p = v.co * freq + Vector((randomness * 3.7, randomness * 2.1, 0.0))
        n_val = mnoise.noise(p)
        if bump_form == 'FLAT':
            n_val = max(0.0, n_val)
        elif bump_form == 'SHARP':
            n_val = abs(n_val) ** 0.5 * (1 if n_val > 0 else -1)
        disp = n_val * intensity * w
        if mode == 'NORMAL':
            v.co += v.normal * disp
        else:  # TANGENT
            tangent = v.normal.cross(Vector((0, 0, 1))).normalized()
            v.co += tangent * disp

    if preserve_uv:
        reproject_box_uv(bm, orig_positions)


def apply_crease(bm, vgroup_index, params, weight_mul, protect_set, preserve_uv, orig_positions):
    deform_layer = bm.verts.layers.deform.active
    if deform_layer is None:
        return
    depth = params.crease_depth * weight_mul
    width = max(params.crease_width, 0.001)
    sharpness = params.crease_sharpness

    # Find centroid of zone
    zone_verts = [(v, v[deform_layer].get(vgroup_index, 0.0)) for v in bm.verts if v[deform_layer].get(vgroup_index, 0.0) > 0]
    if not zone_verts:
        return
    centroid = sum((v.co for v, _ in zone_verts), Vector()) / len(zone_verts)

    for v, w in zone_verts:
        if v.index in protect_set:
            continue
        dist = (v.co - centroid).length
        if dist > width:
            continue
        t = dist / width
        if sharpness == 'SHARP':
            falloff_w = (1.0 - t) ** 2
        else:
            falloff_w = 1.0 - t
        disp = -v.normal * depth * falloff_w * w
        v.co += disp

    if preserve_uv:
        reproject_box_uv(bm, orig_positions)


def apply_puff(bm, vgroup_index, params, weight_mul, protect_set, preserve_uv, orig_positions):
    deform_layer = bm.verts.layers.deform.active
    if deform_layer is None:
        return
    height = params.puff_height * weight_mul
    profile = params.puff_profile
    asym = params.puff_asymmetry

    zone_verts = [(v, v[deform_layer].get(vgroup_index, 0.0)) for v in bm.verts if v[deform_layer].get(vgroup_index, 0.0) > 0]
    if not zone_verts:
        return
    centroid = sum((v.co for v, _ in zone_verts), Vector()) / len(zone_verts)
    max_dist = max((v.co - centroid).length for v, _ in zone_verts) or 1.0

    for v, w in zone_verts:
        if v.index in protect_set:
            continue
        delta = v.co - centroid
        dist = delta.length / max_dist
        # Apply asymmetry
        if asym == 'X':
            dist *= 1.0 + 0.5 * (delta.x / (max_dist + 0.0001))
        elif asym == 'Y':
            dist *= 1.0 + 0.5 * (delta.y / (max_dist + 0.0001))
        dist = min(dist, 1.0)
        if profile == 'SPHERE':
            fw = 1.0 - dist * dist
        elif profile == 'FLAT':
            fw = max(0.0, 1.0 - dist * 3)
        else:  # OVAL
            fw = math.cos(dist * math.pi / 2)
        fw = max(fw, 0.0)
        v.co += v.normal * height * fw * w

    if preserve_uv:
        reproject_box_uv(bm, orig_positions)


def apply_smooth(bm, vgroup_index, params, weight_mul, protect_set):
    deform_layer = bm.verts.layers.deform.active
    if deform_layer is None:
        return
    strength = params.smooth_strength * weight_mul
    iterations = params.smooth_iterations
    preserve_boundary = params.smooth_preserve_boundary

    for _ in range(iterations):
        new_positions = {}
        for v in bm.verts:
            w = v[deform_layer].get(vgroup_index, 0.0)
            if w <= 0.0:
                continue
            if v.index in protect_set:
                continue
            if preserve_boundary:
                # Skip vertices on boundary edges
                on_boundary = any(not e.is_manifold for e in v.link_edges)
                if on_boundary:
                    continue
            neighbors = [e.other_vert(v).co for e in v.link_edges]
            if not neighbors:
                continue
            avg = sum(neighbors, Vector()) / len(neighbors)
            new_positions[v.index] = v.co.lerp(avg, strength * w)
        for v in bm.verts:
            if v.index in new_positions:
                v.co = new_positions[v.index]


def apply_noise(bm, vgroup_index, params, weight_mul, protect_set, preserve_uv, orig_positions):
    deform_layer = bm.verts.layers.deform.active
    if deform_layer is None:
        return
    intensity = params.noise_intensity * weight_mul
    scale = params.noise_scale
    detail = params.noise_detail
    roughness = params.noise_roughness
    seed = params.noise_seed

    for v in bm.verts:
        if v.index in protect_set:
            continue
        w = v[deform_layer].get(vgroup_index, 0.0)
        if w <= 0.0:
            continue
        p = v.co * scale + Vector((seed * 0.1, seed * 0.17, seed * 0.31))
        try:
            n_val = mnoise.fractal(p, 1.0, roughness, detail, noise_basis='PERLIN_ORIGINAL')
        except Exception:
            n_val = mnoise.noise(p)
        v.co += v.normal * n_val * intensity * w

    if preserve_uv:
        reproject_box_uv(bm, orig_positions)


# ---------------------------------------------------------------------------
# Core: apply all operations for a zone
# ---------------------------------------------------------------------------

def apply_zone_operations(obj, zone_vgroup_name, zone_meta, protect_mesh, preserve_uv):
    """
    Apply all operations listed in zone_meta to obj.
    Uses bmesh. Returns True on success.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    vg = obj.vertex_groups.get(zone_vgroup_name)
    if vg is None:
        bm.free()
        return False
    vgroup_index = vg.index

    # Collect original positions for UV preservation
    orig_positions = {v.index: v.co.copy() for v in bm.verts}

    protect_set = get_protected_vert_indices(bm) if protect_mesh else set()

    operations = zone_meta.get("operations", [])
    # Global intensity factor for this zone (1.0 keeps legacy behavior).
    zone_master_weight = float(zone_meta.get("master_weight", 1.0))
    for op_data in operations:
        if not op_data.get("enabled", True):
            continue
        op_type = op_data.get("type", "")
        # Combine per-zone and per-operation slider into one final multiplier.
        weight_mul = zone_master_weight * op_data.get("weight", 0.5)

        # Build a simple namespace for params
        class Params:
            pass
        p = Params()
        for k, val in op_data.get("params", {}).items():
            setattr(p, k, val)

        try:
            if op_type == 'INFLATE':
                apply_inflate(bm, vgroup_index, p, weight_mul, protect_set, preserve_uv, orig_positions)
            elif op_type == 'WAVES':
                apply_waves(bm, vgroup_index, p, weight_mul, protect_set, preserve_uv, orig_positions)
            elif op_type == 'FOLDS':
                apply_folds(bm, vgroup_index, p, weight_mul, protect_set, preserve_uv, orig_positions)
            elif op_type == 'BUMPS':
                apply_bumps(bm, vgroup_index, p, weight_mul, protect_set, preserve_uv, orig_positions)
            elif op_type == 'CREASE':
                apply_crease(bm, vgroup_index, p, weight_mul, protect_set, preserve_uv, orig_positions)
            elif op_type == 'PUFF':
                apply_puff(bm, vgroup_index, p, weight_mul, protect_set, preserve_uv, orig_positions)
            elif op_type == 'SMOOTH':
                apply_smooth(bm, vgroup_index, p, weight_mul, protect_set)
            elif op_type == 'NOISE':
                apply_noise(bm, vgroup_index, p, weight_mul, protect_set, preserve_uv, orig_positions)
        except Exception as e:
            print(f"[SoftForm] Error applying {op_type}: {e}")
            traceback.print_exc()

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return True


# ---------------------------------------------------------------------------
# PropertyGroups
# ---------------------------------------------------------------------------

class SoftFormOperation(bpy.types.PropertyGroup):
    op_type: bpy.props.EnumProperty(
        name="Type",
        items=[
            ('INFLATE', 'Inflate/Flatten', 'Opblazen of afvlakken'),
            ('WAVES', 'Waves/Golven', 'Golvend oppervlak'),
            ('FOLDS', 'Folds/Vouwen', 'Vouwen en kreuken'),
            ('BUMPS', 'Bumps/Hobbels', 'Bobbels en textuur'),
            ('CREASE', 'Crease/Naad', 'Naad-inkeping'),
            ('PUFF', 'Puff/Kussen', 'Kussen opblazen'),
            ('SMOOTH', 'Smooth/Glad', 'Afvlakken'),
            ('NOISE', 'Noise Displace', 'Willekeurige verplaatsing'),
        ],
        default='INFLATE',
        update=lambda self, ctx: on_softform_param_changed(self, ctx)
    )
    enabled: bpy.props.BoolProperty(name="Aan", default=True, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    weight: bpy.props.FloatProperty(name="Subtiel ↔ Extreem", min=0.0, max=1.0, default=0.5, update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # --- INFLATE ---
    inflate_direction: bpy.props.FloatProperty(name="Richting", min=-1.0, max=1.0, default=1.0,
        description="Positief = inflate, negatief = flatten", update=lambda self, ctx: on_softform_param_changed(self, ctx))
    inflate_intensity: bpy.props.FloatProperty(name="Intensiteit", min=0.0, max=1.0, default=0.05, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    inflate_falloff: bpy.props.EnumProperty(name="Falloff",
        items=[('SHARP','Scherp',''),('SMOOTH','Zacht',''),('LINEAR','Lineair','')], default='SMOOTH', update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # --- WAVES ---
    waves_direction: bpy.props.FloatProperty(name="Richting (°)", min=0, max=360, default=0, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    waves_amplitude: bpy.props.FloatProperty(name="Amplitude", min=0.001, max=0.1, default=0.02, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    waves_wavelength: bpy.props.FloatProperty(name="Golflengte", min=0.01, max=1.0, default=0.2, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    waves_phase: bpy.props.FloatProperty(name="Fase", min=0.0, max=1.0, default=0.0, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    waves_randomness: bpy.props.FloatProperty(name="Willekeur", min=0.0, max=1.0, default=0.1, update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # --- FOLDS ---
    folds_direction: bpy.props.FloatProperty(name="Richting (°)", min=0, max=360, default=0, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    folds_intensity: bpy.props.FloatProperty(name="Intensiteit", min=0.001, max=0.15, default=0.03, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    folds_length: bpy.props.FloatProperty(name="Lengte", min=0.01, max=0.5, default=0.1, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    folds_frequency: bpy.props.IntProperty(name="Frequentie", min=1, max=20, default=5, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    folds_randomness: bpy.props.FloatProperty(name="Willekeur", min=0.0, max=1.0, default=0.2, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    folds_profile: bpy.props.EnumProperty(name="Profiel",
        items=[('V','V-vouw',''),('U','U-vouw',''),('ASYM','Asymmetrisch','')], default='V', update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # --- BUMPS ---
    bumps_intensity: bpy.props.FloatProperty(name="Intensiteit", min=0.001, max=0.1, default=0.02, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    bumps_frequency: bpy.props.FloatProperty(name="Frequentie", min=1, max=30, default=8, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    bumps_randomness: bpy.props.FloatProperty(name="Willekeur", min=0.0, max=1.0, default=0.5, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    bumps_direction: bpy.props.EnumProperty(name="Richting",
        items=[('NORMAL','Normaal',''),('TANGENT','Tangent','')], default='NORMAL', update=lambda self, ctx: on_softform_param_changed(self, ctx))
    bumps_form: bpy.props.EnumProperty(name="Vorm",
        items=[('ROUND','Rond',''),('FLAT','Plat',''),('SHARP','Scherp','')], default='ROUND', update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # --- CREASE ---
    crease_depth: bpy.props.FloatProperty(name="Diepte", min=0.001, max=0.05, default=0.01, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    crease_width: bpy.props.FloatProperty(name="Breedte", min=0.001, max=0.05, default=0.02, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    crease_sharpness: bpy.props.EnumProperty(name="Scherpte",
        items=[('SHARP','Scherp',''),('SOFT','Zacht','')], default='SOFT', update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # --- PUFF ---
    puff_height: bpy.props.FloatProperty(name="Hoogte", min=0.001, max=0.2, default=0.05, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    puff_profile: bpy.props.EnumProperty(name="Profiel",
        items=[('SPHERE','Bol',''),('FLAT','Plat',''),('OVAL','Ovaal','')], default='SPHERE', update=lambda self, ctx: on_softform_param_changed(self, ctx))
    puff_asymmetry: bpy.props.EnumProperty(name="Asymmetrie",
        items=[('NONE','Geen',''),('X','X-as',''),('Y','Y-as','')], default='NONE', update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # --- SMOOTH ---
    smooth_strength: bpy.props.FloatProperty(name="Sterkte", min=0.0, max=1.0, default=0.5, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    smooth_iterations: bpy.props.IntProperty(name="Iteraties", min=1, max=20, default=3, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    smooth_preserve_boundary: bpy.props.BoolProperty(name="Grens bewaren", default=True, update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # --- NOISE ---
    noise_intensity: bpy.props.FloatProperty(name="Intensiteit", min=0.001, max=0.1, default=0.02, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    noise_scale: bpy.props.FloatProperty(name="Schaal", min=0.1, max=10.0, default=2.0, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    noise_detail: bpy.props.FloatProperty(name="Detail (octaven)", min=1, max=8, default=4, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    noise_roughness: bpy.props.FloatProperty(name="Ruwheid", min=0.0, max=1.0, default=0.5, update=lambda self, ctx: on_softform_param_changed(self, ctx))
    noise_seed: bpy.props.IntProperty(name="Seed", default=0, update=lambda self, ctx: on_softform_param_changed(self, ctx))


class SoftFormZone(bpy.types.PropertyGroup):
    zone_id: bpy.props.StringProperty(name="Zone ID", default="")
    zone_name: bpy.props.StringProperty(name="Naam", default="Zone")
    # Legacy fallback for older saved scenes; new code uses object_vgroups_json.
    vgroup_name: bpy.props.StringProperty(name="Vertex Group")
    # JSON mapping: {object_name: vertex_group_name}
    object_vgroups_json: bpy.props.StringProperty(name="Object Groups", default="{}")
    color: bpy.props.FloatVectorProperty(name="Kleur", subtype='COLOR', size=4,
        min=0.0, max=1.0, default=(0.9, 0.2, 0.2, 1.0))
    # Master slider for all operations in this zone.
    master_weight: bpy.props.FloatProperty(
        name="Algemene intensiteit",
        min=0.0,
        max=1.0,
        default=1.0,
        update=lambda self, ctx: on_softform_param_changed(self, ctx)
    )
    operations: bpy.props.CollectionProperty(type=SoftFormOperation)
    active_op_index: bpy.props.IntProperty(default=0)
    show_ops: bpy.props.BoolProperty(name="Toon bewerkingen", default=True)


class SoftFormPreset(bpy.types.PropertyGroup):
    preset_name: bpy.props.StringProperty(name="Naam")
    data: bpy.props.StringProperty(name="Data (JSON)")


class SoftFormSceneProps(bpy.types.PropertyGroup):
    # Wizard step
    wizard_step: bpy.props.IntProperty(default=1, min=1, max=3)

    # Selection mode
    selection_mode: bpy.props.EnumProperty(
        name="Selectiemodus",
        items=[
            ('OBJECT', 'Object select', 'Selecteer objecten'),
            ('FACE_LINKED', 'Face select (linked)', 'Selecteer verbonden faces'),
            ('FACE_LASSO', 'Face select (lasso)', 'Lasso selectie'),
        ],
        default='OBJECT'
    )

    # Live preview
    live_preview: bpy.props.BoolProperty(name="Live preview", default=True,
        update=lambda self, ctx: on_live_preview_toggle(self, ctx))

    # Zone list
    zones: bpy.props.CollectionProperty(type=SoftFormZone)
    active_zone_index: bpy.props.IntProperty(default=0)
    show_zone_highlights: bpy.props.BoolProperty(name="Toon zone-highlights", default=True)

    # Brush settings for weight paint
    brush_radius: bpy.props.FloatProperty(name="Brush radius", min=0.01, max=1.0, default=0.2)
    brush_strength: bpy.props.FloatProperty(name="Brush sterkte", min=0.0, max=1.0, default=1.0)

    # Protect / UV options
    protect_edges: bpy.props.BoolProperty(name="Bescherm mesh-randen", default=True,
        description="Voorkom gaten op naden: bevriest open/seam-randen",
        update=lambda self, ctx: on_softform_param_changed(self, ctx))
    preserve_uv: bpy.props.BoolProperty(name="Box UV schaal behouden", default=True,
        description="UV's herprojecteren na bewerking zodat textuur niet mee rekt",
        update=lambda self, ctx: on_softform_param_changed(self, ctx))

    # Preset
    preset_name_input: bpy.props.StringProperty(name="Preset naam", default="")
    presets: bpy.props.CollectionProperty(type=SoftFormPreset)
    active_preset_index: bpy.props.IntProperty(default=0)


def on_live_preview_toggle(self, context):
    """Called when live_preview is toggled."""
    sf = context.scene.softform
    if sf.live_preview:
        refresh_preview(context)
    else:
        # Restore all objects
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                restore_original_positions(obj)


def on_softform_param_changed(self, context):
    """
    Centrale update-callback voor parameters die de preview beïnvloeden.
    Bij live preview direct opnieuw berekenen, zonder extra knop.
    """
    if context is None or context.scene is None or not hasattr(context.scene, "softform"):
        return
    sf = context.scene.softform
    if sf.live_preview:
        refresh_preview(context)


# ---------------------------------------------------------------------------
# Preview system
# ---------------------------------------------------------------------------

def refresh_preview(context):
    """Apply current zone operations as a preview on all affected mesh objects."""
    sf = context.scene.softform
    if not sf.live_preview:
        return
    if sf.active_zone_index >= len(sf.zones):
        return

    zone = sf.zones[sf.active_zone_index]
    # Build zone meta from PropertyGroup once and reuse for all mapped objects.
    meta = zone_pg_to_dict(zone)

    for obj, vg_name in iter_zone_object_groups(context, zone):
        # Store originals if not stored yet
        if SF_ORIG_KEY not in obj:
            store_original_positions(obj)
        else:
            restore_original_positions(obj)

        try:
            apply_zone_operations(obj, vg_name, meta, sf.protect_edges, sf.preserve_uv)
        except Exception as e:
            print(f"[SoftForm] preview error: {e}")
            traceback.print_exc()


def zone_pg_to_dict(zone):
    """Convert SoftFormZone PropertyGroup to dict for apply_zone_operations."""
    ops = []
    for op in zone.operations:
        params = op_pg_to_params_dict(op)
        ops.append({
            "type": op.op_type,
            "enabled": op.enabled,
            "weight": op.weight,
            "params": params,
        })
    return {
        "master_weight": zone.master_weight,
        "operations": ops,
    }


def op_pg_to_params_dict(op):
    """Extract all parameter values from a SoftFormOperation into a dict."""
    ot = op.op_type
    d = {}
    if ot == 'INFLATE':
        d = dict(inflate_direction=op.inflate_direction, inflate_intensity=op.inflate_intensity,
                 inflate_falloff=op.inflate_falloff)
    elif ot == 'WAVES':
        d = dict(waves_direction=op.waves_direction, waves_amplitude=op.waves_amplitude,
                 waves_wavelength=op.waves_wavelength, waves_phase=op.waves_phase,
                 waves_randomness=op.waves_randomness)
    elif ot == 'FOLDS':
        d = dict(folds_direction=op.folds_direction, folds_intensity=op.folds_intensity,
                 folds_length=op.folds_length, folds_frequency=op.folds_frequency,
                 folds_randomness=op.folds_randomness, folds_profile=op.folds_profile)
    elif ot == 'BUMPS':
        d = dict(bumps_intensity=op.bumps_intensity, bumps_frequency=op.bumps_frequency,
                 bumps_randomness=op.bumps_randomness, bumps_direction=op.bumps_direction,
                 bumps_form=op.bumps_form)
    elif ot == 'CREASE':
        d = dict(crease_depth=op.crease_depth, crease_width=op.crease_width,
                 crease_sharpness=op.crease_sharpness)
    elif ot == 'PUFF':
        d = dict(puff_height=op.puff_height, puff_profile=op.puff_profile,
                 puff_asymmetry=op.puff_asymmetry)
    elif ot == 'SMOOTH':
        d = dict(smooth_strength=op.smooth_strength, smooth_iterations=op.smooth_iterations,
                 smooth_preserve_boundary=op.smooth_preserve_boundary)
    elif ot == 'NOISE':
        d = dict(noise_intensity=op.noise_intensity, noise_scale=op.noise_scale,
                 noise_detail=op.noise_detail, noise_roughness=op.noise_roughness,
                 noise_seed=op.noise_seed)
    return d


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class SF_OT_SetSelectionMode(bpy.types.Operator):
    """Set the SoftForm selection mode."""
    bl_idname = "softform.set_selection_mode"
    bl_label = "Stel selectiemodus in"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.StringProperty()

    def execute(self, context):
        sf = context.scene.softform
        sf.selection_mode = self.mode
        sf.wizard_step = 1

        if self.mode in ('FACE_LINKED', 'FACE_LASSO'):
            # Switch to edit mode for selected objects
            selected = [o for o in context.selected_objects if o.type == 'MESH']
            if not selected:
                self.report({'WARNING'}, "Geen mesh-objecten geselecteerd")
                return {'CANCELLED'}
            context.view_layer.objects.active = selected[0]
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='FACE')
            if self.mode == 'FACE_LASSO':
                # Activate lasso select tool
                bpy.ops.wm.tool_set_by_id(name="builtin.select_lasso")
        return {'FINISHED'}


class SF_OT_SelectLinkedFaces(bpy.types.Operator):
    """Select all connected faces from the current face selection (same as L)."""
    bl_idname = "softform.select_linked_faces"
    bl_label = "Selecteer verbonden faces (L)"
    bl_description = "Selecteert verbonden faces, hetzelfde als sneltoets L in Edit Mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object

        # Safety checks so the button gives a clear warning instead of a hard failure.
        if obj is None or obj.type != 'MESH':
            self.report({'WARNING'}, "Actief object moet een mesh zijn")
            return {'CANCELLED'}
        if obj.mode != 'EDIT':
            self.report({'WARNING'}, "Zet het mesh-object eerst in Edit Mode")
            return {'CANCELLED'}

        # Ensure face select mode and then run Blender's linked-face selection.
        bpy.ops.mesh.select_mode(type='FACE')
        bpy.ops.mesh.select_linked(delimit=set())
        return {'FINISHED'}


class SF_OT_CreateZoneNoInpaint(bpy.types.Operator):
    """Create a zone from the current selection without weight painting."""
    bl_idname = "softform.create_zone_no_inpaint"
    bl_label = "Nieuwe zone (zonder inpaint)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sf = context.scene.softform
        mode = sf.selection_mode

        if mode == 'OBJECT':
            selected = [o for o in context.selected_objects if o.type == 'MESH']
            if not selected:
                self.report({'ERROR'}, "Geen mesh-objecten geselecteerd")
                return {'CANCELLED'}
            self._create_full_object_zone(context, sf, selected)
        else:
            # Face select mode with multi-object edit support.
            selected_by_object = gather_multi_object_face_selection(context)
            if not selected_by_object:
                self.report({'WARNING'}, "Geen geselecteerde faces gevonden in Edit Mode")
                return {'CANCELLED'}
            self._create_face_select_zone(context, sf, selected_by_object)

        sf.wizard_step = 2
        return {'FINISHED'}

    def _create_full_object_zone(self, context, sf, objects):
        """Create one logical zone over full-mesh selections on multiple objects."""
        logical_zone_id = get_next_logical_zone_id(sf)
        object_map = {}
        color_index = len(sf.zones) + 1

        for obj in objects:
            idx = get_next_zone_index(obj)
            vgname = f"{SF_PREFIX}{idx}"
            vg = obj.vertex_groups.new(name=vgname)
            # Assign all vertices with weight 1.0 for this object.
            all_indices = [v.index for v in obj.data.vertices]
            vg.add(all_indices, 1.0, 'REPLACE')
            object_map[obj.name] = vgname

        zone_name = f"{SF_PREFIX}{logical_zone_id}"
        fallback_vg = next(iter(object_map.values()), "")
        color = get_zone_color(color_index)
        create_zone_entry(sf, logical_zone_id, zone_name, color, object_map, fallback_vg)
        print(f"[SoftForm] Created logical full-object zone: {zone_name} -> {object_map}")

    def _create_face_select_zone(self, context, sf, selected_by_object):
        """Create one logical zone from selected faces across multiple edit-mode objects."""
        # Leave edit mode once to safely create vertex groups on all involved objects.
        bpy.ops.object.mode_set(mode='OBJECT')

        logical_zone_id = get_next_logical_zone_id(sf)
        object_map = {}
        color_index = len(sf.zones) + 1

        for obj, selected_verts in selected_by_object.items():
            idx = get_next_zone_index(obj)
            vgname = f"{SF_PREFIX}{idx}"
            vg = obj.vertex_groups.new(name=vgname)
            vg.add(selected_verts, 1.0, 'REPLACE')
            object_map[obj.name] = vgname

        zone_name = f"{SF_PREFIX}{logical_zone_id}"
        fallback_vg = next(iter(object_map.values()), "")
        color = get_zone_color(color_index)
        create_zone_entry(sf, logical_zone_id, zone_name, color, object_map, fallback_vg)
        print(f"[SoftForm] Created logical face-select zone: {zone_name} -> {object_map}")


class SF_OT_CreateZoneWithInpaint(bpy.types.Operator):
    """Create a zone and enter weight paint mode for brush input."""
    bl_idname = "softform.create_zone_with_inpaint"
    bl_label = "Nieuwe zone (met inpaint)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sf = context.scene.softform
        if sf.selection_mode != 'OBJECT':
            self.report({'ERROR'}, "Inpaint alleen beschikbaar in object-selectiemodus")
            return {'CANCELLED'}

        selected = [o for o in context.selected_objects if o.type == 'MESH']
        if not selected:
            self.report({'ERROR'}, "Geen mesh-objecten geselecteerd")
            return {'CANCELLED'}

        obj = selected[0]
        context.view_layer.objects.active = obj

        idx = get_next_zone_index(obj)
        vgname = f"{SF_PREFIX}{idx}"
        vg = obj.vertex_groups.new(name=vgname)
        # Start with zero weights — user will paint
        vg.add([v.index for v in obj.data.vertices], 0.0, 'REPLACE')

        logical_zone_id = get_next_logical_zone_id(sf)
        object_map = {obj.name: vgname}
        create_zone_entry(
            sf,
            logical_zone_id,
            f"{SF_PREFIX}{logical_zone_id}",
            get_zone_color(len(sf.zones) + 1),
            object_map,
            vgname,
        )

        # Enter weight paint mode
        obj.vertex_groups.active = vg
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

        # Configure brush
        brush = context.tool_settings.weight_paint.brush
        if brush:
            brush.weight = sf.brush_strength
            # Blender renamed this API in newer versions (unprojected_radius -> unprojected_size).
            if hasattr(brush, "unprojected_size"):
                brush.unprojected_size = sf.brush_radius
            elif hasattr(brush, "unprojected_radius"):
                brush.unprojected_radius = sf.brush_radius

        sf.wizard_step = 2
        print(f"[SoftForm] Created inpaint zone: {vgname} on {obj.name}, entering weight paint mode")
        return {'FINISHED'}


class SF_OT_FinishPainting(bpy.types.Operator):
    """Finish weight painting and return to object mode."""
    bl_idname = "softform.finish_painting"
    bl_label = "Klaar met schilderen"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')
        context.scene.softform.wizard_step = 2
        return {'FINISHED'}


class SF_OT_SelectZone(bpy.types.Operator):
    """Select a zone by index."""
    bl_idname = "softform.select_zone"
    bl_label = "Selecteer zone"

    index: bpy.props.IntProperty()

    def execute(self, context):
        sf = context.scene.softform
        sf.active_zone_index = self.index
        if sf.live_preview:
            refresh_preview(context)
        return {'FINISHED'}


class SF_OT_DeleteZone(bpy.types.Operator):
    """Delete a zone and its vertex group."""
    bl_idname = "softform.delete_zone"
    bl_label = "Verwijder zone"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty()

    def execute(self, context):
        sf = context.scene.softform
        if self.index >= len(sf.zones):
            return {'CANCELLED'}
        zone = sf.zones[self.index]
        # Remove all vertex groups linked to this logical zone.
        for obj, vgname in iter_zone_object_groups(context, zone):
            vg = obj.vertex_groups.get(vgname)
            if vg:
                obj.vertex_groups.remove(vg)
        sf.zones.remove(self.index)
        sf.active_zone_index = max(0, sf.active_zone_index - 1)
        return {'FINISHED'}


class SF_OT_AddOperation(bpy.types.Operator):
    """Add a new operation to the active zone."""
    bl_idname = "softform.add_operation"
    bl_label = "+ Bewerking toevoegen"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sf = context.scene.softform
        if sf.active_zone_index >= len(sf.zones):
            self.report({'ERROR'}, "Geen actieve zone")
            return {'CANCELLED'}
        zone = sf.zones[sf.active_zone_index]
        op = zone.operations.add()
        op.op_type = 'INFLATE'
        op.enabled = True
        op.weight = 0.5
        zone.active_op_index = len(zone.operations) - 1
        if sf.live_preview:
            refresh_preview(context)
        return {'FINISHED'}


class SF_OT_RemoveOperation(bpy.types.Operator):
    """Remove an operation from the active zone."""
    bl_idname = "softform.remove_operation"
    bl_label = "Verwijder bewerking"
    bl_options = {'REGISTER', 'UNDO'}

    op_index: bpy.props.IntProperty()

    def execute(self, context):
        sf = context.scene.softform
        if sf.active_zone_index >= len(sf.zones):
            return {'CANCELLED'}
        zone = sf.zones[sf.active_zone_index]
        if self.op_index < len(zone.operations):
            zone.operations.remove(self.op_index)
        if sf.live_preview:
            refresh_preview(context)
        return {'FINISHED'}


class SF_OT_MoveOperationUp(bpy.types.Operator):
    """Move operation up in list."""
    bl_idname = "softform.move_op_up"
    bl_label = "Omhoog"
    bl_options = {'REGISTER', 'UNDO'}

    op_index: bpy.props.IntProperty()

    def execute(self, context):
        sf = context.scene.softform
        zone = sf.zones[sf.active_zone_index]
        if self.op_index > 0:
            zone.operations.move(self.op_index, self.op_index - 1)
        if sf.live_preview:
            refresh_preview(context)
        return {'FINISHED'}


class SF_OT_MoveOperationDown(bpy.types.Operator):
    """Move operation down in list."""
    bl_idname = "softform.move_op_down"
    bl_label = "Omlaag"
    bl_options = {'REGISTER', 'UNDO'}

    op_index: bpy.props.IntProperty()

    def execute(self, context):
        sf = context.scene.softform
        zone = sf.zones[sf.active_zone_index]
        if self.op_index < len(zone.operations) - 1:
            zone.operations.move(self.op_index, self.op_index + 1)
        if sf.live_preview:
            refresh_preview(context)
        return {'FINISHED'}


class SF_OT_RefreshPreview(bpy.types.Operator):
    """Refresh live preview."""
    bl_idname = "softform.refresh_preview"
    bl_label = "Preview bijwerken"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        refresh_preview(context)
        return {'FINISHED'}


class SF_OT_ApplyConfirm(bpy.types.Operator):
    """Apply operations with confirmation dialog."""
    bl_idname = "softform.apply_confirm"
    bl_label = "Bevestig toepassing"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        sf = context.scene.softform
        if sf.active_zone_index >= len(sf.zones):
            self.report({'ERROR'}, "Geen actieve zone")
            return {'CANCELLED'}
        zone = sf.zones[sf.active_zone_index]
        meta = zone_pg_to_dict(zone)

        success = False
        for obj, vg_name in iter_zone_object_groups(context, zone):
            # Restore to clean state first
            restore_original_positions(obj)
            try:
                result = apply_zone_operations(obj, vg_name, meta, sf.protect_edges, sf.preserve_uv)
                if result:
                    success = True
                    # Save zone config with logical zone-id and per-object vg reference.
                    obj_meta = load_zone_meta(obj)
                    obj_meta[zone.zone_id or zone.zone_name] = {
                        "zone_name": zone.zone_name,
                        "vgroup_name": vg_name,
                        "master_weight": meta.get("master_weight", 1.0),
                        "operations": meta.get("operations", []),
                    }
                    save_zone_meta(obj, obj_meta)
                    # Clear stored originals (operations are now baked)
                    if SF_ORIG_KEY in obj:
                        del obj[SF_ORIG_KEY]
            except Exception as e:
                self.report({'ERROR'}, f"Fout bij toepassen: {e}")
                print(f"[SoftForm] apply error: {e}")
                traceback.print_exc()
                return {'CANCELLED'}

        if success:
            self.report({'INFO'}, f"Zone '{zone.zone_name}' toegepast")
        else:
            self.report({'WARNING'}, "Geen objecten gevonden voor deze zone")
        return {'FINISHED'}


class SF_OT_ResetAll(bpy.types.Operator):
    """Reset all SoftForm data."""
    bl_idname = "softform.reset_all"
    bl_label = "Alles terugzetten"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        sf = context.scene.softform
        # Restore mesh positions
        for obj in context.scene.objects:
            if obj.type != 'MESH':
                continue
            restore_original_positions(obj)
            # Remove SF_ vertex groups
            to_remove = [vg for vg in obj.vertex_groups if vg.name.startswith(SF_PREFIX)]
            for vg in to_remove:
                obj.vertex_groups.remove(vg)
            # Clear custom properties
            if SF_META_KEY in obj:
                del obj[SF_META_KEY]
            if SF_ORIG_KEY in obj:
                del obj[SF_ORIG_KEY]
        # Clear zones
        sf.zones.clear()
        sf.active_zone_index = 0
        sf.wizard_step = 1
        self.report({'INFO'}, "Alles teruggesteld")
        return {'FINISHED'}


class SF_OT_SavePreset(bpy.types.Operator):
    """Save current zone operations as a named preset."""
    bl_idname = "softform.save_preset"
    bl_label = "Preset opslaan"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sf = context.scene.softform
        name = sf.preset_name_input.strip()
        if not name:
            self.report({'ERROR'}, "Voer een preset-naam in")
            return {'CANCELLED'}
        if sf.active_zone_index >= len(sf.zones):
            self.report({'ERROR'}, "Geen actieve zone")
            return {'CANCELLED'}
        zone = sf.zones[sf.active_zone_index]
        meta = zone_pg_to_dict(zone)
        data_str = json.dumps(meta)
        # Overwrite if name exists
        for preset in sf.presets:
            if preset.preset_name == name:
                preset.data = data_str
                self.report({'INFO'}, f"Preset '{name}' bijgewerkt")
                return {'FINISHED'}
        preset = sf.presets.add()
        preset.preset_name = name
        preset.data = data_str
        self.report({'INFO'}, f"Preset '{name}' opgeslagen")
        return {'FINISHED'}


class SF_OT_LoadPreset(bpy.types.Operator):
    """Load a preset into the active zone."""
    bl_idname = "softform.load_preset"
    bl_label = "Preset inladen"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sf = context.scene.softform
        if sf.active_preset_index >= len(sf.presets):
            self.report({'ERROR'}, "Geen preset geselecteerd")
            return {'CANCELLED'}
        if sf.active_zone_index >= len(sf.zones):
            self.report({'ERROR'}, "Geen actieve zone")
            return {'CANCELLED'}
        preset = sf.presets[sf.active_preset_index]
        try:
            meta = json.loads(preset.data)
        except Exception:
            self.report({'ERROR'}, "Ongeldige preset data")
            return {'CANCELLED'}
        zone = sf.zones[sf.active_zone_index]
        # Preset can now control one global intensity slider for the whole zone.
        zone.master_weight = meta.get("master_weight", 1.0)
        zone.operations.clear()
        for op_data in meta.get("operations", []):
            op = zone.operations.add()
            op.op_type = op_data.get("type", "INFLATE")
            op.enabled = op_data.get("enabled", True)
            op.weight = op_data.get("weight", 0.5)
            params = op_data.get("params", {})
            for k, v in params.items():
                try:
                    setattr(op, k, v)
                except Exception:
                    pass
        if sf.live_preview:
            refresh_preview(context)
        self.report({'INFO'}, f"Preset '{preset.preset_name}' ingeladen")
        return {'FINISHED'}


class SF_OT_DeletePreset(bpy.types.Operator):
    """Delete the selected preset."""
    bl_idname = "softform.delete_preset"
    bl_label = "Preset verwijderen"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sf = context.scene.softform
        if sf.active_preset_index >= len(sf.presets):
            return {'CANCELLED'}
        name = sf.presets[sf.active_preset_index].preset_name
        sf.presets.remove(sf.active_preset_index)
        sf.active_preset_index = max(0, sf.active_preset_index - 1)
        self.report({'INFO'}, f"Preset '{name}' verwijderd")
        return {'FINISHED'}


class SF_OT_GoToStep(bpy.types.Operator):
    """Navigate wizard to a specific step."""
    bl_idname = "softform.go_to_step"
    bl_label = "Ga naar stap"

    step: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.softform.wizard_step = self.step
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI Panel
# ---------------------------------------------------------------------------

def draw_op_params(layout, op):
    """Draw parameters for a single operation."""
    ot = op.op_type
    box = layout.box()
    if ot == 'INFLATE':
        box.prop(op, "inflate_direction", slider=True)
        box.prop(op, "inflate_intensity", slider=True)
        box.prop(op, "inflate_falloff")
    elif ot == 'WAVES':
        box.prop(op, "waves_direction", slider=True)
        box.prop(op, "waves_amplitude", slider=True)
        box.prop(op, "waves_wavelength", slider=True)
        box.prop(op, "waves_phase", slider=True)
        box.prop(op, "waves_randomness", slider=True)
    elif ot == 'FOLDS':
        box.prop(op, "folds_direction", slider=True)
        box.prop(op, "folds_intensity", slider=True)
        box.prop(op, "folds_length", slider=True)
        box.prop(op, "folds_frequency")
        box.prop(op, "folds_randomness", slider=True)
        box.prop(op, "folds_profile")
    elif ot == 'BUMPS':
        box.prop(op, "bumps_intensity", slider=True)
        box.prop(op, "bumps_frequency", slider=True)
        box.prop(op, "bumps_randomness", slider=True)
        box.prop(op, "bumps_direction")
        box.prop(op, "bumps_form")
    elif ot == 'CREASE':
        box.prop(op, "crease_depth", slider=True)
        box.prop(op, "crease_width", slider=True)
        box.prop(op, "crease_sharpness")
    elif ot == 'PUFF':
        box.prop(op, "puff_height", slider=True)
        box.prop(op, "puff_profile")
        box.prop(op, "puff_asymmetry")
    elif ot == 'SMOOTH':
        box.prop(op, "smooth_strength", slider=True)
        box.prop(op, "smooth_iterations")
        box.prop(op, "smooth_preserve_boundary")
    elif ot == 'NOISE':
        box.prop(op, "noise_intensity", slider=True)
        box.prop(op, "noise_scale", slider=True)
        box.prop(op, "noise_detail", slider=True)
        box.prop(op, "noise_roughness", slider=True)
        box.prop(op, "noise_seed")


class SF_PT_MainPanel(bpy.types.Panel):
    bl_label = "SoftForm"
    bl_idname = "SF_PT_MainPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SoftForm"

    def draw(self, context):
        layout = self.layout
        sf = context.scene.softform

        # --- HEADER ---
        header_box = layout.box()
        row = header_box.row(align=True)
        icon = 'HIDE_OFF' if sf.live_preview else 'HIDE_ON'
        row.prop(sf, "live_preview", toggle=True, icon=icon, text="Live preview")

        # Progress indicator
        row2 = header_box.row(align=True)
        for s in range(1, 4):
            label = ["1: Selectie", "2: Zones", "3: Effecten"][s - 1]
            op = row2.operator("softform.go_to_step", text=label,
                               depress=(sf.wizard_step == s))
            op.step = s

        layout.separator()

        # --- STEP 1 ---
        if sf.wizard_step == 1:
            self._draw_step1(layout, context, sf)

        # --- STEP 2 ---
        elif sf.wizard_step == 2:
            self._draw_step2(layout, context, sf)

        # --- STEP 3 ---
        elif sf.wizard_step == 3:
            self._draw_step3(layout, context, sf)

        # --- FOOTER (always visible) ---
        layout.separator()
        self._draw_footer(layout, context, sf)

    # -----------------------------------------------------------------------
    def _draw_step1(self, layout, context, sf):
        box = layout.box()
        box.label(text="Stap 1: Selecteer gebied", icon='CURSOR')

        col = box.column(align=True)
        col.label(text="Selectiemodus:")
        row = col.row(align=True)
        op = row.operator("softform.set_selection_mode", text="Object",
                          depress=(sf.selection_mode == 'OBJECT'))
        op.mode = 'OBJECT'
        op = row.operator("softform.set_selection_mode", text="Face (linked)",
                          depress=(sf.selection_mode == 'FACE_LINKED'))
        op.mode = 'FACE_LINKED'
        op = row.operator("softform.set_selection_mode", text="Face (lasso)",
                          depress=(sf.selection_mode == 'FACE_LASSO'))
        op.mode = 'FACE_LASSO'

        box.separator()
        col2 = box.column(align=True)
        col2.label(text="Zone aanmaken:")

        if sf.selection_mode == 'OBJECT':
            col2.operator("softform.create_zone_with_inpaint", icon='BRUSH_DATA')
            col2.operator("softform.create_zone_no_inpaint", icon='MESH_DATA')
        else:
            if sf.selection_mode in {'FACE_LINKED', 'FACE_LASSO'}:
                # UI shortcut for users who do not know the keyboard shortcut (L).
                col2.operator("softform.select_linked_faces", icon='SELECT_EXTEND')
            col2.operator("softform.create_zone_no_inpaint", icon='MESH_DATA')
            no_inpaint_row = col2.row()
            no_inpaint_row.enabled = False
            no_inpaint_row.operator("softform.create_zone_with_inpaint",
                                    icon='BRUSH_DATA', text="Inpaint (n.v.t. bij face-select)")

        # Brush controls — only visible when in weight paint mode
        if context.active_object and context.active_object.mode == 'WEIGHT_PAINT':
            box.separator()
            box.label(text="Schilder-instellingen:", icon='BRUSH_DATA')
            box.prop(sf, "brush_radius", slider=True)
            box.prop(sf, "brush_strength", slider=True)
            box.operator("softform.finish_painting", icon='CHECKMARK')

    # -----------------------------------------------------------------------
    def _draw_step2(self, layout, context, sf):
        box = layout.box()
        box.label(text="Stap 2: Zones beheren", icon='GROUP_VERTEX')

        box.prop(sf, "show_zone_highlights", icon='HIDE_OFF')

        if not sf.zones:
            box.label(text="Geen zones aangemaakt", icon='INFO')
            box.operator("softform.create_zone_no_inpaint", text="+ Zone aanmaken", icon='ADD')
            return

        for i, zone in enumerate(sf.zones):
            row = box.row(align=True)
            # Color square
            row.prop(zone, "color", text="")
            # Zone name — click to select
            sel_op = row.operator("softform.select_zone",
                                   text=zone.zone_name,
                                   depress=(sf.active_zone_index == i),
                                   emboss=True)
            sel_op.index = i
            # Delete button
            del_op = row.operator("softform.delete_zone", text="", icon='TRASH')
            del_op.index = i

        box.separator()
        row = box.row(align=True)
        row.operator("softform.create_zone_with_inpaint", text="+ Inpaint", icon='BRUSH_DATA')
        row.operator("softform.create_zone_no_inpaint", text="+ Zonder paint", icon='ADD')

        box.separator()
        box.operator("softform.go_to_step", text="Verder naar effecten →", icon='FORWARD').step = 3

    # -----------------------------------------------------------------------
    def _draw_step3(self, layout, context, sf):
        box = layout.box()
        box.label(text="Stap 3: Effecten kiezen", icon='MOD_DISPLACE')

        if not sf.zones:
            box.label(text="Maak eerst een zone aan", icon='INFO')
            box.operator("softform.go_to_step", text="← Terug naar stap 2").step = 2
            return

        if sf.active_zone_index >= len(sf.zones):
            sf.active_zone_index = 0

        zone = sf.zones[sf.active_zone_index]

        # Zone selector
        row = box.row(align=True)
        row.label(text="Zone:")
        for i, z in enumerate(sf.zones):
            op = row.operator("softform.select_zone", text=z.zone_name,
                              depress=(sf.active_zone_index == i))
            op.index = i

        box.separator()
        box.prop(zone, "master_weight", slider=True)
        box.separator()

        # Operations
        if not zone.operations:
            box.label(text="Geen bewerkingen toegevoegd", icon='INFO')

        for i, op in enumerate(zone.operations):
            op_box = box.box()
            row = op_box.row(align=True)
            row.prop(op, "enabled", text="", icon='CHECKMARK' if op.enabled else 'X')
            row.prop(op, "op_type", text="")
            # Move up/down
            up_op = row.operator("softform.move_op_up", text="", icon='TRIA_UP')
            up_op.op_index = i
            dn_op = row.operator("softform.move_op_down", text="", icon='TRIA_DOWN')
            dn_op.op_index = i
            # Remove
            rm_op = row.operator("softform.remove_operation", text="", icon='TRASH')
            rm_op.op_index = i

            if op.enabled:
                row2 = op_box.row()
                row2.prop(op, "weight", text="Subtiel ↔ Extreem", slider=True)
                draw_op_params(op_box, op)

        box.separator()
        box.operator("softform.add_operation", icon='ADD')

        # Options
        box.separator()
        opts = box.box()
        opts.label(text="Opties:", icon='SETTINGS')
        opts.prop(sf, "protect_edges", icon='LOCKED')
        opts.prop(sf, "preserve_uv", icon='UV')

        # Preset section
        box.separator()
        preset_box = box.box()
        preset_box.label(text="Presets", icon='PRESET')
        preset_box.prop(sf, "preset_name_input", text="Naam")
        row = preset_box.row(align=True)
        row.operator("softform.save_preset", text="Opslaan", icon='FILE_TICK')
        row.operator("softform.load_preset", text="Inladen", icon='IMPORT')
        row.operator("softform.delete_preset", text="", icon='TRASH')
        if sf.presets:
            preset_box.template_list("UI_UL_list", "sf_presets", sf, "presets",
                                     sf, "active_preset_index", rows=2)

    # -----------------------------------------------------------------------
    def _draw_footer(self, layout, context, sf):
        footer = layout.box()
        row = footer.row(align=True)
        row.scale_y = 1.3
        row.operator("softform.apply_confirm", text="✅ Toepassen", icon='CHECKMARK')
        row.operator("softform.reset_all", text="↩ Terugzetten", icon='LOOP_BACK')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

CLASSES = [
    SoftFormOperation,
    SoftFormZone,
    SoftFormPreset,
    SoftFormSceneProps,
    SF_OT_SetSelectionMode,
    SF_OT_SelectLinkedFaces,
    SF_OT_CreateZoneNoInpaint,
    SF_OT_CreateZoneWithInpaint,
    SF_OT_FinishPainting,
    SF_OT_SelectZone,
    SF_OT_DeleteZone,
    SF_OT_AddOperation,
    SF_OT_RemoveOperation,
    SF_OT_MoveOperationUp,
    SF_OT_MoveOperationDown,
    SF_OT_RefreshPreview,
    SF_OT_ApplyConfirm,
    SF_OT_ResetAll,
    SF_OT_SavePreset,
    SF_OT_LoadPreset,
    SF_OT_DeletePreset,
    SF_OT_GoToStep,
    SF_PT_MainPanel,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.softform = bpy.props.PointerProperty(type=SoftFormSceneProps)
    print("[SoftForm] Add-on geregistreerd")


def unregister():
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"[SoftForm] Fout bij deregistreren {cls}: {e}")
    if hasattr(bpy.types.Scene, "softform"):
        del bpy.types.Scene.softform
    print("[SoftForm] Add-on gederegistreerd")


if __name__ == "__main__":
    register()
