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
import math
import json
import random
import traceback
import logging
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.app.handlers import persistent
from mathutils import Vector, noise

logger = logging.getLogger("SoftForm")

PREVIEW_CACHE = {}


def safe_report(op, level, message):
    try:
        op.report(level, message)
    except Exception:
        print(f"SoftForm report fallback: {message}")


def next_zone_name(scene):
    i = 1
    existing = {z.name for z in scene.softform_zones}
    while f"Zone {i}" in existing:
        i += 1
    return f"Zone {i}"


def unique_zone_color(index):
    # Distinct-ish color wheel palette
    hue = (index * 0.61803398875) % 1.0
    r = abs(hue * 6 - 3) - 1
    g = 2 - abs(hue * 6 - 2)
    b = 2 - abs(hue * 6 - 4)
    return (
        max(0.0, min(r, 1.0)),
        max(0.0, min(g, 1.0)),
        max(0.0, min(b, 1.0)),
        1.0,
    )


def get_active_zone(scene):
    if not scene.softform_zones:
        return None
    idx = max(0, min(scene.softform_active_zone, len(scene.softform_zones) - 1))
    return scene.softform_zones[idx]


def get_object_for_zone(zone):
    return bpy.data.objects.get(zone.object_name)


def ensure_deform_layer(bm):
    layer = bm.verts.layers.deform.active
    if layer is None:
        layer = bm.verts.layers.deform.new()
    return layer


def collect_protected_vertices(mesh):
    protected = set()
    for e in mesh.edges:
        if not e.is_manifold or e.use_seam:
            protected.update(e.vertices)
    return protected


def vertex_weight(dvert, group_index):
    try:
        return dvert.get(group_index, 0.0)
    except Exception:
        return 0.0


def falloff_value(weight, mode):
    if mode == 'SHARP':
        return weight * weight
    if mode == 'LINEAR':
        return weight
    return weight * weight * (3.0 - 2.0 * weight)  # smoothstep


def perlin_at(co, scale, detail, roughness, seed):
    p = Vector(co) * scale + Vector((seed, seed, seed))
    return noise.fractal(p, 1.0, roughness, int(detail), noise_basis='PERLIN_ORIGINAL')


def preserve_box_uv(mesh, original_coords):
    if not mesh.uv_layers.active:
        return
    uv_layer = mesh.uv_layers.active.data
    for poly in mesh.polygons:
        n = poly.normal.normalized() if poly.normal.length > 0 else Vector((0, 0, 1))
        ax = max(range(3), key=lambda i: abs(n[i]))
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            co = original_coords.get(vi)
            if co is None:
                continue
            # Basic box projection from original coordinate => no UV stretch by deformation
            if ax == 0:
                uv = (co.y, co.z)
            elif ax == 1:
                uv = (co.x, co.z)
            else:
                uv = (co.x, co.y)
            uv_layer[li].uv = uv


def save_original_coords(obj):
    data = {}
    for v in obj.data.vertices:
        data[v.index] = v.co.copy()
    return data


def restore_original_coords(obj, coord_map):
    for v in obj.data.vertices:
        if v.index in coord_map:
            v.co = coord_map[v.index]
    obj.data.update()


def apply_inflate_flatten(mesh, group_index, op, global_weight, protect):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform = ensure_deform_layer(bm)
    protected = collect_protected_vertices(mesh) if protect else set()
    for v in bm.verts:
        if v.index in protected:
            continue
        w = vertex_weight(v[deform], group_index)
        if w <= 0:
            continue
        f = falloff_value(w, op.falloff)
        direction = 1.0 if op.inflate_dir >= 0 else -1.0
        strength = abs(op.inflate_dir) * op.inflate_intensity * op.weight * global_weight
        v.co += v.normal * direction * strength * f
    bm.to_mesh(mesh)
    bm.free()


def apply_waves(mesh, group_index, op, global_weight, protect):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform = ensure_deform_layer(bm)
    protected = collect_protected_vertices(mesh) if protect else set()
    ang = math.radians(op.waves_angle)
    axis = Vector((math.cos(ang), math.sin(ang), 0.0))
    if axis.length == 0:
        axis = Vector((1, 0, 0))
    axis.normalize()
    for v in bm.verts:
        if v.index in protected:
            continue
        w = vertex_weight(v[deform], group_index)
        if w <= 0:
            continue
        proj = v.co.dot(axis)
        phase = op.waves_phase * math.tau
        wl = max(op.waves_length, 0.0001)
        wave = math.sin((proj / wl) * math.tau + phase)
        rnd = (random.random() - 0.5) * 2.0 * op.waves_randomness
        spread_mul = {'NARROW': 0.6, 'NORMAL': 1.0, 'WIDE': 1.5}[op.waves_spread]
        amp = op.waves_amplitude * op.weight * global_weight
        v.co += v.normal * (wave + rnd) * amp * w * spread_mul
    bm.to_mesh(mesh)
    bm.free()


def apply_folds(mesh, group_index, op, global_weight, protect):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform = ensure_deform_layer(bm)
    protected = collect_protected_vertices(mesh) if protect else set()
    ang = math.radians(op.folds_angle)
    axis = Vector((math.cos(ang), math.sin(ang), 0.0))
    axis.normalize()
    for v in bm.verts:
        if v.index in protected:
            continue
        w = vertex_weight(v[deform], group_index)
        if w <= 0:
            continue
        t = v.co.dot(axis) * op.folds_frequency / max(op.folds_length, 0.001)
        s = math.sin(t * math.tau)
        if op.folds_profile == 'V':
            shape = -abs(s)
        elif op.folds_profile == 'ASYM':
            shape = s if s < 0 else -0.3 * s
        else:
            shape = -0.5 * (1 - math.cos(t * math.tau))
        rnd = (random.random() - 0.5) * 2.0 * op.folds_randomness
        influence = max(0.0, 1.0 - (1.0 - w) * op.folds_spread)
        v.co += v.normal * (shape + rnd) * op.folds_intensity * op.weight * global_weight * influence
    bm.to_mesh(mesh)
    bm.free()


def apply_bumps(mesh, group_index, op, global_weight, protect):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform = ensure_deform_layer(bm)
    protected = collect_protected_vertices(mesh) if protect else set()
    for v in bm.verts:
        if v.index in protected:
            continue
        w = vertex_weight(v[deform], group_index)
        if w <= 0:
            continue
        p = v.co * op.bumps_frequency
        n = noise.noise(p)
        grid = math.sin(p.x) * math.cos(p.y)
        val = (1.0 - op.bumps_randomness) * grid + op.bumps_randomness * n
        if op.bumps_shape == 'FLAT':
            val = max(-0.2, min(val, 0.2))
        elif op.bumps_shape == 'SHARP':
            val = math.copysign(abs(val) ** 0.4, val)
        disp = op.bumps_intensity * op.weight * global_weight * w * val
        if op.bumps_direction == 'NORMAL':
            v.co += v.normal * disp
        else:
            tangent = v.normal.cross(Vector((0, 0, 1)))
            if tangent.length < 1e-6:
                tangent = Vector((1, 0, 0))
            tangent.normalize()
            v.co += tangent * disp
    bm.to_mesh(mesh)
    bm.free()


def apply_crease(mesh, group_index, op, global_weight, protect):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform = ensure_deform_layer(bm)
    protected = collect_protected_vertices(mesh) if protect else set()
    center = Vector((0, 0, 0))
    count = 0
    for v in bm.verts:
        if v.index in protected:
            continue
        w = vertex_weight(v[deform], group_index)
        if w > 0:
            center += v.co * w
            count += 1
    if count == 0:
        bm.free()
        return
    center /= count
    angle = math.radians(op.crease_angle)
    axis = Vector((math.cos(angle), math.sin(angle), 0)).normalized()
    for v in bm.verts:
        if v.index in protected:
            continue
        w = vertex_weight(v[deform], group_index)
        if w <= 0:
            continue
        rel = v.co - center
        dist = abs(rel.cross(axis).length)
        if dist > op.crease_width:
            continue
        t = 1.0 - (dist / max(op.crease_width, 1e-6))
        if op.crease_sharpness == 'SOFT':
            t = t * t * (3 - 2 * t)
        depth = op.crease_depth * t * w * op.weight * global_weight
        v.co -= v.normal * depth
    bm.to_mesh(mesh)
    bm.free()


def apply_puff(mesh, group_index, op, global_weight, protect):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform = ensure_deform_layer(bm)
    protected = collect_protected_vertices(mesh) if protect else set()
    center = Vector((0, 0, 0))
    total_w = 0.0
    zone_verts = []
    for v in bm.verts:
        if v.index in protected:
            continue
        w = vertex_weight(v[deform], group_index)
        if w > 0:
            center += v.co * w
            total_w += w
            zone_verts.append((v, w))
    if total_w <= 0:
        bm.free()
        return
    center /= total_w
    max_dist = max((v.co - center).length for v, _w in zone_verts) if zone_verts else 0.001
    for v, w in zone_verts:
        d = (v.co - center).length / max(max_dist, 1e-6)
        fall = math.exp(-(d ** 2) * 2.2)
        if op.puff_profile == 'FLAT':
            fall *= 0.7
        elif op.puff_profile == 'OVAL':
            fall *= (1.2 if abs((v.co - center).x) > abs((v.co - center).y) else 0.9)
        if op.puff_asym == 'X':
            fall *= 1.0 + 0.3 * math.copysign(1.0, (v.co - center).x)
        elif op.puff_asym == 'Y':
            fall *= 1.0 + 0.3 * math.copysign(1.0, (v.co - center).y)
        v.co += v.normal * op.puff_height * fall * w * op.weight * global_weight
    bm.to_mesh(mesh)
    bm.free()


def apply_smooth(mesh, group_index, op, global_weight, protect):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform = ensure_deform_layer(bm)
    protected = collect_protected_vertices(mesh) if protect else set()
    for _i in range(op.smooth_iterations):
        new_pos = {}
        for v in bm.verts:
            if v.index in protected:
                continue
            w = vertex_weight(v[deform], group_index)
            if w <= 0:
                continue
            if op.smooth_keep_border and any(e.is_boundary for e in v.link_edges):
                continue
            if not v.link_edges:
                continue
            avg = Vector((0, 0, 0))
            n = 0
            for e in v.link_edges:
                avg += e.other_vert(v).co
                n += 1
            if n == 0:
                continue
            avg /= n
            strength = op.smooth_strength * op.weight * global_weight * w
            new_pos[v.index] = v.co.lerp(avg, strength)
        for idx, pos in new_pos.items():
            bm.verts[idx].co = pos
    bm.to_mesh(mesh)
    bm.free()


def apply_noise(mesh, group_index, op, global_weight, protect):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform = ensure_deform_layer(bm)
    protected = collect_protected_vertices(mesh) if protect else set()
    for v in bm.verts:
        if v.index in protected:
            continue
        w = vertex_weight(v[deform], group_index)
        if w <= 0:
            continue
        n = perlin_at(v.co, op.noise_scale, op.noise_detail, op.noise_roughness, op.noise_seed)
        disp = n * op.noise_intensity * op.weight * global_weight * w
        v.co += v.normal * disp
    bm.to_mesh(mesh)
    bm.free()


def apply_operation(mesh, group_index, op, global_weight, protect):
    if not op.enabled:
        return
    if op.op_type == 'INFLATE':
        apply_inflate_flatten(mesh, group_index, op, global_weight, protect)
    elif op.op_type == 'WAVES':
        apply_waves(mesh, group_index, op, global_weight, protect)
    elif op.op_type == 'FOLDS':
        apply_folds(mesh, group_index, op, global_weight, protect)
    elif op.op_type == 'BUMPS':
        apply_bumps(mesh, group_index, op, global_weight, protect)
    elif op.op_type == 'CREASE':
        apply_crease(mesh, group_index, op, global_weight, protect)
    elif op.op_type == 'PUFF':
        apply_puff(mesh, group_index, op, global_weight, protect)
    elif op.op_type == 'SMOOTH':
        apply_smooth(mesh, group_index, op, global_weight, protect)
    elif op.op_type == 'NOISE':
        apply_noise(mesh, group_index, op, global_weight, protect)


def apply_zone_stack(zone, scene):
    obj = get_object_for_zone(zone)
    if not obj or obj.type != 'MESH':
        return
    vg = obj.vertex_groups.get(zone.vgroup_name)
    if not vg:
        return
    original = save_original_coords(obj)
    for op in zone.operations:
        apply_operation(obj.data, vg.index, op, scene.softform_global_weight, scene.softform_protect_edges)
    if scene.softform_preserve_box_uv:
        preserve_box_uv(obj.data, original)
    obj.data.update()


def store_zone_meta_on_object(zone):
    obj = get_object_for_zone(zone)
    if not obj:
        return
    key = "softform_zones_json"
    existing = {}
    try:
        existing = json.loads(obj.get(key, "{}"))
    except Exception:
        existing = {}
    existing[zone.name] = {
        "vgroup_name": zone.vgroup_name,
        "color": list(zone.color),
        "operations": [
            {
                "op_type": o.op_type,
                "enabled": o.enabled,
                "weight": o.weight,
                "inflate_dir": o.inflate_dir,
                "inflate_intensity": o.inflate_intensity,
                "falloff": o.falloff,
                "waves_angle": o.waves_angle,
                "waves_amplitude": o.waves_amplitude,
                "waves_length": o.waves_length,
                "waves_phase": o.waves_phase,
                "waves_randomness": o.waves_randomness,
                "waves_spread": o.waves_spread,
                "folds_angle": o.folds_angle,
                "folds_intensity": o.folds_intensity,
                "folds_length": o.folds_length,
                "folds_frequency": o.folds_frequency,
                "folds_randomness": o.folds_randomness,
                "folds_spread": o.folds_spread,
                "folds_profile": o.folds_profile,
                "bumps_intensity": o.bumps_intensity,
                "bumps_frequency": o.bumps_frequency,
                "bumps_randomness": o.bumps_randomness,
                "bumps_direction": o.bumps_direction,
                "bumps_shape": o.bumps_shape,
                "crease_depth": o.crease_depth,
                "crease_width": o.crease_width,
                "crease_sharpness": o.crease_sharpness,
                "crease_angle": o.crease_angle,
                "puff_height": o.puff_height,
                "puff_profile": o.puff_profile,
                "puff_asym": o.puff_asym,
                "smooth_strength": o.smooth_strength,
                "smooth_iterations": o.smooth_iterations,
                "smooth_keep_border": o.smooth_keep_border,
                "noise_intensity": o.noise_intensity,
                "noise_scale": o.noise_scale,
                "noise_detail": o.noise_detail,
                "noise_roughness": o.noise_roughness,
                "noise_seed": o.noise_seed,
            }
            for o in zone.operations
        ],
    }
    obj[key] = json.dumps(existing)


class SoftFormOperation(bpy.types.PropertyGroup):
    op_type: EnumProperty(items=[
        ('INFLATE', 'Inflate/Flatten', ''),
        ('WAVES', 'Waves/Golven', ''),
        ('FOLDS', 'Folds/Vouwen', ''),
        ('BUMPS', 'Bumps/Hobbels', ''),
        ('CREASE', 'Crease/Naad', ''),
        ('PUFF', 'Puff/Kussen', ''),
        ('SMOOTH', 'Smooth/Glad', ''),
        ('NOISE', 'Noise Displace', ''),
    ], name="Bewerking")
    enabled: BoolProperty(name="Actief", default=True)
    weight: FloatProperty(name="Subtiel ↔ Extreem", min=0.0, max=1.0, default=0.5)

    inflate_dir: FloatProperty(name="Richting", min=-1.0, max=1.0, default=0.25)
    inflate_intensity: FloatProperty(name="Intensiteit", min=0.0, max=1.0, default=0.2)
    falloff: EnumProperty(name="Falloff", items=[('SHARP', 'Scherp', ''), ('SMOOTH', 'Zacht', ''), ('LINEAR', 'Lineair', '')], default='SMOOTH')

    waves_angle: FloatProperty(name="Richting", min=0.0, max=360.0, default=0.0)
    waves_amplitude: FloatProperty(name="Amplitude", min=0.001, max=0.1, default=0.01)
    waves_length: FloatProperty(name="Golflengte", min=0.01, max=1.0, default=0.2)
    waves_phase: FloatProperty(name="Fase", min=0.0, max=1.0, default=0.0)
    waves_randomness: FloatProperty(name="Randomness", min=0.0, max=1.0, default=0.1)
    waves_spread: EnumProperty(name="Spreiding", items=[('NARROW', 'Narrow', ''), ('NORMAL', 'Normal', ''), ('WIDE', 'Wide', '')], default='NORMAL')

    folds_angle: FloatProperty(name="Richting", min=0.0, max=360.0, default=0.0)
    folds_intensity: FloatProperty(name="Intensiteit", min=0.001, max=0.15, default=0.01)
    folds_length: FloatProperty(name="Lengte", min=0.01, max=0.5, default=0.1)
    folds_frequency: IntProperty(name="Frequentie", min=1, max=20, default=4)
    folds_randomness: FloatProperty(name="Randomness", min=0.0, max=1.0, default=0.1)
    folds_spread: FloatProperty(name="Spreiding", min=0.1, max=2.0, default=1.0)
    folds_profile: EnumProperty(name="Profiel", items=[('V', 'V-vouw', ''), ('U', 'U-vouw', ''), ('ASYM', 'Asym', '')], default='V')

    bumps_intensity: FloatProperty(name="Intensiteit", min=0.001, max=0.1, default=0.02)
    bumps_frequency: FloatProperty(name="Frequentie", min=1.0, max=30.0, default=8.0)
    bumps_randomness: FloatProperty(name="Randomness", min=0.0, max=1.0, default=0.5)
    bumps_direction: EnumProperty(name="Richting", items=[('NORMAL', 'Normal', ''), ('TANGENT', 'Tangent', '')], default='NORMAL')
    bumps_shape: EnumProperty(name="Vorm", items=[('ROUND', 'Round', ''), ('FLAT', 'Flat-top', ''), ('SHARP', 'Sharp', '')], default='ROUND')

    crease_depth: FloatProperty(name="Diepte", min=0.001, max=0.05, default=0.01)
    crease_width: FloatProperty(name="Breedte", min=0.001, max=0.05, default=0.02)
    crease_sharpness: EnumProperty(name="Scherpte", items=[('SHARP', 'Scherp', ''), ('SOFT', 'Zacht', '')], default='SOFT')
    crease_angle: FloatProperty(name="Richtingslijn", min=0.0, max=360.0, default=0.0)

    puff_height: FloatProperty(name="Hoogte", min=0.001, max=0.2, default=0.03)
    puff_profile: EnumProperty(name="Profiel", items=[('SPHERE', 'Bol', ''), ('FLAT', 'Plat', ''), ('OVAL', 'Ovaal', '')], default='SPHERE')
    puff_asym: EnumProperty(name="Asymmetrie", items=[('NONE', 'Geen', ''), ('X', 'X-as', ''), ('Y', 'Y-as', '')], default='NONE')

    smooth_strength: FloatProperty(name="Sterkte", min=0.0, max=1.0, default=0.4)
    smooth_iterations: IntProperty(name="Iteraties", min=1, max=20, default=4)
    smooth_keep_border: BoolProperty(name="Grens bewaren", default=True)

    noise_intensity: FloatProperty(name="Intensiteit", min=0.001, max=0.1, default=0.01)
    noise_scale: FloatProperty(name="Schaal", min=0.1, max=10.0, default=2.0)
    noise_detail: IntProperty(name="Detail", min=1, max=8, default=4)
    noise_roughness: FloatProperty(name="Ruwheid", min=0.0, max=1.0, default=0.5)
    noise_seed: IntProperty(name="Seed", default=1)


class SoftFormZone(bpy.types.PropertyGroup):
    name: StringProperty(name="Naam")
    object_name: StringProperty(name="Object")
    vgroup_name: StringProperty(name="Vertex Group")
    color: FloatVectorProperty(name="Kleur", subtype='COLOR', size=4, default=(1, 0, 0, 1), min=0.0, max=1.0)
    operations: CollectionProperty(type=SoftFormOperation)
    active_op_index: IntProperty(default=0)


class SOFTFORM_OT_set_mode(bpy.types.Operator):
    bl_idname = "softform.set_mode"
    bl_label = "Selectiemethode"
    bl_options = {'REGISTER', 'UNDO'}
    mode: EnumProperty(items=[('OBJECT', 'Object select', ''), ('FACE_LINKED', 'Face select (linked)', ''), ('FACE_LASSO', 'Face select (lasso)', '')])

    def execute(self, context):
        scn = context.scene
        scn.softform_selection_mode = self.mode
        if self.mode == 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        else:
            try:
                bpy.ops.object.mode_set(mode='EDIT')
                context.tool_settings.mesh_select_mode = (False, False, True)
            except Exception:
                pass
        scn.softform_wizard_step = 1
        return {'FINISHED'}


class SOFTFORM_OT_add_zone(bpy.types.Operator):
    bl_idname = "softform.add_zone"
    bl_label = "Nieuwe zone"
    bl_options = {'REGISTER', 'UNDO'}
    use_paint: BoolProperty(default=False)

    def execute(self, context):
        scn = context.scene
        try:
            mode = scn.softform_selection_mode
            if mode == 'OBJECT':
                targets = [o for o in context.selected_objects if o and o.type == 'MESH']
                if not targets:
                    safe_report(self, {'ERROR'}, "Selecteer minimaal één mesh-object.")
                    return {'CANCELLED'}
                for obj in targets:
                    vg = obj.vertex_groups.new(name=f"SF_{next_zone_name(scn).replace(' ', '_')}")
                    all_idx = [v.index for v in obj.data.vertices]
                    if all_idx:
                        vg.add(all_idx, 1.0, 'REPLACE')
                    zone = scn.softform_zones.add()
                    zone.name = next_zone_name(scn)
                    zone.object_name = obj.name
                    zone.vgroup_name = vg.name
                    zone.color = unique_zone_color(len(scn.softform_zones))
                    if self.use_paint:
                        context.view_layer.objects.active = obj
                        obj.vertex_groups.active = vg
                        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
                        scn.softform_painting_zone = zone.name
            else:
                obj = context.object
                if not obj or obj.type != 'MESH':
                    safe_report(self, {'ERROR'}, "Selecteer een mesh-object in Edit Mode.")
                    return {'CANCELLED'}
                bm = bmesh.from_edit_mesh(obj.data)
                selected = [f for f in bm.faces if f.select]
                if not selected:
                    safe_report(self, {'ERROR'}, "Selecteer eerst faces voor de zone.")
                    return {'CANCELLED'}
                verts = sorted({v.index for f in selected for v in f.verts})
                bpy.ops.object.mode_set(mode='OBJECT')
                vg = obj.vertex_groups.new(name=f"SF_{next_zone_name(scn).replace(' ', '_')}")
                vg.add(verts, 1.0, 'REPLACE')
                zone = scn.softform_zones.add()
                zone.name = next_zone_name(scn)
                zone.object_name = obj.name
                zone.vgroup_name = vg.name
                zone.color = (1.0, 0.1, 0.1, 1.0)
                bpy.ops.object.mode_set(mode='EDIT')
            scn.softform_wizard_step = 2
            scn.softform_active_zone = len(scn.softform_zones) - 1
            return {'FINISHED'}
        except Exception as ex:
            logger.error("SoftForm add zone failed: %s\n%s", ex, traceback.format_exc())
            safe_report(self, {'ERROR'}, "Zone maken is mislukt. Bekijk de console voor details.")
            return {'CANCELLED'}


class SOFTFORM_OT_finish_paint(bpy.types.Operator):
    bl_idname = "softform.finish_paint"
    bl_label = "Klaar met schilderen"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        scn.softform_painting_zone = ""
        scn.softform_wizard_step = 2
        return {'FINISHED'}


class SOFTFORM_OT_select_zone(bpy.types.Operator):
    bl_idname = "softform.select_zone"
    bl_label = "Selecteer zone"
    index: IntProperty()

    def execute(self, context):
        scn = context.scene
        if self.index < 0 or self.index >= len(scn.softform_zones):
            return {'CANCELLED'}
        scn.softform_active_zone = self.index
        zone = scn.softform_zones[self.index]
        obj = get_object_for_zone(zone)
        if obj:
            context.view_layer.objects.active = obj
            obj.select_set(True)
            if scn.softform_show_highlights:
                try:
                    vg = obj.vertex_groups.get(zone.vgroup_name)
                    if vg:
                        obj.vertex_groups.active = vg
                except Exception:
                    pass
        return {'FINISHED'}


class SOFTFORM_OT_delete_zone(bpy.types.Operator):
    bl_idname = "softform.delete_zone"
    bl_label = "Verwijder zone"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty()

    def execute(self, context):
        scn = context.scene
        if self.index < 0 or self.index >= len(scn.softform_zones):
            return {'CANCELLED'}
        zone = scn.softform_zones[self.index]
        obj = get_object_for_zone(zone)
        if obj:
            vg = obj.vertex_groups.get(zone.vgroup_name)
            if vg:
                obj.vertex_groups.remove(vg)
        scn.softform_zones.remove(self.index)
        scn.softform_active_zone = max(0, min(scn.softform_active_zone, len(scn.softform_zones)-1))
        return {'FINISHED'}


class SOFTFORM_OT_add_operation(bpy.types.Operator):
    bl_idname = "softform.add_operation"
    bl_label = "+ Bewerking toevoegen"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        zone = get_active_zone(context.scene)
        if not zone:
            safe_report(self, {'ERROR'}, "Maak of selecteer eerst een zone.")
            return {'CANCELLED'}
        op = zone.operations.add()
        op.op_type = 'INFLATE'
        zone.active_op_index = len(zone.operations) - 1
        context.scene.softform_wizard_step = 3
        return {'FINISHED'}


class SOFTFORM_OT_remove_operation(bpy.types.Operator):
    bl_idname = "softform.remove_operation"
    bl_label = "Verwijder bewerking"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty()

    def execute(self, context):
        zone = get_active_zone(context.scene)
        if not zone:
            return {'CANCELLED'}
        if self.index < 0 or self.index >= len(zone.operations):
            return {'CANCELLED'}
        zone.operations.remove(self.index)
        zone.active_op_index = max(0, min(zone.active_op_index, len(zone.operations)-1))
        return {'FINISHED'}


class SOFTFORM_OT_move_operation(bpy.types.Operator):
    bl_idname = "softform.move_operation"
    bl_label = "Verplaats bewerking"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty()
    direction: IntProperty()

    def execute(self, context):
        zone = get_active_zone(context.scene)
        if not zone:
            return {'CANCELLED'}
        ni = self.index + self.direction
        if ni < 0 or ni >= len(zone.operations):
            return {'CANCELLED'}
        zone.operations.move(self.index, ni)
        zone.active_op_index = ni
        return {'FINISHED'}


class SOFTFORM_OT_apply_preview(bpy.types.Operator):
    bl_idname = "softform.apply_preview"
    bl_label = "Preview bijwerken"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene
        try:
            if not scn.softform_live_preview:
                return {'FINISHED'}
            zone = get_active_zone(scn)
            if not zone:
                return {'CANCELLED'}
            obj = get_object_for_zone(zone)
            if not obj:
                return {'CANCELLED'}
            key = obj.name_full
            if key not in PREVIEW_CACHE:
                PREVIEW_CACHE[key] = save_original_coords(obj)
            restore_original_coords(obj, PREVIEW_CACHE[key])
            apply_zone_stack(zone, scn)
            return {'FINISHED'}
        except Exception as ex:
            logger.error("SoftForm preview failed: %s\n%s", ex, traceback.format_exc())
            safe_report(self, {'ERROR'}, "Live preview mislukt. Controleer de console.")
            return {'CANCELLED'}


class SOFTFORM_OT_commit(bpy.types.Operator):
    bl_idname = "softform.commit"
    bl_label = "✅ Toepassen"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        scn = context.scene
        try:
            for zone in scn.softform_zones:
                obj = get_object_for_zone(zone)
                if not obj or obj.type != 'MESH':
                    continue
                if obj.name_full in PREVIEW_CACHE:
                    restore_original_coords(obj, PREVIEW_CACHE[obj.name_full])
                apply_zone_stack(zone, scn)
                store_zone_meta_on_object(zone)
                PREVIEW_CACHE.pop(obj.name_full, None)
            safe_report(self, {'INFO'}, "SoftForm bewerkingen toegepast.")
            return {'FINISHED'}
        except Exception as ex:
            logger.error("SoftForm commit failed: %s\n%s", ex, traceback.format_exc())
            safe_report(self, {'ERROR'}, "Toepassen mislukt. Bekijk console voor details.")
            return {'CANCELLED'}


class SOFTFORM_OT_reset_all(bpy.types.Operator):
    bl_idname = "softform.reset_all"
    bl_label = "↩ Alles terugzetten"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene
        try:
            for zone in list(scn.softform_zones):
                obj = get_object_for_zone(zone)
                if obj and obj.name_full in PREVIEW_CACHE:
                    restore_original_coords(obj, PREVIEW_CACHE[obj.name_full])
            PREVIEW_CACHE.clear()
            for obj in bpy.data.objects:
                if obj.type != 'MESH':
                    continue
                for vg in list(obj.vertex_groups):
                    if vg.name.startswith("SF_"):
                        obj.vertex_groups.remove(vg)
                if "softform_zones_json" in obj:
                    del obj["softform_zones_json"]
            scn.softform_zones.clear()
            scn.softform_active_zone = 0
            safe_report(self, {'INFO'}, "Alles is teruggezet.")
            return {'FINISHED'}
        except Exception as ex:
            logger.error("SoftForm reset failed: %s\n%s", ex, traceback.format_exc())
            safe_report(self, {'ERROR'}, "Reset mislukt. Bekijk console voor details.")
            return {'CANCELLED'}


class SOFTFORM_OT_save_preset(bpy.types.Operator):
    bl_idname = "softform.save_preset"
    bl_label = "Preset opslaan"

    def execute(self, context):
        scn = context.scene
        zone = get_active_zone(scn)
        if not zone:
            safe_report(self, {'ERROR'}, "Selecteer eerst een zone.")
            return {'CANCELLED'}
        name = scn.softform_preset_name.strip()
        if not name:
            safe_report(self, {'ERROR'}, "Geef eerst een preset-naam op.")
            return {'CANCELLED'}
        data = []
        for o in zone.operations:
            data.append({k: getattr(o, k) for k in o.bl_rna.properties.keys() if k not in {"rna_type"}})
        presets = json.loads(scn.get("softform_presets_json", "{}"))
        presets[name] = data
        scn["softform_presets_json"] = json.dumps(presets)
        safe_report(self, {'INFO'}, f"Preset '{name}' opgeslagen.")
        return {'FINISHED'}


class SOFTFORM_OT_load_preset(bpy.types.Operator):
    bl_idname = "softform.load_preset"
    bl_label = "Preset inladen"

    def execute(self, context):
        scn = context.scene
        zone = get_active_zone(scn)
        if not zone:
            safe_report(self, {'ERROR'}, "Selecteer eerst een zone.")
            return {'CANCELLED'}
        name = scn.softform_preset_name.strip()
        presets = json.loads(scn.get("softform_presets_json", "{}"))
        if name not in presets:
            safe_report(self, {'ERROR'}, "Preset niet gevonden.")
            return {'CANCELLED'}
        zone.operations.clear()
        for item in presets[name]:
            op = zone.operations.add()
            for k, v in item.items():
                if k == "rna_type":
                    continue
                try:
                    setattr(op, k, v)
                except Exception:
                    pass
        zone.active_op_index = 0
        bpy.ops.softform.apply_preview()
        return {'FINISHED'}


class SOFTFORM_OT_delete_preset(bpy.types.Operator):
    bl_idname = "softform.delete_preset"
    bl_label = "Preset verwijderen"

    def execute(self, context):
        scn = context.scene
        name = scn.softform_preset_name.strip()
        presets = json.loads(scn.get("softform_presets_json", "{}"))
        if name in presets:
            del presets[name]
            scn["softform_presets_json"] = json.dumps(presets)
            safe_report(self, {'INFO'}, f"Preset '{name}' verwijderd.")
        return {'FINISHED'}


class SOFTFORM_OT_pick_preset(bpy.types.Operator):
    bl_idname = "softform.pick_preset"
    bl_label = "Kies preset"
    name: StringProperty()

    def execute(self, context):
        context.scene.softform_preset_name = self.name
        return {'FINISHED'}


class VIEW3D_PT_softform(bpy.types.Panel):
    bl_label = "SoftForm"
    bl_idname = "VIEW3D_PT_softform"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SoftForm'

    def draw_operation(self, layout, op, idx):
        box = layout.box()
        row = box.row(align=True)
        row.prop(op, "enabled", text="")
        row.prop(op, "op_type", text="")
        up = row.operator("softform.move_operation", text="↑")
        up.index = idx
        up.direction = -1
        down = row.operator("softform.move_operation", text="↓")
        down.index = idx
        down.direction = 1
        rem = row.operator("softform.remove_operation", text="🗑")
        rem.index = idx

        box.prop(op, "weight")
        t = op.op_type
        if t == 'INFLATE':
            box.prop(op, "inflate_dir")
            box.prop(op, "inflate_intensity")
            box.prop(op, "falloff")
        elif t == 'WAVES':
            for p in ["waves_angle", "waves_amplitude", "waves_length", "waves_phase", "waves_randomness", "waves_spread"]:
                box.prop(op, p)
        elif t == 'FOLDS':
            for p in ["folds_angle", "folds_intensity", "folds_length", "folds_frequency", "folds_randomness", "folds_spread", "folds_profile"]:
                box.prop(op, p)
        elif t == 'BUMPS':
            for p in ["bumps_intensity", "bumps_frequency", "bumps_randomness", "bumps_direction", "bumps_shape"]:
                box.prop(op, p)
        elif t == 'CREASE':
            for p in ["crease_depth", "crease_width", "crease_sharpness", "crease_angle"]:
                box.prop(op, p)
        elif t == 'PUFF':
            for p in ["puff_height", "puff_profile", "puff_asym"]:
                box.prop(op, p)
        elif t == 'SMOOTH':
            for p in ["smooth_strength", "smooth_iterations", "smooth_keep_border"]:
                box.prop(op, p)
        elif t == 'NOISE':
            for p in ["noise_intensity", "noise_scale", "noise_detail", "noise_roughness", "noise_seed"]:
                box.prop(op, p)

    def draw(self, context):
        scn = context.scene
        layout = self.layout

        head = layout.box()
        row = head.row(align=True)
        row.label(text=f"Stap {scn.softform_wizard_step}/3")
        row.prop(scn, "softform_live_preview", text="Live preview")

        # Step 1
        box1 = layout.box()
        box1.label(text="Stap 1 · Selectie")
        row = box1.row(align=True)
        b = row.operator("softform.set_mode", text="Object select")
        b.mode = 'OBJECT'
        b = row.operator("softform.set_mode", text="Face linked")
        b.mode = 'FACE_LINKED'
        b = row.operator("softform.set_mode", text="Face lasso")
        b.mode = 'FACE_LASSO'
        box1.label(text=f"Actieve methode: {scn.softform_selection_mode}")

        # Step 2
        box2 = layout.box()
        box2.label(text="Stap 2 · Zone maken")
        row = box2.row(align=True)
        if scn.softform_selection_mode == 'OBJECT':
            row.operator("softform.add_zone", text="Nieuwe zone (inpaint)").use_paint = True
        row.operator("softform.add_zone", text="Nieuwe zone (zonder paint)").use_paint = False
        if scn.softform_painting_zone:
            p = box2.box()
            p.label(text=f"Schilder zone: {scn.softform_painting_zone}")
            p.prop(scn, "softform_brush_radius")
            p.prop(scn, "softform_brush_strength")
            p.operator("softform.finish_paint")

        box2.prop(scn, "softform_show_highlights")
        for i, z in enumerate(scn.softform_zones):
            r = box2.row(align=True)
            op = r.operator("softform.select_zone", text=z.name)
            op.index = i
            r.prop(z, "color", text="")
            d = r.operator("softform.delete_zone", text="🗑")
            d.index = i

        # Step 3
        box3 = layout.box()
        box3.label(text="Stap 3 · Effecten")
        zone = get_active_zone(scn)
        if not zone:
            box3.label(text="Selecteer eerst een zone.")
        else:
            box3.label(text=f"Zone: {zone.name} ({zone.object_name})")
            box3.prop(scn, "softform_global_weight")
            box3.operator("softform.add_operation")
            for idx, op in enumerate(zone.operations):
                self.draw_operation(box3, op, idx)
            row = box3.row(align=True)
            row.prop(scn, "softform_preset_name", text="Preset")
            row.operator("softform.save_preset", text="Opslaan")
            row.operator("softform.load_preset", text="Inladen")
            row.operator("softform.delete_preset", text="Verwijder")
            presets = json.loads(scn.get("softform_presets_json", "{}"))
            if presets:
                col = box3.column(align=True)
                col.label(text="Preset lijst")
                for name in sorted(presets.keys()):
                    p = col.operator("softform.pick_preset", text=name)
                    p.name = name
            box3.operator("softform.apply_preview", text="Preview bijwerken")

        options = layout.box()
        options.label(text="Bescherming")
        options.prop(scn, "softform_protect_edges")
        options.prop(scn, "softform_preserve_box_uv")

        foot = layout.box()
        foot.operator("softform.commit")
        foot.operator("softform.reset_all")


@persistent
def softform_load_handler(_dummy):
    PREVIEW_CACHE.clear()


classes = (
    SoftFormOperation,
    SoftFormZone,
    SOFTFORM_OT_set_mode,
    SOFTFORM_OT_add_zone,
    SOFTFORM_OT_finish_paint,
    SOFTFORM_OT_select_zone,
    SOFTFORM_OT_delete_zone,
    SOFTFORM_OT_add_operation,
    SOFTFORM_OT_remove_operation,
    SOFTFORM_OT_move_operation,
    SOFTFORM_OT_apply_preview,
    SOFTFORM_OT_commit,
    SOFTFORM_OT_reset_all,
    SOFTFORM_OT_save_preset,
    SOFTFORM_OT_load_preset,
    SOFTFORM_OT_delete_preset,
    SOFTFORM_OT_pick_preset,
    VIEW3D_PT_softform,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.softform_zones = CollectionProperty(type=SoftFormZone)
    bpy.types.Scene.softform_active_zone = IntProperty(default=0)
    bpy.types.Scene.softform_wizard_step = IntProperty(default=1, min=1, max=3)
    bpy.types.Scene.softform_selection_mode = EnumProperty(
        name="Selectiemodus",
        items=[('OBJECT', 'Object select', ''), ('FACE_LINKED', 'Face linked', ''), ('FACE_LASSO', 'Face lasso', '')],
        default='OBJECT',
    )
    bpy.types.Scene.softform_live_preview = BoolProperty(name="Live preview", default=True)
    bpy.types.Scene.softform_show_highlights = BoolProperty(name="Toon zone-highlights", default=True)
    bpy.types.Scene.softform_painting_zone = StringProperty(default="")
    bpy.types.Scene.softform_brush_radius = FloatProperty(name="Brush radius", min=0.0, max=1.0, default=0.3)
    bpy.types.Scene.softform_brush_strength = FloatProperty(name="Brush sterkte", min=0.0, max=1.0, default=0.5)
    bpy.types.Scene.softform_global_weight = FloatProperty(name="Gewicht", min=0.0, max=1.0, default=0.5)
    bpy.types.Scene.softform_preset_name = StringProperty(name="Preset naam")
    bpy.types.Scene.softform_protect_edges = BoolProperty(name="Bescherm mesh-randen (voorkom gaten op naden)", default=True)
    bpy.types.Scene.softform_preserve_box_uv = BoolProperty(name="Box UV schaal behouden", default=True)

    if softform_load_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(softform_load_handler)


def unregister():
    if softform_load_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(softform_load_handler)

    props = [
        "softform_zones", "softform_active_zone", "softform_wizard_step", "softform_selection_mode",
        "softform_live_preview", "softform_show_highlights", "softform_painting_zone",
        "softform_brush_radius", "softform_brush_strength", "softform_global_weight",
        "softform_preset_name", "softform_protect_edges", "softform_preserve_box_uv",
    ]
    for p in props:
        if hasattr(bpy.types.Scene, p):
            delattr(bpy.types.Scene, p)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
