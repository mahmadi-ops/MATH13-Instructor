"""Blender scene for the rope-between-two-trees animation.

build() makes every object once; apply(st) poses them for one frame from
the state dict of geom.state().  Rendering is driven frame by frame from
render.py, so nothing here is keyframed.
"""
import math
import os

import bpy
from mathutils import Matrix, Vector

import geom as g

FONT_DIR = None  # set by build()


def srgb(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


# ---------------------------------------------------------------- materials
def _fadeable(name, surface_builder):
    """Material whose surface is mixed with Transparent by a Value node
    called 'Alpha' (1 = opaque)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    surf = surface_builder(nt)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    val = nt.nodes.new("ShaderNodeValue")
    val.name = "Alpha"
    val.outputs[0].default_value = 1.0
    nt.links.new(val.outputs[0], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(surf, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def mat_lit(name, hexcol, rough=0.7, alpha=None):
    def b(nt):
        n = nt.nodes.new("ShaderNodeBsdfPrincipled")
        n.inputs["Base Color"].default_value = (*srgb(hexcol), 1)
        n.inputs["Roughness"].default_value = rough
        if alpha is not None:
            n.inputs["Alpha"].default_value = alpha
        return n.outputs[0]
    return _fadeable(name, b)


def mat_flat(name, hexcol, strength=1.0):
    def b(nt):
        n = nt.nodes.new("ShaderNodeEmission")
        n.inputs["Color"].default_value = (*srgb(hexcol), 1)
        n.inputs["Strength"].default_value = strength
        return n.outputs[0]
    return _fadeable(name, b)


def mat_marker(name, hexcol):
    """Lit but with a strong emission floor, so the colour reads in shade."""
    def b(nt):
        n = nt.nodes.new("ShaderNodeBsdfPrincipled")
        n.inputs["Base Color"].default_value = (*srgb(hexcol), 1)
        n.inputs["Roughness"].default_value = 0.35
        n.inputs["Emission Color"].default_value = (*srgb(hexcol), 1)
        n.inputs["Emission Strength"].default_value = 0.55
        return n.outputs[0]
    return _fadeable(name, b)


def set_alpha(m, a):
    m.node_tree.nodes["Alpha"].outputs[0].default_value = max(0.0, min(1.0, a))


# ---------------------------------------------------------------- helpers
def link(obj, coll="Scene"):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def new_mesh_obj(name, bm_func, mat=None):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm_func(bm)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    link(ob)
    if mat:
        ob.data.materials.append(mat)
    return ob


def cylinder(name, r, depth, verts=16, mat=None, smooth=True):
    import bmesh

    def f(bm):
        bmesh.ops.create_cone(bm, cap_ends=True, segments=verts,
                              radius1=r, radius2=r, depth=depth)
    ob = new_mesh_obj(name, f, mat)
    if smooth:
        for p in ob.data.polygons:
            p.use_smooth = len(p.vertices) == 4
    return ob


def cone(name, r1, r2, depth, verts=9, mat=None):
    import bmesh

    def f(bm):
        bmesh.ops.create_cone(bm, cap_ends=True, segments=verts,
                              radius1=r1, radius2=r2, depth=depth)
    return new_mesh_obj(name, f, mat)


def sphere(name, r, mat=None, seg=24, rings=12, smooth=True):
    import bmesh

    def f(bm):
        bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=r)
    ob = new_mesh_obj(name, f, mat)
    if smooth:
        for p in ob.data.polygons:
            p.use_smooth = True
    return ob


def place_segment(ob, p, q):
    """ob is a unit-length cylinder along z centred at its origin."""
    p, q = Vector(p), Vector(q)
    d = q - p
    L = max(d.length, 1e-4)
    rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    ob.matrix_world = Matrix.Translation((p + q) / 2) @ rot @ Matrix.Diagonal((1, 1, L, 1))


def capsule(name, p, q, r, mat, parent=None):
    """Cylinder from p to q with spherical ends (local coordinates of parent)."""
    p, q = Vector(p), Vector(q)
    c = cylinder(name, r, 1.0, 14, mat)
    place_segment(c, p, q)
    parts = [c]
    for i, e in enumerate((p, q)):
        s = sphere(f"{name}_end{i}", r, mat, 14, 8)
        s.location = e
        parts.append(s)
    if parent:
        for o in parts:
            mw = o.matrix_world.copy()
            o.parent = parent
            o.matrix_world = mw  # parent sits at the origin while building
    return parts


def text(name, body, size, mat, font, align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.font = font
    ob = bpy.data.objects.new(name, cu)
    link(ob)
    ob.data.materials.append(mat)
    return ob


def billboard(ob, cam):
    c = ob.constraints.new("COPY_ROTATION")
    c.target = cam


def poly_curve(name, pts, bevel, mat):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = bevel
    cu.bevel_resolution = 4
    cu.use_fill_caps = True
    if pts:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for i, (x, z) in enumerate(pts):
            sp.points[i].co = (x, 0.0, z, 1.0)
    ob = bpy.data.objects.new(name, cu)
    link(ob)
    ob.data.materials.append(mat)
    return ob


# ---------------------------------------------------------------- the scene
OBJ = {}
MAT = {}
DASHES = []


def _compute_dashes():
    """Dash intervals, by arc length, over the out-of-reach arc [T1, T0 + 2 pi]."""
    DASHES.clear()
    n = 4000
    ts = [g.T1 + (g.T0 + 2 * math.pi - g.T1) * i / n for i in range(n + 1)]
    s = [0.0]
    for i in range(n):
        s.append(s[-1] + g.dist(g.ell(ts[i]), g.ell(ts[i + 1])))
    on, off = 2.6, 1.9
    total = s[-1]

    def t_of(sv):
        j = max(0, min(n - 1, int(sv / total * n)))
        while j < n and s[j + 1] < sv:
            j += 1
        while j > 0 and s[j] > sv:
            j -= 1
        f = (sv - s[j]) / max(s[j + 1] - s[j], 1e-9)
        return ts[j] + f * (ts[j + 1] - ts[j])

    start = 0.0
    while start < total:
        end = min(start + on, total)
        DASHES.append((t_of(start), t_of(end)))
        start += on + off


RIDER_MATS = ("bottle", "cap", "shirt", "pants", "skin", "hair", "shoe")
OBJ_NAMES = {"cam": "Cam", "ropeAP": "ropeAP", "ropePB": "ropePB", "markP": "markP",
             "rider": "Rider", "trail": "trail", "trail_pre": "trail_pre", "dash": "dash",
             "labP": "labP", "ropelab": "ropeLab", "vplane": "VPlane"}


def rebind():
    """Refill OBJ, MAT and DASHES from an already-built scene (used by the
    driver script saved inside the .blend)."""
    OBJ.clear()
    MAT.clear()
    for k, name in OBJ_NAMES.items():
        OBJ[k] = bpy.data.objects[name]
    MAT.update({m.name: m for m in bpy.data.materials})
    OBJ["rider_mats"] = [MAT[k] for k in RIDER_MATS]
    _compute_dashes()


def build(font_dir):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    fonts = {
        "math": bpy.data.fonts.load(os.path.join(font_dir, "cmmi10.ttf")),
        "sans": bpy.data.fonts.load(os.path.join(font_dir, "DejaVuSans-Bold.ttf")),
    }

    # --- render settings
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 8
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = True
    sc.cycles.denoising_prefilter = "FAST"
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 2
    sc.cycles.transparent_max_bounces = 24
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.compression = 20
    try:
        sc.view_settings.view_transform = "Khronos PBR Neutral"
    except TypeError:
        sc.view_settings.view_transform = "Standard"
    sc.view_settings.exposure = 0.0

    # --- world: sky gradient for the camera, soft uniform sky for lighting
    w = bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    lp = nt.nodes.new("ShaderNodeLightPath")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    rampn = nt.nodes.new("ShaderNodeValToRGB")
    rampn.color_ramp.elements[0].position = 0.0
    rampn.color_ramp.elements[0].color = (*srgb("#EEF5FB"), 1)
    rampn.color_ramp.elements[1].position = 1.0
    rampn.color_ramp.elements[1].color = (*srgb("#9CC8EA"), 1)
    bg_cam = nt.nodes.new("ShaderNodeBackground")
    bg_light = nt.nodes.new("ShaderNodeBackground")
    bg_light.inputs["Color"].default_value = (*srgb("#C9DEF0"), 1)
    bg_light.inputs["Strength"].default_value = 0.75
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tc.outputs["Window"], sep.inputs[0])
    nt.links.new(sep.outputs["Y"], rampn.inputs["Fac"])
    nt.links.new(rampn.outputs["Color"], bg_cam.inputs["Color"])
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bg_light.outputs[0], mix.inputs[1])
    nt.links.new(bg_cam.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    # --- sun
    sun_d = bpy.data.lights.new("Sun", "SUN")
    sun_d.energy = 3.2
    sun_d.angle = math.radians(3)
    sun = bpy.data.objects.new("Sun", sun_d)
    link(sun)
    sun.rotation_euler = (math.radians(50), 0, math.radians(-38))

    # --- camera
    cam_d = bpy.data.cameras.new("Cam")
    cam_d.clip_start = 1.0
    cam_d.clip_end = 6000
    cam = bpy.data.objects.new("Cam", cam_d)
    link(cam)
    sc.camera = cam
    OBJ["cam"] = cam

    # --- materials
    MAT["ground"] = mat_lit("ground", "#7DAA5A", 0.95)
    MAT["trunk"] = mat_lit("trunk", "#7A5230", 0.9)
    MAT["leaf"] = mat_lit("leaf", "#3F7F45", 0.85)
    MAT["trunk_bg"] = mat_lit("trunk_bg", "#7A5230", 0.9)
    MAT["leaf_bg"] = mat_lit("leaf_bg", "#5E9468", 0.9)
    MAT["rope"] = mat_lit("rope", "#C9A66B", 0.8)
    MAT["mark"] = mat_marker("mark", "#D55E00")
    MAT["markP"] = mat_marker("markP", "#D55E00")
    MAT["trail"] = mat_flat("trail", "#0072B2", 1.0)
    MAT["trail2"] = mat_flat("trail2", "#0072B2", 1.0)
    MAT["dash"] = mat_flat("dash", "#6E6E6E", 1.0)
    MAT["labAB"] = mat_flat("labAB", "#C04F00", 1.0)
    MAT["labP"] = mat_flat("labP", "#C04F00", 1.0)
    MAT["dims"] = mat_flat("dims", "#333333", 1.0)
    MAT["ropelab"] = mat_flat("ropelab", "#6B4B1B", 1.0)
    MAT["plane"] = mat_lit("plane", "#5DA9E9", 0.4)
    MAT["bottle"] = mat_lit("bottle", "#9FD3F5", 0.15)
    MAT["cap"] = mat_lit("cap", "#1F6FD1", 0.4)
    MAT["shirt"] = mat_lit("shirt", "#D1344B", 0.8)
    MAT["pants"] = mat_lit("pants", "#2B3A67", 0.8)
    MAT["skin"] = mat_lit("skin", "#E8B98F", 0.7)
    MAT["hair"] = mat_lit("hair", "#4A2F1B", 0.8)
    MAT["shoe"] = mat_lit("shoe", "#2A2A2A", 0.7)

    # --- ground, with a few background trees
    import bmesh

    def plane(bm):
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=3000)
    gr = new_mesh_obj("Ground", plane, MAT["ground"])

    def conifer(name, x, y, trunk_top, crown_lo, crown_hi, r_trunk, r_crown, tm, lm):
        parts = []
        tr = cylinder(name + "_trunk", r_trunk, trunk_top, 12, tm)
        tr.location = (x, y, trunk_top / 2)
        parts.append(tr)
        n = 4
        h = (crown_hi - crown_lo)
        for i in range(n):
            z0 = crown_lo + i * h * 0.21
            depth = h * (0.42 - 0.03 * i)
            r = r_crown * (1 - 0.2 * i)
            c = cone(f"{name}_crown{i}", r, 0.0, depth, 9, lm)
            c.location = (x, y, z0 + depth / 2)
            c.rotation_euler = (0, 0, 0.6 * i)
            parts.append(c)
        return parts

    OBJ["tree1"] = conifer("Tree1", 0, 0, 122, 103, 146, 1.5, 12.0, MAT["trunk"], MAT["leaf"])
    OBJ["tree2"] = conifer("Tree2", 100, 0, 70, 54, 86, 1.35, 10.0, MAT["trunk"], MAT["leaf"])
    import random
    rnd = random.Random(7)
    for i in range(16):
        x = rnd.uniform(-260, 380)
        y = rnd.uniform(260, 700)
        hgt = rnd.uniform(45, 95)
        conifer(f"BG{i}", x, y, hgt * 0.8, hgt * 0.25, hgt, 1.2, hgt * 0.17,
                MAT["trunk_bg"], MAT["leaf_bg"])

    # --- attachment points A, B
    for nm, (x, z) in (("A", (g.AX, g.AZ)), ("B", (g.BX, g.BZ))):
        s = sphere("mark" + nm, 1.75, MAT["mark"])
        s.location = (x, 0, z)

    # --- rope: two straight pieces A-P and P-B
    OBJ["ropeAP"] = cylinder("ropeAP", 0.3, 1.0, 12, MAT["rope"])
    OBJ["ropePB"] = cylinder("ropePB", 0.3, 1.0, 12, MAT["rope"])

    # --- P marker
    OBJ["markP"] = sphere("markP", 0.95, MAT["markP"])

    # --- the rider: folded bottle + child, hanging below P
    rider = bpy.data.objects.new("Rider", None)
    link(rider)
    OBJ["rider"] = rider
    half = 2.7
    tilt = math.radians(21)
    ends = []
    for sgn in (1, -1):
        b = cylinder(f"bottle{sgn}", 0.46, half, 20, MAT["bottle"])
        b.location = (0, sgn * half / 2 * math.sin(tilt), -half / 2 * math.cos(tilt))
        b.rotation_euler = (sgn * tilt, 0, 0)
        b.parent = rider
        e = Vector((0, sgn * half * math.sin(tilt), -half * math.cos(tilt)))
        ends.append(e)
        cp = cylinder(f"cap{sgn}", 0.3, 0.32, 16, MAT["cap"])
        cp.location = e + Vector((0, sgn * 0.14 * math.sin(tilt), -0.14 * math.cos(tilt)))
        cp.rotation_euler = (sgn * tilt, 0, 0)
        cp.parent = rider
    fold = sphere("fold", 0.5, MAT["bottle"])
    fold.parent = rider
    # child (not to scale with the trees: drawn about 1.5x so it shows)
    sh_z, hip_z = -4.35, -6.55
    head = sphere("head", 0.62, MAT["skin"])
    head.location = (0, 0, -3.72)
    head.parent = rider
    hair = sphere("hair", 0.66, MAT["hair"])
    hair.location = (-0.08, 0, -3.62)
    hair.scale = (1, 1, 0.82)
    hair.parent = rider
    torso = sphere("torso", 1.0, MAT["shirt"])
    torso.location = (0, 0, (sh_z + hip_z) / 2)
    torso.scale = (0.55, 0.78, 1.25)
    torso.parent = rider
    for sgn, e in zip((1, -1), ends):
        capsule(f"arm{sgn}", (0, sgn * 0.62, sh_z + 0.2), e + Vector((0, 0, -0.25)), 0.21,
                MAT["shirt"], rider)
        hand = sphere(f"hand{sgn}", 0.26, MAT["skin"])
        hand.location = e + Vector((0, 0, -0.3))
        hand.parent = rider
        capsule(f"leg{sgn}", (0, sgn * 0.3, hip_z + 0.3), (0.25, sgn * 0.36, -8.75), 0.27,
                MAT["pants"], rider)
        shoe = sphere(f"shoe{sgn}", 0.34, MAT["shoe"])
        shoe.location = (0.42, sgn * 0.36, -9.0)
        shoe.scale = (1.5, 0.95, 0.75)
        shoe.parent = rider
    rider.scale = (1.35, 1.35, 1.35)
    OBJ["rider_mats"] = [MAT[k] for k in RIDER_MATS]

    # --- trail of P on the reachable arc [T0, T1], and the dashed rest
    OBJ["trail"] = poly_curve("trail", [], 0.42, MAT["trail"])
    OBJ["trail_pre"] = poly_curve("trail_pre", [], 0.42, MAT["trail2"])
    OBJ["dash"] = poly_curve("dash", [], 0.3, MAT["dash"])
    _compute_dashes()

    # --- labels (billboarded to the camera)
    for nm, (x, z) in (("A", (-7.0, g.AZ - 4.5)), ("B", (107.0, g.BZ + 1.5))):
        tx = text("lab" + nm, nm, 7.0, MAT["labAB"], fonts["math"])
        tx.location = (x, -4.0, z)
        billboard(tx, cam)
    OBJ["labP"] = text("labP", "P", 7.0, MAT["labP"], fonts["math"])
    billboard(OBJ["labP"], cam)

    # --- dimensions (setup scene only)
    dm = MAT["dims"]

    def dim_line(name, p, q, tick_dir):
        ln = cylinder(name, 0.22, 1.0, 8, dm)
        place_segment(ln, p, q)
        for i, e in enumerate((Vector(p), Vector(q))):
            tk = cylinder(f"{name}_t{i}", 0.22, 1.0, 8, dm)
            td = Vector(tick_dir) * 2.0
            place_segment(tk, e - td, e + td)

    dim_line("dimGround", (0, -6, 0.6), (100, -6, 0.6), (0, 0, 1))
    tg = text("dimGroundLab", "100 ft", 6.5, dm, fonts["sans"])
    tg.location = (50, -6, 5.0)
    billboard(tg, cam)
    dim_line("dimA", (-7, -4, 0.3), (-7, -4, g.AZ), (1, 0, 0))
    ta = text("dimALab", "100 ft", 6.5, dm, fonts["sans"], "RIGHT")
    ta.location = (-10, -4, 52)
    billboard(ta, cam)
    dim_line("dimB", (107, -4, 0.3), (107, -4, g.BZ), (1, 0, 0))
    tb = text("dimBLab", "50 ft", 6.5, dm, fonts["sans"], "LEFT")
    tb.location = (110, -4, 27)
    billboard(tb, cam)
    OBJ["ropelab"] = text("ropeLab", "rope: 120 ft", 6.0, MAT["ropelab"], fonts["sans"])
    billboard(OBJ["ropelab"], cam)

    # --- the vertical plane of the trees (final scene)
    def vplane(bm):
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    pl = new_mesh_obj("VPlane", vplane, MAT["plane"])
    pl.rotation_euler = (math.radians(90), 0, 0)
    pl.scale = (150, 132, 1)
    pl.location = (50, 0.05, 66)
    OBJ["vplane"] = pl


def _set_trail(ob, interval):
    """Rebuild the trail as the arc of the ellipse over the t-interval."""
    cu = ob.data
    cu.splines.clear()
    if interval is None or interval[1] - interval[0] < 1e-4:
        ob.hide_render = True
        return
    ob.hide_render = False
    a, b = interval
    m = max(2, int((b - a) / 0.004) + 1)
    sp = cu.splines.new("POLY")
    sp.points.add(m - 1)
    for i in range(m):
        x, z = g.ell(a + (b - a) * i / (m - 1))
        sp.points[i].co = (x, 0.0, z, 1.0)


def _set_dashes(upto):
    cu = OBJ["dash"].data
    cu.splines.clear()
    if upto is None:
        OBJ["dash"].hide_render = True
        return
    OBJ["dash"].hide_render = False
    for a, b in DASHES:
        if a >= upto:
            break
        b = min(b, upto)
        m = max(2, int((b - a) / 0.01))
        sp = cu.splines.new("POLY")
        sp.points.add(m - 1)
        for i in range(m):
            x, z = g.ell(a + (b - a) * i / (m - 1))
            sp.points[i].co = (x, 0.0, z, 1.0)


def apply(st):
    cam = OBJ["cam"]
    loc, tgt = Vector(st["cam_loc"]), Vector(st["cam_tgt"])
    cam.location = loc
    cam.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = st["lens"]

    px, pz = st["P"]
    P = Vector((px, 0, pz))
    A = Vector((g.AX, 0, g.AZ))
    B = Vector((g.BX, 0, g.BZ))
    place_segment(OBJ["ropeAP"], A, P)
    place_segment(OBJ["ropePB"], P, B)
    OBJ["markP"].location = P
    OBJ["rider"].location = P

    for m in OBJ["rider_mats"]:
        set_alpha(m, st["rider"])
    for o in OBJ["rider"].children:
        o.hide_render = st["rider"] < 1e-3
    for k in ("trunk", "leaf"):
        set_alpha(MAT[k], st["trees"])
    set_alpha(MAT["labAB"], st["labAB"])
    set_alpha(MAT["dims"], st["dims"])
    set_alpha(MAT["ropelab"], st["rope_label"])
    set_alpha(MAT["plane"], st["plane"])
    OBJ["vplane"].hide_render = st["plane"] < 1e-3

    # P label: along the inward normal (between the rope pieces) while the
    # rider hangs below P, along the outward normal once it is gone.
    uA = (A - P).normalized()
    uB = (B - P).normalized()
    inward = (uA + uB)
    inward = inward.normalized() if inward.length > 1e-6 else Vector((0, 0, 1))
    a_in, a_out = st["labP_in"], st["labP_out"]
    if a_out > a_in:
        OBJ["labP"].location = P - inward * 6.0
        set_alpha(MAT["labP"], a_out)
    else:
        OBJ["labP"].location = P + inward * 6.5
        set_alpha(MAT["labP"], a_in)

    # rope label under the middle of the longer piece
    mid = (P + B) / 2 if (B - P).length > (A - P).length else (A + P) / 2
    OBJ["ropelab"].location = mid + Vector((2.0, 0, -6.5))

    _set_trail(OBJ["trail"], st["solid_main"])
    _set_trail(OBJ["trail_pre"], st["solid_pre"])
    _set_dashes(st["dash_upto"])
