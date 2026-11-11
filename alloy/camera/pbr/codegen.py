# SPDX-License-Identifier: Apache-2.0
"""Blender material node tree (scene pack v2 ``materials.<name>.tree``) -> Warp source code.

Per material and per LINKED Principled input (except Normal / Coat Normal) one ``@wp.func``
``mat_<name>__<Input>(p, n, uv, col, rnd, <image arrays>) -> float | wp.vec3``, and, when a Bump node feeds the
normal, one height function ``mat_<name>__height``. ``p`` = object-space position, ``n`` = object-space shading
normal, ``uv`` = (u, v, 0) of the mesh's UV layer interpolated at the hit (Blender convention: origin bottom-left;
zeros on meshes without UVs), ``col`` = the mesh's per-corner colour attribute interpolated at the hit (zeros
without), ``rnd`` = Object Info Random. Supported nodes: Texture Coordinate (UV, Object, Normal), UV Map,
Attribute (``UVMap`` or a UV layer name = ``uv``, the scene's colour attribute = ``col``; rest_p, rest_n, has_rest and
other names evaluate to 0), Object Info (Random), Mix (vector MIX; colour MIX / MULTIPLY, uniform factor), Math,
Vector Math, Separate / Combine XYZ, Noise Texture (3D fBM), Image Texture (flat, cubic or linear, repeat; an unlinked
Vector input reads the UV, as in Blender; an sRGB image is decoded after interpolation), Bump, Bevel. Other nodes
raise NotImplementedError naming the node.

Example::

    src, meta = gen_material("steel", pack["materials"]["steel"], pack["textures"], uv_names=("UVMap",))
    # src: Warp functions calling alloy.camera.pbr.nodes as ``nd``; meta: function names, images, bump, bevel
"""
import re

FLOAT_OUTS = {"Fac", "Value", "Random", "Alpha", "Index", "Fraction"}
IMAGE_SAMPLERS = {"Cubic": "image_cubic_repeat", "Linear": "image_linear_repeat"}  # TexImage interpolation -> nd func
COLOR_BLENDS = {"MIX": 0, "MULTIPLY": 1}  # Mix (RGBA) blend_type -> nd.mix_color's blend


def _lit(v):
    if isinstance(v, (list, tuple)):
        return "wp.vec3(%r, %r, %r)" % (float(v[0]), float(v[1]), float(v[2]))
    return repr(float(v))


class TreeGen:
    def __init__(self, tree, textures, uv_names=(), col_names=()):
        self.uv_names = {"UVMap", *uv_names}
        self.col_names = set(col_names)
        self.nodes = {n["name"]: n for n in tree["nodes"]}
        self.links = {}
        for lk in tree["links"]:
            if not lk.get("muted"):
                self.links[(lk["to"], lk["to_id"])] = (lk["from"], lk["from_id"])
        self.textures = textures
        self.lines = []
        self.cache = {}
        self.images = []  # (arg name, texture key)
        self.n = 0

    # ---- helpers -----------------------------------------------------------------------------------
    def _new(self):
        self.n += 1
        return "v%d" % self.n

    def _emit(self, s):
        self.lines.append("    " + s)

    @staticmethod
    def out_kind(node, out_id):
        return "f" if out_id in FLOAT_OUTS else "v"

    def _sock(self, node, sid):
        return next(i for i in node["inputs"] if i["id"] == sid)

    def inp(self, node, sid, want):
        """Expression for input `sid` as kind `want` ('f' or 'v')."""
        lk = self.links.get((node["name"], sid))
        if lk is None:
            val = self._sock(node, sid)["value"]
            if want == "f":
                return _lit(val if not isinstance(val, list) else sum(val[:3]) / 3.0)
            return _lit(val if isinstance(val, list) else [val] * 3)
        src, kind = self.expr(*lk)
        return self.convert(src, kind, want)

    @staticmethod
    def convert(src, kind, want):
        if kind == want:
            return src
        if want == "v":
            return "wp.vec3(%s, %s, %s)" % ((src,) * 3)
        return "((%s[0] + %s[1] + %s[2]) / 3.0)" % ((src,) * 3)  # Cycles: vector -> float is the average

    # ---- node emitters -----------------------------------------------------------------------------
    def expr(self, name, out_id):
        key = (name, out_id)
        if key in self.cache:
            return self.cache[key]
        node = self.nodes[name]
        t = node["type"]
        fn = getattr(self, "_" + t, None)
        if fn is None:
            raise NotImplementedError("node type %s (%s)" % (t, name))
        res = fn(node, out_id)
        self.cache[key] = res
        return res

    def _ShaderNodeTexCoord(self, node, out_id):
        if out_id == "Object":
            return "p", "v"
        if out_id == "Normal":
            return "n", "v"
        if out_id == "UV":
            return "uv", "v"
        raise NotImplementedError("TexCoord %s" % out_id)

    def _ShaderNodeUVMap(self, node, out_id):
        name = node["props"].get("uv_map", "")
        if name and name not in self.uv_names:
            raise NotImplementedError("UV Map node names UV layer %r; the mesh has %s" % (name, sorted(self.uv_names)))
        if out_id != "UV":
            raise NotImplementedError("UVMap %s" % out_id)
        return "uv", "v"

    def _ShaderNodeAttribute(self, node, out_id):
        name = node["props"].get("attribute_name", "")
        if name in self.uv_names or name in self.col_names:  # a mesh UV layer / the scene's colour attribute
            vec = "uv" if name in self.uv_names else "col"
            if out_id == "Fac":
                return self.convert(vec, "v", "f"), "f"  # Cycles: average of the components
            return ("1.0", "f") if out_id == "Alpha" else (vec, "v")
        # rest_p / rest_n / has_rest and unknown names do not exist on the meshes: Cycles returns 0
        return ("0.0", "f") if out_id in FLOAT_OUTS else ("wp.vec3(0.0, 0.0, 0.0)", "v")

    def _ShaderNodeObjectInfo(self, node, out_id):
        if out_id != "Random":
            raise NotImplementedError("ObjectInfo %s" % out_id)
        return "rnd", "f"

    def _ShaderNodeMix(self, node, out_id):
        pr = node["props"]
        uniform = pr.get("factor_mode", "UNIFORM") == "UNIFORM"
        if pr["data_type"] == "RGBA" and uniform and pr["blend_type"] in COLOR_BLENDS:
            return self._mix_color(node)
        if pr["data_type"] != "VECTOR" or pr["blend_type"] != "MIX":
            raise NotImplementedError("Mix %s" % node["props"])
        fac = self.inp(node, "Factor_Float", "f")
        a = self.inp(node, "A_Vector", "v")
        b = self.inp(node, "B_Vector", "v")
        v = self._new()
        self._emit("%s = nd.mix_vector(%s, %s, %s, %d)" % (v, fac, a, b, int(node["props"]["clamp_factor"])))
        return v, "v"

    def _mix_color(self, node):
        """Mix, RGBA, uniform factor, MIX or MULTIPLY (e.g. a glTF baseColorFactor times its texture)."""
        pr = node["props"]
        fac = self.inp(node, "Factor_Float", "f")
        a = self.inp(node, "A_Color", "v")
        b = self.inp(node, "B_Color", "v")
        v = self._new()
        flags = (COLOR_BLENDS[pr["blend_type"]], int(pr["clamp_factor"]), int(pr.get("clamp_result", False)))
        self._emit("%s = nd.mix_color(%s, %s, %s, %d, %d, %d)" % (v, fac, a, b, *flags))
        return v, "v"

    def _ShaderNodeMath(self, node, out_id):
        op = node["props"]["operation"]
        enum = {"FRACT": "FRACTION"}.get(op, op)
        enum = "INV_SQRT" if enum == "INVERSE_SQRT" else enum
        a, b, c = (self.inp(node, s, "f") for s in ("Value", "Value_001", "Value_002"))
        v = self._new()
        self._emit("%s = nd.svm_math(nd.MATH_%s, %s, %s, %s)" % (v, enum, a, b, c))
        if node["props"].get("use_clamp"):
            self._emit("%s = nd.saturatef(%s)" % (v, v))
        return v, "f"

    def _ShaderNodeMapRange(self, node, out_id):
        from .nodes.mathnodes import MAP_RANGE_KINDS
        pr = node["props"]
        if pr.get("data_type", "FLOAT") != "FLOAT" or pr.get("interpolation_type", "LINEAR") not in MAP_RANGE_KINDS:
            raise NotImplementedError("Map Range %s" % pr)
        args = [self.inp(node, s, "f") for s in ("Value", "From Min", "From Max", "To Min", "To Max", "Steps")]
        v = self._new()
        kind = MAP_RANGE_KINDS[pr.get("interpolation_type", "LINEAR")]
        self._emit("%s = nd.map_range(%s, %d, %d)" % (v, ", ".join(args), kind, int(bool(pr.get("clamp", True)))))
        return v, "f"

    def _ShaderNodeVectorMath(self, node, out_id):
        op = node["props"]["operation"]
        enum = {"CROSS_PRODUCT": "CROSS", "DOT_PRODUCT": "DOT", "MINIMUM": "MIN", "MAXIMUM": "MAX", "SINE": "SIN",
                "COSINE": "COS", "TANGENT": "TAN"}.get(op, op)
        a = self.inp(node, "Vector", "v")
        b = self.inp(node, "Vector_001", "v")
        c = self.inp(node, "Vector_002", "v")
        s = self.inp(node, "Scale", "f")
        val, vec = self._new(), self._new()
        self._emit("%s, %s = nd.svm_vector_math(nd.VMATH_%s, %s, %s, %s, %s)" % (val, vec, enum, a, b, c, s))
        self.cache[(node["name"], "Value")] = (val, "f")
        self.cache[(node["name"], "Vector")] = (vec, "v")
        return self.cache[(node["name"], out_id)]

    def _ShaderNodeSeparateXYZ(self, node, out_id):
        v = self.inp(node, "Vector", "v")
        return "(%s)[%d]" % (v, "XYZ".index(out_id)), "f"

    def _ShaderNodeCombineXYZ(self, node, out_id):
        x, y, z = (self.inp(node, s, "f") for s in "XYZ")
        v = self._new()
        self._emit("%s = wp.vec3(%s, %s, %s)" % (v, x, y, z))
        return v, "v"

    def _ShaderNodeTexNoise(self, node, out_id):
        pr = node["props"]
        if pr["noise_dimensions"] != "3D" or pr["noise_type"] != "FBM":
            raise NotImplementedError("Noise %s" % pr)
        args = [self.inp(node, "Vector", "v")] + [self.inp(node, s, "f") for s in
                                                  ("Scale", "Detail", "Roughness", "Lacunarity", "Distortion")]
        fac, col = self._new(), self._new()
        self._emit("%s, %s = nd.tex_noise_3d(%s, %s, %s, %s, %s, %s, %d)" % (fac, col, *args, int(pr["normalize"])))
        self.cache[(node["name"], "Fac")] = (fac, "f")
        self.cache[(node["name"], "Color")] = (col, "v")
        return self.cache[(node["name"], out_id)]

    def _ShaderNodeTexImage(self, node, out_id):
        pr = node["props"]
        if pr["projection"] != "FLAT" or pr["interpolation"] not in IMAGE_SAMPLERS or pr["extension"] != "REPEAT":
            raise NotImplementedError("TexImage %s" % pr)
        if out_id != "Color":
            raise NotImplementedError("TexImage %s" % out_id)
        key = node["image"]
        if key not in [k for _, k in self.images]:
            self.images.append(("img_%d" % len(self.images), key))
        arg = next(a for a, k in self.images if k == key)
        co = "uv" if (node["name"], "Vector") not in self.links else self.inp(node, "Vector", "v")  # Blender: UV
        v, c = self._new(), self._new()
        self._emit("%s = %s" % (c, co))
        self._emit("%s = nd.%s(%s, %s[0], %s[1])" % (v, IMAGE_SAMPLERS[pr["interpolation"]], arg, c, c))
        if self.textures[key]["colorspace"] == "sRGB":
            self._emit("%s = nd.srgb_to_linear3(%s)" % (v, v))
        return v, "v"


def principled_inputs(tree):
    pr = next(n for n in tree["nodes"] if n["type"] == "ShaderNodeBsdfPrincipled")
    return pr, [i["id"] for i in pr["inputs"] if i["linked"]]


def gen_material(name, mat, textures, uv_names=(), col_names=()):
    """(Warp source, meta) of one material; ``uv_names`` / ``col_names``: attribute names that read ``uv`` / ``col``."""
    tree = mat["tree"]
    pr, linked = principled_inputs(tree)
    funcs = {}
    code = []
    meta = {"functions": {}, "images": [], "bump": None, "bevel": None}
    targets = [(i, (pr["name"], i)) for i in linked if i not in ("Normal", "Coat Normal")]
    nodes = {n["name"]: n for n in tree["nodes"]}
    bump = next((n for n in tree["nodes"] if n["type"] == "ShaderNodeBump"), None)
    bevel = next((n for n in tree["nodes"] if n["type"] == "ShaderNodeBevel"), None)
    allimgs = []
    outs = []
    for label, (node_name, sid) in targets:
        g = TreeGen(tree, textures, uv_names, col_names)
        src, kind = g.expr(*g.links[(node_name, sid)])
        want = "f" if g._sock(pr, sid)["type"].startswith("NodeSocketFloat") else "v"
        outs.append((label.replace(" ", "_"), g, g.convert(src, kind, want), "float" if want == "f" else "wp.vec3"))
    if bump is not None and (bump["name"], "Height") in TreeGen(tree, textures, uv_names, col_names).links:
        g = TreeGen(tree, textures, uv_names, col_names)
        src, kind = g.expr(*g.links[(bump["name"], "Height")])
        outs.append(("height", g, g.convert(src, kind, "f"), "float"))
        sv = {i["id"]: i["value"] for i in bump["inputs"]}
        assert not any((bump["name"], k) in g.links for k in ("Strength", "Distance", "Filter Width")), "linked bump params"
        meta["bump"] = {"strength": sv["Strength"], "distance": sv["Distance"], "filter_width": sv["Filter Width"],
                        "invert": bool(bump["props"].get("invert", False)),
                        "normal_from_bevel": (bump["name"], "Normal") in g.links}
    if bevel is not None:
        bv = {i["id"]: i["value"] for i in bevel["inputs"]}
        meta["bevel"] = {"radius": bv["Radius"], "samples": bevel["props"]["samples"]}
    safe = re.sub(r"\W", "_", name)
    for label, g, ret, rtype in outs:
        args = ["p: wp.vec3", "n: wp.vec3", "uv: wp.vec3", "col: wp.vec3", "rnd: float"] + ["%s: wp.array3d(dtype=float)" % a for a, _ in g.images]
        fname = "mat_%s__%s" % (safe, label)
        code.append("@wp.func\ndef %s(%s) -> %s:\n%s\n    return %s\n" % (fname, ", ".join(args), rtype, "\n".join(g.lines), ret))
        meta["functions"][label] = {"name": fname, "returns": "float" if rtype == "float" else "vec3",
                                    "images": [k for _, k in g.images]}
        for _, k in g.images:
            if k not in allimgs:
                allimgs.append(k)
    meta["images"] = [{"key": k, "file": textures[k]["file"], "srgb": textures[k]["colorspace"] == "sRGB"} for k in allimgs]
    return "\n\n".join(code), meta
