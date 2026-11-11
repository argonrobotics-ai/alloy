# SPDX-License-Identifier: Apache-2.0
"""Socket and property templates of the Blender nodes the look materials use (Blender 4.x defaults).

Generated once from a real Alloy Blender export; each input is (id, name, socket type, enabled, default).
Defaults are the values of a freshly added node. Edit only to follow a new Blender version.
"""

NODE_TEMPLATES: dict = {
    "ShaderNodeOutputMaterial": {
        "props": {"is_active_output": True, "target": "ALL"},
        "outputs": [],
        "inputs": [
            ("Surface", "Surface", "NodeSocketShader", True, None),
            ("Volume", "Volume", "NodeSocketShader", True, None),
            ("Displacement", "Displacement", "NodeSocketVector", True, (0, 0, 0)),
            ("Thickness", "Thickness", "NodeSocketFloat", True, 0),
        ],
    },
    "ShaderNodeBsdfPrincipled": {
        "props": {"distribution": "MULTI_GGX", "subsurface_method": "RANDOM_WALK"},
        "outputs": ['BSDF'],
        "inputs": [
            ("Base Color", "Base Color", "NodeSocketColor", True, (0.8, 0.8, 0.8, 1)),
            ("Metallic", "Metallic", "NodeSocketFloatFactor", True, 0),
            ("Roughness", "Roughness", "NodeSocketFloatFactor", True, 0.5),
            ("IOR", "IOR", "NodeSocketFloat", True, 1.5),
            ("Alpha", "Alpha", "NodeSocketFloatFactor", True, 1),
            ("Normal", "Normal", "NodeSocketVector", True, (0, 0, 0)),
            ("Weight", "Weight", "NodeSocketFloat", False, 0),
            ("Diffuse Roughness", "Diffuse Roughness", "NodeSocketFloatFactor", True, 0),
            ("Subsurface Weight", "Subsurface Weight", "NodeSocketFloatFactor", True, 0),
            ("Subsurface Radius", "Subsurface Radius", "NodeSocketVector", True, (1, 0.55, 0.3)),
            ("Subsurface Scale", "Subsurface Scale", "NodeSocketFloatDistance", True, 0.001),
            ("Subsurface IOR", "Subsurface IOR", "NodeSocketFloatFactor", False, 1.4),
            ("Subsurface Anisotropy", "Subsurface Anisotropy", "NodeSocketFloatFactor", True, 0),
            ("Specular IOR Level", "Specular IOR Level", "NodeSocketFloatFactor", True, 0.5),
            ("Specular Tint", "Specular Tint", "NodeSocketColor", True, (1, 1, 1, 1)),
            ("Anisotropic", "Anisotropic", "NodeSocketFloatFactor", True, 0),
            ("Anisotropic Rotation", "Anisotropic Rotation", "NodeSocketFloatFactor", True, 0),
            ("Tangent", "Tangent", "NodeSocketVector", True, (0, 0, 0)),
            ("Transmission Weight", "Transmission Weight", "NodeSocketFloatFactor", True, 0),
            ("Coat Weight", "Coat Weight", "NodeSocketFloatFactor", True, 0),
            ("Coat Roughness", "Coat Roughness", "NodeSocketFloatFactor", True, 0.05),
            ("Coat IOR", "Coat IOR", "NodeSocketFloat", True, 1.5),
            ("Coat Tint", "Coat Tint", "NodeSocketColor", True, (1, 1, 1, 1)),
            ("Coat Normal", "Coat Normal", "NodeSocketVector", True, (0, 0, 0)),
            ("Sheen Weight", "Sheen Weight", "NodeSocketFloatFactor", True, 0),
            ("Sheen Roughness", "Sheen Roughness", "NodeSocketFloatFactor", True, 0.5),
            ("Sheen Tint", "Sheen Tint", "NodeSocketColor", True, (1, 1, 1, 1)),
            ("Emission Color", "Emission Color", "NodeSocketColor", True, (1, 1, 1, 1)),
            ("Emission Strength", "Emission Strength", "NodeSocketFloat", True, 0),
            ("Thin Film Thickness", "Thin Film Thickness", "NodeSocketFloatWavelength", True, 0),
            ("Thin Film IOR", "Thin Film IOR", "NodeSocketFloat", True, 1.33),
        ],
    },
    "ShaderNodeTexCoord": {
        "props": {"from_instancer": False},
        "outputs": ['Generated', 'Normal', 'UV', 'Object', 'Camera', 'Window', 'Reflection'],
        "inputs": [
        ],
    },
    "ShaderNodeAttribute": {
        "props": {"attribute_type": "GEOMETRY", "attribute_name": "rest_p"},
        "outputs": ['Color', 'Vector', 'Fac', 'Alpha'],
        "inputs": [
        ],
    },
    "ShaderNodeMix": {
        "props": {"data_type": "FLOAT", "factor_mode": "UNIFORM", "blend_type": "MIX", "clamp_factor": True, "clamp_result": False},
        "outputs": ['Result_Float', 'Result_Vector', 'Result_Color', 'Result_Rotation'],
        "inputs": [
            ("Factor_Float", "Factor", "NodeSocketFloatFactor", True, 0.5),
            ("Factor_Vector", "Factor", "NodeSocketVectorFactor", False, (0.5, 0.5, 0.5)),
            ("A_Float", "A", "NodeSocketFloat", False, 0),
            ("B_Float", "B", "NodeSocketFloat", False, 0),
            ("A_Vector", "A", "NodeSocketVector", True, (0, 0, 0)),
            ("B_Vector", "B", "NodeSocketVector", True, (0, 0, 0)),
            ("A_Color", "A", "NodeSocketColor", False, (0.5, 0.5, 0.5, 1)),
            ("B_Color", "B", "NodeSocketColor", False, (0.5, 0.5, 0.5, 1)),
            ("A_Rotation", "A", "NodeSocketRotation", False, (0, 0, 0)),
            ("B_Rotation", "B", "NodeSocketRotation", False, (0, 0, 0)),
        ],
    },
    "ShaderNodeTexNoise": {
        "props": {"noise_dimensions": "3D", "noise_type": "FBM", "normalize": True},
        "texture_mapping": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "vector_type": "POINT"},
        "outputs": ['Fac', 'Color'],
        "inputs": [
            ("Vector", "Vector", "NodeSocketVector", True, (0, 0, 0)),
            ("W", "W", "NodeSocketFloat", False, 0),
            ("Scale", "Scale", "NodeSocketFloat", True, 5),
            ("Detail", "Detail", "NodeSocketFloat", True, 2),
            ("Roughness", "Roughness", "NodeSocketFloatFactor", True, 0.5),
            ("Lacunarity", "Lacunarity", "NodeSocketFloat", True, 2),
            ("Offset", "Offset", "NodeSocketFloat", False, 0),
            ("Gain", "Gain", "NodeSocketFloat", False, 1),
            ("Distortion", "Distortion", "NodeSocketFloat", True, 0),
        ],
    },
    "ShaderNodeObjectInfo": {
        "props": {},
        "outputs": ['Location', 'Color', 'Alpha', 'Object_Index', 'Material_Index', 'Random'],
        "inputs": [
        ],
    },
    "ShaderNodeMath": {
        "props": {"operation": "ADD", "use_clamp": False},
        "outputs": ['Value'],
        "inputs": [
            ("Value", "Value", "NodeSocketFloat", True, 0.5),
            ("Value_001", "Value", "NodeSocketFloat", True, 0.5),
            ("Value_002", "Value", "NodeSocketFloat", True, 0.5),
        ],
    },
    "ShaderNodeVectorMath": {
        "props": {"operation": "ADD"},
        "outputs": ['Vector', 'Value'],
        "inputs": [
            ("Vector", "Vector", "NodeSocketVector", True, (0, 0, 0)),
            ("Vector_001", "Vector", "NodeSocketVector", False, (0, 0, 0)),
            ("Vector_002", "Vector", "NodeSocketVector", False, (0, 0, 0)),
            ("Scale", "Scale", "NodeSocketFloat", True, 1),
        ],
    },
    "ShaderNodeBevel": {
        "props": {"samples": 8},
        "outputs": ['Normal'],
        "inputs": [
            ("Radius", "Radius", "NodeSocketFloat", True, 0.05),
            ("Normal", "Normal", "NodeSocketVector", True, (0, 0, 0)),
        ],
    },
    "ShaderNodeBump": {
        "props": {"invert": False},
        "outputs": ['Normal'],
        "inputs": [
            ("Strength", "Strength", "NodeSocketFloatFactor", True, 1),
            ("Distance", "Distance", "NodeSocketFloat", True, 1),
            ("Filter Width", "Filter Width", "NodeSocketFloat", True, 0.1),
            ("Height", "Height", "NodeSocketFloat", True, 1),
            ("Normal", "Normal", "NodeSocketVector", True, (0, 0, 0)),
        ],
    },
    "ShaderNodeSeparateXYZ": {
        "props": {},
        "outputs": ['X', 'Y', 'Z'],
        "inputs": [
            ("Vector", "Vector", "NodeSocketVector", True, (0, 0, 0)),
        ],
    },
    "ShaderNodeCombineXYZ": {
        "props": {},
        "outputs": ['Vector'],
        "inputs": [
            ("X", "X", "NodeSocketFloat", True, 0),
            ("Y", "Y", "NodeSocketFloat", True, 0),
            ("Z", "Z", "NodeSocketFloat", True, 0),
        ],
    },
    "ShaderNodeTexImage": {
        "props": {"projection": "FLAT", "interpolation": "Cubic", "projection_blend": 0.0, "extension": "REPEAT"},
        "texture_mapping": {"translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "vector_type": "POINT"},
        "outputs": ['Color', 'Alpha'],
        "inputs": [
            ("Vector", "Vector", "NodeSocketVector", True, (0, 0, 0)),
        ],
    },
}
