import bpy
import bmesh
from mathutils import Vector

class CustomClass(bpy.types.Operator):
    bl_idname = "tools.analyze"
    bl_label = "Analyze"
    bl_options = {'REGISTER', 'UNDO'}
    