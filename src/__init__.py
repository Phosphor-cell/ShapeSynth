import bpy
import numpy


class application_remover(bpy.types.Operator):
    def execute(self, context):
        unregister()


class ShapeSynth(bpy.types.Panel):
    #Setup within the viewport rather than properties tab/panel
    bl_label = "ShapeSynth"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ShapeSynth" #Place on side bar (if it exists it will be placed there)
    
    def draw(self, context):
        layout = self.layout
        layout.label(text = "Analysis")
        row = layout.row()
        row.operator("render.render", text = "Pre-Alpha")
        layout.label(text = "Retopology")
        
        row.operator(application_remover, text="Remove Application")

def register():
    bpy.utils.register_class(ShapeSynth)

def unregister():
    bpy.utils.unregister_class(ShapeSynth)

if __name__ == "__main__":
    register()