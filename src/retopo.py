import os
import math
import matplotlib.pyplot as ply
import numpy as np
#! import pandas as pd


#* Software Based Retopology, Product of ShapeSynth
#* Author: Phosphor-cell(Dev Bhatt)
#* Version: 0.0.0A
#* Date:2025-05-26


VERTEXES = []
FACES = []
GROUPS = []
shrink_wrap = []
bounding_box = []


def extract_shape(filepath: str):
    groupOn = False
    temp_groups = []
    global FACES, VERTEXES
    with open(filepath, "r") as file:
        for lines in file:
            if "v " in lines:
                line = lines.strip().strip("v").split()
                line = np.float32(np.array(line))
                VERTEXES.append(line[:3])
            
            if "elbow" in lines and not "off" in lines or 'g elbow' in lines:
                groupOn = True
            elif "g off" in lines:
                groupOn = False
            if groupOn and "f " in lines:
                face_lines = lines.strip("f").strip("\n").split("//")[0]
                face_lines = face_lines.strip(" ")
                temp_groups.append(np.int64(face_lines))
                FACES = np.array(temp_groups, dtype=np.int64)
    temp_groups = set(FACES)
    FACES = np.array(list(temp_groups), dtype=np.int64)


def vertex_setter(size_num: int):
    #bounding_indexes = [closest to center, max cords(x,y,z), min cords(x,y,z),]
    center_point = []
    global bounding_box
    global shrink_wrap
    bounding_indexes = []
    cord_spaces = []
    for i in FACES:
        bounding_box.append(VERTEXES[i-1])

    bounding_box = np.array(bounding_box)
    center_point = bounding_box.mean(axis=0)
    bounding_indexes.extend([
        np.sum((bounding_box-center_point)**2, axis=1, keepdims=True).argmin(axis=0), 
        bounding_box.argmax(axis=0),
        bounding_box.argmin(axis=0),
        ]
    )
    
    for fst_index in range(len(bounding_indexes)):
        for i in range(len(bounding_indexes[fst_index])):
            shrink_wrap.append(bounding_indexes[fst_index][i])





def create_circles(verts_num: int):
    global bounding_box
    x_max_min = [bounding_box[1][0], bounding_box[2][0]]
    y_max_min = [bounding_box[1][1], bounding_box[2][1]]
    z_max_min = [bounding_box[1][2], bounding_box[2][2]]
    
    z_val = np.random.uniform(z_max_min[1], z_max_min[0])
    x_val = np.random.uniform(x_max_min[1], x_max_min[0])
    y_val = np.random.uniform(y_max_min[1], y_max_min[0])
    
    for i in range(2):
        for j in range(2):
            cord_plane_z = np.array([x_max_min[i], y_max_min[j], z_val])
            cord_plane_x = np.array([x_val, y_max_min[i], z_max_min[j]])
            cord_plane_y = np.array([x_max_min[i], y_val, z_max_min[j]])
    if not np.any(np.all(bounding_box == cord_plane_x, axis=1)):
        bounding_box = np.vstack([bounding_box, cord_plane_x])
    
    if not np.any(np.all(bounding_box == cord_plane_y, axis=1)):
        bounding_box = np.vstack([bounding_box, cord_plane_y])
    
    if not np.any(np.all(bounding_box == cord_plane_z, axis=1)):
        bounding_box = np.vstack([bounding_box, cord_plane_z])


def connect_faces():
    global bounding_box
    global GROUPS
    
    x_vals = 1

def create_circle_helper():
    global bounding_box
    global shrink_wrap



def draw_model(filepath:str):
    global bounding_box

    with open(filepath, 'w') as file:
        for i in range(len(bounding_box)):
            file.write(f"v ")
            for j in range(len(bounding_box[i])):
                file.write(f"{bounding_box[i][j]} ")
            file.write("\n")
        val_f = 1
        for i in range(len(bounding_box)):
            file.write(f"f ")
            for j in range(3):
                file.write(f"{val_f+j} ")
            val_f += 3
            file.write(f"\n")
extract_shape("mesh_7.obj")
print("Done")
vertex_setter(3)
create_circles()
print(bounding_box)
draw_model("mesh_12.obj")