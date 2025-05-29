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


def vertex_setter(density: int):
    #bounding_indexes = [closest to center, max cords, min cords,]
    center_point = []
    bounding_box= []
    shrink_wrap = []
    bounding_indexes = []
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
    

extract_shape("mesh_7.obj")
print("Done")
vertex_setter(6)
