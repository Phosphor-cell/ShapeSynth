import os
import math
import matplotlib.pyplot as ply
import numpy as np
import pandas as pd


"""
    Software Based Retopology, Product of ShapeSynth
    Author: Phosphor-cell(Dev Bhatt)
    Version: 0.0.0A
    Date:2025-05-26
"""

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
            
            if "elbow" in lines and not "off" in lines:
                groupOn = True
            if groupOn and "f " in lines:
                face_lines = lines.strip("f").strip("\n").split("//")[0]
                face_lines = face_lines.strip(" ")
                temp_groups.append(np.int64(face_lines))
                FACES = np.array(temp_groups, dtype=np.int64)
    temp_groups = set(FACES)
    FACES = np.array(list(temp_groups), dtype=np.int64)

def vertex_setter(vertstoFaces):
    #VIew Points = [Highest, Lowest, Right Most, Left Most, Furthest, Closest]
    view_points = []
    repoints = []
    for indexes in FACES:
        for i in range (0,3):
            for j in range(0,3):
                if not len(view_points) < i or VERTEXES[indexes-1][i] > view_points[i][j]:
                    view_points.insert(i, VERTEXES[indexes-1])
                elif not len(view_points) < (4+(i%3)) or VERTEXES[indexes-1][i] < view_points[4+(i%3)][j]:
                    view_points.insert(4+(i%3), VERTEXES[indexes-1])
    print(view_points)

extract_shape("mesh_1.obj")
print("Done")
vertex_setter(FACES)
