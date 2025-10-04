import os
import torch
import numpy as np
import trimesh
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm


MESH_PATH = r"G:\RefinedV2\Left Arm_paddlegirl.obj"


base_name, _ = os.path.splitext(os.path.basename(MESH_PATH))



torch.set_default_device('cuda')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESOLUTION = 128
BOUNDING_BOX = [1.1,-1.1]
PN_M = False

x_cords = torch.linspace(BOUNDING_BOX[0], BOUNDING_BOX[1], RESOLUTION, device=DEVICE, pin_memory=PN_M)
y_cords = torch.linspace(BOUNDING_BOX[0], BOUNDING_BOX[1], RESOLUTION, device=DEVICE, pin_memory=PN_M)
z_cords = torch.linspace(BOUNDING_BOX[0], BOUNDING_BOX[1], RESOLUTION, device=DEVICE, pin_memory=PN_M)

x_cords, y_cords, z_cords = torch.meshgrid(x_cords, y_cords, z_cords, indexing='ij')

query_points = torch.stack([torch.ravel(x_cords),torch.ravel(y_cords),torch.ravel(z_cords)], dim=1)

dictionary_parts = {}
parts_id = {}


edge_case = set()

dict_edge_case = {}


def UDFs_signing(query_points, p1, p2, p3):
    p2_to_p1 = p2-p1 #Value B-A ... AB
    p3_to_p1 = p1-p3 #Value C-A ... AC
    p3_to_p2 = p2-p3 #Value C-B ... BC
    
    query_to_p1 = query_points-p1
    query_to_p2 = query_points-p2
    query_to_p3 = query_points-p3
    
    numerator = query_to_p1 @ p2_to_p1
    denominator = p2_to_p1 @ p2_to_p1
    
    t = numerator/denominator
    
    numerator2 = query_to_p2 @ p2
    denominator2 = p3_to_p2 @ p3_to_p2
    
    u = numerator2/denominator2
    
    numerator3 = query_to_p2 @ p3
    denominator3 = p3_to_p1 @ p3_to_p1
    
    v = numerator3/denominator3
    
    
    #v = v.tolist()
    #u = u.tolist()
    #t = t.tolist()
    
    #dist_list = [v,u,t]
    dist_list = torch.stack([t,u,v], dim=0).cuda()
    #print(dist_list.shape)
    final_dist, idx = torch.min(dist_list, dim=0)
    
    #print(final_dist)
    #print(final_dist.shape)
    #Trianguate
    normal = torch.cross(p2_to_p1, p3_to_p1, dim=0) #Perpendicular Vector
    
    
    
    #print(query_points.shape[0])
    #print(p2_to_p1.shape)
    #print(p3_to_p1.shape)
    
    #Barycentric Calculations
    
    
    
    #surface_area_ABC = #torch.mul(1/2, torch.mul(p2_to_p1, p3_to_p1).abs())
    surface_area_PBC = torch.cross(query_to_p2, query_to_p3, dim=1)#torch.mul(1/2, torch.mul(query_to_p2, query_to_p3).abs())
    surface_area_PCA = torch.cross(query_to_p1,query_to_p3, dim=1)#torch.mul(1/2, torch.mul(query_to_p1, query_to_p3).abs())
    surface_area_PAB = torch.cross(query_to_p1,query_to_p2,dim=1)#torch.mul(1/2, torch.mul(query_to_p1, query_to_p2).abs())
    
    alpha_val = torch.div(surface_area_PBC, normal)
    beta_val = torch.div(surface_area_PCA, normal)
    gamma_val = torch.div(surface_area_PAB, normal)
    
    
    #print(surface_area_ABC)
    #print(surface_area_PBC)
    
    surface_distance = torch.mul(alpha_val, p1) + torch.mul(beta_val, p2) + torch.mul(gamma_val, p3)
    surface_distance = surface_distance.abs()
    #print(surface_distance)
    
    return surface_distance

def sampleing_mesh(mesh_path):
    current = "default"
    
    with open(mesh_path, 'r') as file:
        counter = 0
        for lines in file:
            line = lines.strip()
            if not line:
                continue
            if line.startswith('g') and "off" not in line:
                parts = line.split()
                face_verts = []
                current = parts[1]
                if current not in dictionary_parts:
                    dictionary_parts[current] = []
                    parts_id[current] = counter
                    counter += 1
                
            if line.startswith('f'):
                parts = line.split()
                parts = parts[1:]
                face_verts = []
                #for faces in parts:
                face_verts.extend(item.split('//')[0] for item in parts)
                #print(face_verts)
                dictionary_parts[current].append(list(map(int, face_verts)))
    
        



def transformations(mesh):
    
    rand_deg = np.random.randint(low=0, high=360)#, size=(1,), dtype=torch.int16)
    angle = np.deg2rad(rand_deg)
    
    rand_axis_x = np.random.randint(low=0, high=3)#, size=(1,),# dtype=torch.int16, device=DEVICE)
    rand_axis_y = np.random.randint(low=0, high=3)#, size=(1,),# dtype=torch.int16, device=DEVICE)
    rand_axis_z = np.random.randint(low=0, high=3)#,# size=(1,),# dtype=torch.int16, device=DEVICE)
    
    #rand_axis_x = rand_axis_x.item()
    #rand_axis_y = rand_axis_y.item()
    #rand_axis_z = rand_axis_z.item()
    #angle = angle.item()
    
    #CREATE CUSTOM ROTATION MATRIX AND TAKE IN VERTICIES INSTEAD TO FORCE GPU AND IMPROVE
    
    #print(angle)
    #print(rand_axis_x)
    #print(rand_axis_y)
    #print(rand_axis_z)
    
    axis = [rand_axis_x, rand_axis_y, rand_axis_z]
    
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
    
    mesh.apply_transform(rotation_matrix)
    
    return mesh


def preprocess(args):
    
    
    mesh_path, ver = args
    
    mesh = trimesh.load(mesh_path)
    
    mesh = transformations(mesh)
    
    
    
    vertices = mesh.vertices
    faces = mesh.faces
        
    vertices_gpu = torch.tensor(vertices, dtype=torch.float64)
    faces_gpu = torch.tensor(faces, dtype=torch.long)
    
    triangulation = vertices_gpu[faces_gpu]
    
    max_cords, max_idx = torch.max(vertices_gpu, dim=0)
    min_cords, min_idx = torch.min(vertices_gpu, dim=0)
    
    center = torch.sum(torch.add(max_cords, min_cords)/torch.tensor(2), dtype=torch.float64)
    
    vertices_gpu = vertices_gpu - center
    
    sampleing_mesh(mesh_path)

    for p1,p2,p3 in triangulation:
        signed_points = UDFs_signing(query_points, p1, p2, p3)

    for key in parts_id.keys():
        edge_case.clear()

        dict_edge_case[key] = []
        if key not in dictionary_parts:
            continue
        
        if len(dictionary_parts[key]) == 0:
            continue
        
        for vals in dictionary_parts[key]:
            if len(vals) == 3:
                f1,f2,f3 = vals
                edge_case.add(tuple(sorted((f1,f2))))
                edge_case.add(tuple(sorted((f2,f3))))
                edge_case.add(tuple(sorted((f3,f1))))
            
            elif len(vals) == 4:
                f1,f2,f3,f4 = vals
                edge_case.add(tuple(sorted((f1,f2))))
                edge_case.add(tuple(sorted((f2,f3))))
                edge_case.add(tuple(sorted((f3,f4))))
                edge_case.add(tuple(sorted((f1,f4))))
            #print(edge_case)
            edges = torch.tensor(list(edge_case)).t().contiguous()
            edge_cases = torch.cat([edges, edges.flip(0)], dim=1)
            print(edge_cases)
        dict_edge_case[key].append(edge_cases)
        #print(dict_edge_case)



    print(f"Resolution: {RESOLUTION}x{RESOLUTION}x{RESOLUTION}")
    print(f"Total Number of Query Points Generated: {len(query_points)} per axis")
    print(f"Shape of Final 'query_points': {query_points.shape}")
    print(f"\nFirst 5 Query: {query_points[:5]}")
    print(f"\nLast 5 Query: {query_points[-5:]}")
    
    np.savez_compressed(fr"G:\RefinedV2\data\{base_name}_{ver}.npz", query_points = query_points.cpu().numpy(), parts = parts_id, faces_per_part = dictionary_parts, edges = dict_edge_case, faces = faces_gpu.cpu().numpy(), vertices = vertices_gpu.cpu().numpy(), surface_distance = signed_points.cpu().numpy(), resolution = RESOLUTION)



if __name__ == "__main__":
    
    NUM_RUNS = 5
    
    num_processes = min(NUM_RUNS, cpu_count() - 1 if cpu_count() > 1 else 1)
    
    start_time = time.time()
    
    print(f"Starting preprocessing for {NUM_RUNS} augmentations using {num_processes} processes...")
    
    pool_args = [(MESH_PATH, i) for i in range(NUM_RUNS)]
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(preprocess, pool_args), total=NUM_RUNS))
        
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    
    print("\n--- Preprocessing Complete ---\n")
    print(f"Saved SDFs DATA\nTime Elapsed: {elapsed_time}")
