import os
import trimesh
import numpy as np
from tqdm import tqdm

DATA_DIR ="3D_Topology/Data"
TARGET_DIR = "3D_Topology/target"

if __name__ == "__main__":
    # --- Configuration ---
    # The number of points to sample for each shape
    NUM_POINTS_PER_SHAPE = 16384

    # --- Setup ---
    os.makedirs(TARGET_DIR, exist_ok=True)
    mesh_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.obj')]
    print(f"Found {len(mesh_files)} models to process.")

    # --- Main Loop ---
    for filename in tqdm(mesh_files, desc="Processing Meshes for SDF"):
        source_path = os.path.join(DATA_DIR, filename)
        
        try:
            mesh = trimesh.load(source_path, force="mesh")

            # --- 1. Normalize the Mesh (Your excellent code) ---
            # This is crucial for consistent training!
            bounds = mesh.bounds
            scale = 1.0 / max(bounds[1] - bounds[0])
            translation = -bounds[0] - (bounds[1] - bounds[0]) / 2.0
            mesh.apply_translation(translation)
            mesh.apply_scale(scale)

            # --- 2. Sample Points (New Logic) ---
            # We sample half the points on the surface and half randomly in the volume
            num_surface_points = NUM_POINTS_PER_SHAPE // 2
            num_random_points = NUM_POINTS_PER_SHAPE - num_surface_points

            surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface_points)
            # Sample random points in a slightly larger [-0.55, 0.55] cube
            random_points = (np.random.rand(num_random_points, 3) - 0.5) * 1.1
            
            points = np.vstack([surface_points, random_points])

            # --- 3. Calculate Signed Distances (New Logic) ---
            # This is the core step: calculate the SDF for every sampled point
            (sdfs, _, _) = mesh.nearest.signed_distance(points)
            sdfs = sdfs.reshape(-1, 1) # Reshape to a column vector (N, 1)

            # --- 4. Combine and Save (New Logic) ---
            # Stack the (N, 3) points and (N, 1) sdfs to get our (N, 4) array
            final_data = np.hstack([points, sdfs])

            # Save the result
            new_filename = os.path.splitext(filename)[0] + ".npy"
            target_path = os.path.join(TARGET_DIR, new_filename)
            np.save(target_path, final_data.astype(np.float32))

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print("\nPreprocessing complete!")
    print(f"SDF data saved to: {TARGET_DIR}")