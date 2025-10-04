import trimesh
import torch
import json
from main import Transformer_network
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import warnings
with warnings.catch_warnings():
        warnings.simplefilter("ignore")

task_count = {
    'Parts': 6,
    "Topology" : 5,
    "Density": 5
} 
def pred_single_file(model_path, file_path, mapping_path, device):

    with open(mapping_path, 'r') as file:
        mappings = json.load(file)

    reversed_map = {}
    for task_name, mapping_dict in mappings.items():
        reversed_map[task_name] = {idx: value for value, idx in mapping_dict.items()}

    class_nums = {task: len(mapping_dict) for task, mapping_dict in mappings.items()}

    model = Transformer_network(task_classes=task_count)

    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)

    
    model.eval()
    
    print(f"Processing File {file_path}")
    mesh = trimesh.load(file_path, force="mesh")
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.long)
    edge_set = set()
    faces_list = faces_tensor.tolist()
        
    for faces in faces_list:
        v1, v2, v3 = faces
        edge_set.add(tuple(sorted((v1,v2))))
        edge_set.add(tuple(sorted((v2,v3))))
        edge_set.add(tuple(sorted((v1,v3))))
        
    edge_list = list(edge_set)
    edge_index_unsorted = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index =torch.cat([edge_index_unsorted, edge_index_unsorted.flip(0)], dim=1)
    
    data = Data(
        x=torch.tensor(mesh.vertices, dtype=torch.float32), 
        pos=torch.tensor(mesh.sample(2048), dtype=torch.float32),
        faces = faces_tensor,
        edge_index=edge_index
    )
    
    data
    
    infrence_loader = DataLoader([data], batch_size=1)
    with torch.no_grad():
        data_batch = next(iter(infrence_loader))
        data_batch.to(device)
        
        predictions = model(data_batch)
    
    final_prediction = {}
    for task_name, logits in predictions.items():
        # Apply softmax to convert raw scores (logits) to probabilities
        probabilities = torch.softmax(logits, dim=1)
        # Get the index of the class with the highest probability
        predicted_index = torch.argmax(probabilities, dim=1).item()
        # Use the reverse mapping to get the human-readable class name
        predicted_class = reversed_map[task_name][predicted_index]
        
        final_prediction[task_name] = predicted_class
        
    return final_prediction

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path to a new file that was NOT in your training or validation set
    new_file =r'E:\Machine_Learning\Data\mesh_7.obj'
    
    # Get the prediction
    try:
        predicted_labels = pred_single_file(
            model_path=r'E:\Machine_Learning\models\models_task_3D_v2.pth',
            file_path=new_file,
            mapping_path=r'3D_Topology\labels.json',
            device=DEVICE
        )
    
        print("\n--- Model Predictions ---")
        for task, prediction in predicted_labels.items():
            print(f"  - {task}: {prediction}")
        print("-----------------------")

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")