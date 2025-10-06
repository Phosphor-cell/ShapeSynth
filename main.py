print("Importing....")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("fixing dependencies... ")
import pandas as pd
import json
import trimesh
import pymeshlab as pml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset#, DataLoader
from torch.optim import AdamW   
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_edge_index
from torch.cuda.amp import autocast
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

#----   File Paths  ----
CSV_PATH  = "3D_Topology/labels.csv"
JSON_PATH = "3D_Topology/labels.json"
DATA_DIR = "3D_Topology/Data"
version = "v2"

#---- Hyperparameters ----
EPOCHS = 50
EPOCHS_MULTIPLIER = 6
DIM_COUNT = 81
POINTS_SAMPLE =1024
BATCH_SIZE = 8
NUM_TRANSFORMER_LAYER = 6
LEARNING_RATE = 1e-2
GNN_LAYERS = 4
DROP_NUM = 0.2

#--- Device Setup ---
if torch.cuda.is_available():
    torch.set_default_device("cuda")



#---- Utils Helpers ----
def augmentation(mesh):
    scale_matrix = trimesh.transformations.scale_matrix(np.random.uniform(0.4, 3.0))
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-np.pi, np.pi)
    rotation_matrix = trimesh.transformations.scale_matrix(angle, axis)
    combined_matrix = scale_matrix@rotation_matrix
    
    mesh.apply_transform(combined_matrix)
    return mesh

with open(JSON_PATH, 'r') as file:
    global MAPPING
    MAPPING = json.load(file)

def collate_fn(batch_list):
    batched_point_clouds = []
    batch_faces = []
    batch_labels = []
    
    max_len = 0
    for sample in batch_list:
        if sample['point_cloud'].shape[0] > max_len:
            max_len = sample['point_cloud'].shape[0]
        
    
    padded_cloud = []
    padded_faces = []
    
    max_len_face = 2
    for sample_2 in batch_list:
        if sample_2['faces'].shape[0] > max_len_face:
            max_len_face = sample_2['faces'].shape[0]
    
    
    for sample in batch_list:
        if sample is not None:
            pc = sample['point_cloud']
            
            num_points = pc.shape[0]
            padding = torch.zeros(max_len-num_points, 3)
            padded_cloud.append(torch.cat([pc, padding], dim=0))
            
            faces_len = sample_2['faces']
            padding_faces = 2
            
            batch_faces.append(sample["faces"])
            batch_labels.append(sample['labels'])
    
    batched_point_clouds = torch.stack(padded_cloud, dim=0)
    #batch_faces = torch.stack(batch_faces, dim=0)  
    batch_labels = torch.stack(batch_labels, dim=0)
    
    return {
        "batched_clouds":  batched_point_clouds,
        "faces" : batch_faces,
        "labels" : batch_labels
    }

#---- Dataset Preperations ----
class mesh_dataset(Dataset):
    
    def __init__(self, csv_path=CSV_PATH, json_classes=MAPPING, data_dir=DATA_DIR, sample_count=POINTS_SAMPLE, epoch_mutliplier = EPOCHS_MULTIPLIER):
        self.dataframe = pd.read_csv(csv_path)
        self.mappings = json_classes
        self.label_cols = list(json_classes.keys())
        self.data_dir = data_dir
        self.num_points = sample_count
        self.epoch_multi = epoch_mutliplier
        self.real_len = len(self.dataframe)

    def __len__(self):
        return self.real_len*self.epoch_multi
    
    def __getitem__(self, index):
        real_idx = index%self.real_len
        mesh_file = self.dataframe.iloc[real_idx, 1]
        mesh_path = os.path.join(self.data_dir, mesh_file)
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh = augmentation(mesh)
        point_cloud = mesh.sample(self.num_points)
        faces = mesh.faces
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        labels_raw = self.dataframe.iloc[real_idx, 2:]
        
        label_idx = []
        for col in self.label_cols:
            raw_val = labels_raw[col]
            idx_value = self.mappings[col].get(str(raw_val), raw_val)
            #print(f"col {col}, raw {raw_val}, mapped {idx_value}")
            label_idx.append(idx_value)
        
        label_tensor = torch.tensor(label_idx, dtype=torch.long).unsqueeze(0)
        point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32)
        faces_tensor = torch.tensor(faces, dtype=torch.long)
        
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
        
        data_peice = Data(
            x=vertices,
            pos=point_cloud_tensor,
            faces = faces_tensor,
            edge_index=edge_index,
            y= label_tensor
        )
        
        
        return data_peice

mesh_data = mesh_dataset()

mesh_loader = DataLoader(mesh_data, batch_size = BATCH_SIZE, shuffle=False , num_workers=0,collate_fn=collate_fn)

#----   DATA LOADER CHECKER     ----
#for batch in mesh_loader:
#    print(batch["batched_clouds"])
#----   DATA LOADER CHECKER END ----

class Transformer_network(nn.Module):
    def __init__(self, embedding_dim = DIM_COUNT, num_transformer = NUM_TRANSFORMER_LAYER, num_heads = 3, task_classes = None, dropout_rate = DROP_NUM, gnn_layers = GNN_LAYERS, input_dim = 3, sample_count = POINTS_SAMPLE):
        super().__init__()
        self.sample_count = sample_count
        
        
        #----- GNN Calculations -----
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(input_dim, embedding_dim))
        for _ in range(gnn_layers - 1):
            self.gnn_layers.append(GCNConv(embedding_dim, embedding_dim))
        
        
        
        #----- Point CLoud Calculations ---
        self.embed = nn.Linear(3, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer)
        
        self.face_extracted = nn.Sequential(
            nn.Linear(3, 64), # Example: processes coordinates of 3 vertices
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.heads = nn.ModuleDict()
        for task_name, num_classes in task_classes.items():
            head = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim//2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embedding_dim // 2, num_classes)
            )
            self.heads[task_name] = head

    def forward(self, data, face_list=None):
        
        gnn_input_features, num_samples, faces, edge_index, batch_map = data.x, data.pos, data.face, data.edge_index, data.batch
        
        gnn_out = gnn_input_features
        #print("gnn_out")
        for layers in self.gnn_layers:
            gnn_out = F.relu(layers(gnn_out, edge_index))
        graph_vector = global_mean_pool(gnn_out, batch_map)
        #print(f"GNN Vector {graph_vector.shape}")
        
        batch_size = data.num_graphs
        
        
        point_embedding = self.embed(num_samples)
        #print(f'embedded {point_embedding.shape}')
        point_cloud_batch = point_embedding.view(batch_size, self.sample_count, -1)
        
        #print(f"Reshaped Tensor: {point_cloud_batch.shape}")
        transformer_output = self.transformer_encoder(point_cloud_batch)
        #print(f"Transformer Output: {transformer_output.shape}")
        shape_vector = transformer_output.mean(dim=1)
        #print(f"shape_vector: {shape_vector.shape}")
        combined_vector = torch.cat([graph_vector, shape_vector], dim=1)
        
        outputs = {}
        
        for task_name, head in self.heads.items():
            outputs[task_name] = head(shape_vector)
        
        if face_list is not None:
            faces_feature = []
            for face_tensor in face_list:
                processed_faces = self.face_extracted(face_tensor)
                faces_feature.append(processed_faces)
        
        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_count = {
    'Parts': 6,
    "Topology" : 5,
    "Density":5
} 

TASK_NAMES = list(task_count.keys())


model = Transformer_network(task_classes=task_count).to(device)
#model = torch.compile(model=model)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

loss_func = {task: nn.CrossEntropyLoss() for task in TASK_NAMES}
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

if __name__ == "__main__":
    for epochs in range(EPOCHS):
        model.train()
    
        running_loss =0.00
        
        for batch in mesh_loader:
            if batch is None:
                continue
        
        
        #point_cloud = batch['batched_clouds']
        #faces = batch['faces']
            true_labels = batch.y
        
            
            optimizer.zero_grad()
        
        #with autocast(dtype=torch.float16):
            predictions = model(batch)
        
            total_loss = 0
        
            for i, task_name in enumerate(TASK_NAMES):
                task_predictions = predictions[task_name]
                task_labels = true_labels[:, i]
            
                loss = loss_func[task_name](task_predictions, task_labels)
                total_loss += loss
        
            total_loss.backward()
        
            optimizer.step()
        
            running_loss += total_loss.item()
        
        scheduler.step()
    
        avg_epoch_loss = running_loss / len(mesh_loader)
        current_lr = scheduler.get_last_lr()[0]
    
        print(f"Epoch {epochs+1}/{EPOCHS} | Avg. Training Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.6f}")

    print("\n--- Training Finished ---")    

    torch.save(model.state_dict(), f'models/multi_task_3D_{version}.pth')
    print("Model saved to models/multi_task_3D.pth")
