import torch.nn as nn
import re
from types import SimpleNamespace
import torch
from src.distill.models.projector import Projector
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import itertools
from torch.utils.data import random_split
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Teacher model
teacher_projector_config = SimpleNamespace(
    projector_type='mlp2x_gelu',
    retriever_hidden_size=4096,
    hidden_size=4096
)

teacher_projector = Projector(teacher_projector_config)
teacher_projector.load_state_dict(torch.load("/content/MyDrive/drive/tensorstorage/teacher_projector_weights.pth"))
teacher_projector.to(device)

# Student model
student_projector_config = SimpleNamespace(
    projector_type='mlp2x_gelu',
    retriever_hidden_size=4096,
    hidden_size=2304
)

student_projector = Projector(student_projector_config)
student_projector.to(device)

# Projector Layer
projection_layer = nn.Linear(2304, 4096)
projection_layer.to(device)

# Loading embeddings from file
doc_embeds_list = None
with open('/content/MyDrive/drive/tensorstorage/all_doc_embeds.pkl', 'rb') as f:
    doc_embeds_list = pickle.load(f)

# Setting up dataloader
embeds_tensor = torch.stack([tensor[0] for tensor in doc_embeds_list])
dataset = TensorDataset(embeds_tensor)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Set models to appropriate modes
student_projector.train()
projection_layer.train()
teacher_projector.eval() # Teacher model is not being trained

# We want to optimize the parameters of the student projector and the projection layer
params_to_optimize = itertools.chain(student_projector.parameters(), projection_layer.parameters())

# Initialize the optimizer
optimizer = optim.Adam(params_to_optimize, lr=1e-4)

num_epochs = 250
loss_function = torch.nn.MSELoss()
best_val_loss = float('inf')
output_dir = "/content/MyDrive/drive/tensorstorage/distilled_model"
os.makedirs(output_dir, exist_ok=True)


for epoch in range(num_epochs):
    total_train_loss = 0

    for batch in train_dataloader:
        x = batch[0].to(device)
        student_output = student_projector(x.to(next(teacher_projector.parameters()).dtype))
        student_output_projected = projection_layer(student_output)
        teacher_output = None
        with torch.no_grad():
            teacher_output = teacher_projector(x.to(next(teacher_projector.parameters()).dtype))

        loss = loss_function(student_output_projected, teacher_output.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    student_projector.eval()
    projection_layer.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            x = batch[0].to(device)
            student_output = student_projector(x.to(next(teacher_projector.parameters()).dtype))
            student_output_projected = projection_layer(student_output)
            teacher_output = teacher_projector(x.to(next(teacher_projector.parameters()).dtype))

            val_loss = loss_function(student_output_projected, teacher_output)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"New best model found at epoch {epoch+1} with validation loss: {best_val_loss:.4f}. Saving model.")
        torch.save(student_projector.state_dict(), os.path.join(output_dir, "best_student_projector.pth"))
        torch.save(projection_layer.state_dict(), os.path.join(output_dir, "best_projection_layer.pth"))
