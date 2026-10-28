import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])


train_full = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_dataset, val_dataset = random_split(
    train_full,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False
)


print("PyTorch Version:", torch.__version__)
print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Testing Images:", len(test_dataset))

images, labels = next(iter(train_loader))

print("Batch shape:", images.shape)
print("Labels shape:", labels.shape)


image, label = train_dataset[0]

plt.imshow(
    image.squeeze(0) * 0.3530 + 0.2860,
    cmap="gray"
)

plt.title("Fashion-MNIST: " + str(label))
plt.axis("off")
plt.show()






import torch.nn as nn
from torchinfo import summary



class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(


            nn.Flatten(),

            # First hidden layer
            nn.Linear(784, 256),
            nn.ReLU(),


            nn.Linear(256, 128),
            nn.ReLU(),

            # Output layer - 10 categories
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)



mlp_model = MLP()

print("\nMLP MODEL ARCHITECTURE")
print(mlp_model)


total_params = sum(
    p.numel() for p in mlp_model.parameters()
)

print("\nTotal MLP Parameters:", total_params)

# Torchinfo
summary(
    mlp_model,
    input_size=(128, 1, 28, 28)
)


images, labels = next(iter(train_loader))

mlp_model.eval()

with torch.no_grad():
    outputs = mlp_model(images)

print("\nMLP Test Results")
print("Input Shape:", images.shape)
print("Output Shape:", outputs.shape)

print("MLP Architecture Created Successfully!")



# ==========================================
# LAB 3 - TASK 2: CNN ARCHITECTURE
# ==========================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # First convolutional layer
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),


            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Flatten the feature maps
            nn.Flatten(),

            # Output
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)



cnn_model = CNN()

print("\nCNN MODEL ARCHITECTURE")
print(cnn_model)


cnn_params = sum(
    p.numel() for p in cnn_model.parameters()
)

print("\nTotal CNN Parameters:", cnn_params)


summary(
    cnn_model,
    input_size=(128, 1, 28, 28)
)


images, labels = next(iter(train_loader))

cnn_model.eval()

with torch.no_grad():
    cnn_outputs = cnn_model(images)

print("\nCNN Test Results")
print("Input Shape:", images.shape)
print("Output Shape:", cnn_outputs.shape)

print("CNN Architecture Created Successfully!")












class AttentionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Split image into 7x7 patches
        self.patches = nn.Unfold(
            kernel_size=7,
            stride=7
        )

        # Project each patch to 64 dimensions
        self.projection = nn.Linear(49, 64)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Output
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):

        # Converting image into 16 patches:D
        x = self.patches(x)

        # Shape: (batch, 49, 16)
        x = x.transpose(1, 2)

        # Shape: (batch, 16, 49)
        x = self.projection(x)


        x = self.transformer(x)


        x = x.mean(dim=1)


        x = self.classifier(x)

        return x


# Creating attention model
attention_model = AttentionModel()

print("\nATTENTION MODEL ARCHITECTURE")
print(attention_model)


attention_params = sum(
    p.numel() for p in attention_model.parameters()
)

print("\nTotal Attention Parameters:", attention_params)


summary(
    attention_model,
    input_size=(128, 1, 28, 28)
)


images, labels = next(iter(train_loader))

attention_model.eval()

with torch.no_grad():
    attention_outputs = attention_model(images)

print("\nATTENTION TEST RESULTS")
print("Input Shape:", images.shape)
print("Output Shape:", attention_outputs.shape)

print("Attention Architecture Created Successfully!")





# ==========================================
# LAB 3 - TASK 3: COMPUTATIONAL COSTS
# ==========================================

import time
import numpy as np

from pathlib import Path
from thop import profile
from torchinfo import summary

Path("results").mkdir(exist_ok=True)

# Using CPU for comparable timing measurements
device = torch.device("cpu")

models = {
    "MLP": mlp_model,
    "CNN": cnn_model,
    "Attention": attention_model
}

# Hand-calculated FLOPs per single image
# Approximation: one multiply-add = 2 FLOPs
# Excludes activations, biases, pooling, norms
# and softmax operations

mlp_flops = 2 * (
        784 * 256 +
        256 * 128 +
        128 * 10
)

cnn_flops = 2 * (
        28 * 28 * 16 * 1 * 3 * 3 +
        14 * 14 * 32 * 16 * 3 * 3 +
        32 * 7 * 7 * 10
)

# Attention: patch projection
patch_macs = 16 * 49 * 64

# Per transformer encoder layer
qkv_macs = 3 * 16 * 64 * 64
attention_score_macs = 16 * 16 * 64
attention_value_macs = 16 * 16 * 64
output_projection_macs = 16 * 64 * 64

feedforward_macs = (
        16 * 64 * 128 +
        16 * 128 * 64
)

encoder_macs = (
        qkv_macs +
        attention_score_macs +
        attention_value_macs +
        output_projection_macs +
        feedforward_macs
)

attention_flops = 2 * (
        patch_macs +
        2 * encoder_macs +
        64 * 10
)

manual_flops = {
    "MLP": mlp_flops,
    "CNN": cnn_flops,
    "Attention": attention_flops
}



def measure_latency(model, batch_size, runs):

    model.eval()

    x = torch.randn(
        batch_size, 1, 28, 28,
        device=device
    )

    # Warm-up
    with torch.inference_mode():

        for _ in range(10):
            model(x)

        times = []

        for _ in range(runs):

            start = time.perf_counter()

            model(x)

            elapsed = time.perf_counter() - start

            times.append(elapsed * 1000)

    return float(np.median(times))


# ==========================================
# MODEL COMPARISON
# ==========================================

report_lines = []

print("\nLAB 3 - COMPUTATIONAL COST ANALYSIS")

for name, model in models.items():

    model = model.to(device)
    model.eval()

    print(f"\n========== {name} ==========")

    # Count parameters using torchinfo
    info = summary(
        model,
        input_size=(1, 1, 28, 28),
        verbose=0
    )

    parameters = info.total_params

    print("Parameters:", parameters)

    # Hand-calculated FLOPs
    hand_flops = manual_flops[name]

    print("Hand-calculated FLOPs:", hand_flops)

    # THOP profiling
    dummy = torch.randn(1, 1, 28, 28)

    try:
        with torch.no_grad():
            macs, thop_params = profile(
                model,
                inputs=(dummy,),
                verbose=False
            )

        measured_flops = 2 * macs

        print("THOP MACs:", int(macs))
        print("THOP estimated FLOPs:", int(measured_flops))

    except Exception as error:

        macs = None
        measured_flops = None

        print("THOP profiling unavailable:", error)

    # Batch 1: median of 100 runs
    latency_1 = measure_latency(
        model,
        batch_size=1,
        runs=100
    )

    # Batch 128: median of 20 runs
    latency_128 = measure_latency(
        model,
        batch_size=128,
        runs=20
    )

    print(f"Batch 1 Latency: {latency_1:.4f} ms")
    print(f"Batch 128 Latency: {latency_128:.4f} ms")

    line = (
        f"Model: {name}\n"
        f"Parameters: {parameters}\n"
        f"Hand FLOPs: {hand_flops}\n"
        f"THOP FLOPs: {measured_flops}\n"
        f"Batch 1 Latency: {latency_1:.4f} ms\n"
        f"Batch 128 Latency: {latency_128:.4f} ms\n"
    )

    report_lines.append(line)


# ==========================================
# SAVE RESULTS
# ==========================================

with open(
        "results/cost_analysis.txt",
        "w",
        encoding="utf-8"
) as file:

    file.write("LAB 3 - COMPUTATIONAL COST ANALYSIS\n\n")

    for line in report_lines:
        file.write(line + "\n")

    file.write(
        "Note: Hand FLOPs are approximate and "
        "count multiply-add operations. THOP may "
        "omit functional attention operations and "
        "other unsupported layers.\n"
    )

print("\nTASK 3 COMPLETED!")
print("Results saved in results/cost_analysis.txt")







# ==========================================
# LAB 3 - TASK 4: MODEL TRAINING
# ==========================================

import torch.optim as optim
import time

# Evaluation function
def evaluate_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total



epochs = 5
learning_rate = 0.001

training_results = {}
training_histories = {}

print("\nLAB 3 - TASK 4: MODEL TRAINING")

# Train each model separately
for name, model in models.items():

    print(f"\n========== TRAINING {name} ==========")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_history = []
    val_history = []

    start_time = time.perf_counter()

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        # Average training loss
        epoch_loss = running_loss / total_samples

        # Validation accuracy
        val_accuracy = evaluate_model(
            model, val_loader
        )

        loss_history.append(epoch_loss)
        val_history.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

    # Final test accuracy
    test_accuracy = evaluate_model(
        model, test_loader
    )

    training_time = time.perf_counter() - start_time

    training_results[name] = {
        "accuracy": test_accuracy,
        "time": training_time
    }

    training_histories[name] = {
        "loss": loss_history,
        "validation": val_history
    }

    # Save trained model weights
    torch.save(
        model.state_dict(),
        f"results/{name.lower()}_model.pth"
    )

    print(f"\n{name} Training Completed!")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Training Time: {training_time:.2f} seconds")


# ==========================================
# SAVE TRAINING RESULTS
# ==========================================

with open(
        "results/training_results.txt",
        "w",
        encoding="utf-8"
) as file:

    file.write("LAB 3 - TRAINING RESULTS\n\n")

    for name, result in training_results.items():

        line = (
            f"{name}\n"
            f"Test Accuracy: {result['accuracy']:.2f}%\n"
            f"Training Time: {result['time']:.2f} seconds\n\n"
        )

        file.write(line)


# ==========================================
# TRAINING LOSS GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

for name, history in training_histories.items():

    plt.plot(
        range(1, epochs + 1),
        history["loss"],
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("MLP vs CNN vs Attention - Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/training_loss.png",
    dpi=150
)

plt.close()


# ==========================================
# FINAL ACCURACY COMPARISON
# ==========================================

print("\n================================")
print("FINAL TEST ACCURACY COMPARISON")
print("================================")

for name, result in training_results.items():

    print(
        f"{name}: {result['accuracy']:.2f}%"
    )

print("\nTASK 4 COMPLETED!")
print("Training results and models saved.")
