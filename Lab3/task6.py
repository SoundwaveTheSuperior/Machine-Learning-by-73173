# ==========================================
# LAB 3 - TASK 6: GENERATE FINAL REPORT
# ==========================================

from pathlib import Path
import re

project = Path(__file__).resolve().parent
results = project / "results"

# Read previous results
cost_text = (results / "cost_analysis.txt").read_text(
    encoding="utf-8"
)

training_text = (results / "training_results.txt").read_text(
    encoding="utf-8"
)

models = ["MLP", "CNN", "Attention"]
data = {}

for name in models:

    # Find training accuracy
    accuracy_match = re.search(
        rf"(?m)^{name}\s*\nTest Accuracy:\s*([\d.]+)%",
        training_text
    )

    # Find computational cost section
    cost_match = re.search(
        rf"(?ms)^Model:\s*{name}\s*\n"
        rf"(.*?)(?=^Model:\s*|\Z)",
        cost_text
    )

    if not accuracy_match or not cost_match:
        raise ValueError(f"Missing results for {name}")

    section = cost_match.group(1)

    def get_field(label):
        match = re.search(
            rf"(?m)^{re.escape(label)}:\s*(.+?)\s*$",
            section
        )
        if not match:
            raise ValueError(f"{name}: Missing {label}")
        return match.group(1).strip()

    parameters = int(get_field("Parameters"))
    hand_flops = int(float(get_field("Hand FLOPs")))

    thop_value = get_field("THOP FLOPs")
    thop_flops = (
        None if thop_value == "None"
        else int(float(thop_value))
    )

    latency_1 = float(get_field("Batch 1 Latency").split()[0])
    latency_128 = float(get_field("Batch 128 Latency").split()[0])

    data[name] = {
        "parameters": parameters,
        "accuracy": float(accuracy_match.group(1)),
        "hand_flops": hand_flops,
        "thop_flops": thop_flops,
        "latency_1": latency_1,
        "latency_128": latency_128
    }


# Build Markdown results table
comparison_rows = []
flops_rows = []

for name, values in data.items():

    comparison_rows.append(
        f"| {name} "
        f"| {values['parameters']:,} "
        f"| {values['accuracy']:.2f}% "
        f"| {values['latency_1']:.4f} "
        f"| {values['latency_128']:.4f} |"
    )

    measured = (
        f"{values['thop_flops']:,}"
        if values["thop_flops"] is not None
        else "Unavailable"
    )

    flops_rows.append(
        f"| {name} "
        f"| {values['hand_flops']:,} "
        f"| {measured} |"
    )


comparison_table = "\n".join(comparison_rows)
flops_table = "\n".join(flops_rows)


# Generate final report
report = f"""# Lab 3 - Architectures and Their Costs

## 1. Introduction

This laboratory compares three neural network architectures:
Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN),
and an Attention-based Transformer.

The aim is to compare accuracy, parameter count,
computational cost, inference latency, and memory-access patterns.

## 2. Dataset and Training

- Dataset: Fashion-MNIST
- Training images: 55,000
- Validation images: 5,000
- Test images: 10,000
- Image size: 28 x 28
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001
- Training epochs: 5

## 3. Model Architectures

### MLP

Flatten(784) -> Linear(256) -> ReLU ->
Linear(128) -> ReLU -> Linear(10)

### CNN

Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2d(2) ->
Conv2d(16,32,3,padding=1) -> ReLU -> MaxPool2d(2) ->
Flatten -> Linear(1568,10)

### Attention

16 image patches of 7x7 pixels ->
Linear projection to 64 dimensions ->
Two Transformer Encoder layers ->
Mean pooling -> Linear(64,10)

Each Transformer Encoder has 4 attention heads
and a feedforward dimension of 128.

## 4. Experimental Results

| Model | Parameters | Test Accuracy | Batch 1 (ms) | Batch 128 (ms) |
|---|---:|---:|---:|---:|
{comparison_table}

The latency values are CPU measurements, recorded
as the median over 100 runs for batch 1
and 20 runs for batch 128.

## 5. FLOPs Comparison

| Model | Hand-calculated FLOPs | THOP-estimated FLOPs |
|---|---:|---:|
{flops_table}

All FLOPs values are per single image.

The hand calculations use the approximation
that one multiply-accumulate operation equals two FLOPs.

### Hand Calculation Formulas

MLP:

2 x (784 x 256 + 256 x 128 + 128 x 10)

CNN:

2 x (28 x 28 x 16 x 1 x 3 x 3
+ 14 x 14 x 32 x 16 x 3 x 3
+ 32 x 7 x 7 x 10)

Attention:

Patch projection, QKV projections, attention scores,
attention-weighted values, output projections,
and feedforward layers are included.

The hand calculations omit some operations such as
bias additions, activation functions, normalization,
pooling, and softmax.

THOP estimates are obtained by multiplying its
reported MAC count by two. THOP may not count
all functional operations, especially inside
Transformer attention.

Therefore, the difference between hand calculations
and THOP estimates does not necessarily indicate
an error in the architecture.

## 6. Architectural Trade-Offs

### Accuracy vs Parameters

![Accuracy vs Parameters](results/accuracy_vs_parameters.png)

### Accuracy vs Latency

![Accuracy vs Latency](results/accuracy_vs_latency.png)

## 7. Discussion

### MLP

MLP achieved {data['MLP']['accuracy']:.2f}% accuracy
with {data['MLP']['parameters']:,} parameters.

It has simple dense layers and regular matrix operations.
However, flattening the image removes explicit
two-dimensional spatial structure.

Dense layers also require reading many independent
weight values from memory.

### CNN

CNN achieved {data['CNN']['accuracy']:.2f}% accuracy
with {data['CNN']['parameters']:,} parameters.

Convolution uses shared filters to learn local patterns.
This reduces the number of parameters and provides
useful weight reuse.

CNN must also access intermediate feature maps,
so low parameter count does not automatically mean
the lowest computational cost or latency.

### Attention

Attention achieved {data['Attention']['accuracy']:.2f}% accuracy
with {data['Attention']['parameters']:,} parameters.

Self-attention allows image patches to exchange information.
However, QKV projections, attention matrices,
and feedforward layers introduce computational
and memory-access overhead.

This model does not include explicit positional embeddings,
which may limit its ability to represent spatial positions.

## 8. Conclusion

CNN obtained the highest accuracy and used the fewest
parameters in this experiment.

MLP had the lowest batch-1 inference latency.

Attention was slower and less accurate under the tested
five-epoch training configuration.

There is no universally best architecture:
the preferred model depends on accuracy, memory,
computation, latency, and application requirements.

The laboratory manual mentions a target of approximately
100,000-150,000 parameters, while the specified layer
configurations produce different parameter counts.
The prescribed layer configurations were retained.

## 9. Implementation

- main.py: dataset, architectures, profiling, and training
- task5.py: trade-off plots and comparison table
- task6.py: final report generation
- results/: figures, timing, FLOPs, and training results
"""

# Save final report
(project / "report.md").write_text(
    report, encoding="utf-8"
)

print("LAB 3 - TASK 6")
print("report.md created successfully!")
print("Results loaded for:", ", ".join(models))
print("TASK 6 COMPLETED!")










