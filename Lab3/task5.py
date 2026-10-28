# ==========================================
# LAB 3 - TASK 5: ARCHITECTURAL TRADE-OFFS


import re
from pathlib import Path
import matplotlib.pyplot as plt

# Previous experiment results
results_folder = Path("results")

training_text = (
        results_folder / "training_results.txt"
).read_text(encoding="utf-8")

cost_text = (
        results_folder / "cost_analysis.txt"
).read_text(encoding="utf-8")

models = ["MLP", "CNN", "Attention"]
comparison = []

# Read results from previous tasks
for name in models:

    accuracy_match = re.search(
        rf"(?m)^{name}\s*\nTest Accuracy:\s*([\d.]+)%",
        training_text
    )

    cost_match = re.search(
        rf"(?ms)^Model:\s*{name}\s*\n"
        rf"Parameters:\s*(\d+).*?"
        rf"^Batch 1 Latency:\s*([\d.]+)\s*ms",
        cost_text
    )

    if accuracy_match is None or cost_match is None:
        raise ValueError(
            f"Cannot find saved results for {name}"
        )

    accuracy = float(accuracy_match.group(1))
    parameters = int(cost_match.group(1))
    latency = float(cost_match.group(2))

    comparison.append({
        "model": name,
        "accuracy": accuracy,
        "parameters": parameters,
        "latency": latency
    })


# ==========================================
# 1. ACCURACY VS PARAMETERS
# ==========================================

plt.figure(figsize=(8, 5))

for item in comparison:

    x = item["parameters"] / 1000
    y = item["accuracy"]

    plt.scatter(x, y, s=100)

    plt.annotate(
        item["model"],
        (x, y),
        xytext=(7, 7),
        textcoords="offset points"
    )

plt.xlabel("Parameters (Thousands)")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Parameters")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    results_folder / "accuracy_vs_parameters.png",
    dpi=150
)

plt.close()


# ==========================================
# 2. ACCURACY VS LATENCY
# ==========================================

plt.figure(figsize=(8, 5))

for item in comparison:

    x = item["latency"]
    y = item["accuracy"]

    plt.scatter(x, y, s=100)

    plt.annotate(
        item["model"],
        (x, y),
        xytext=(7, 7),
        textcoords="offset points"
    )

plt.xlabel("Inference Latency (ms, Batch 1)")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Inference Latency")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    results_folder / "accuracy_vs_latency.png",
    dpi=150
)

plt.close()


# ==========================================
# 3. CREATE RESULTS TABLE
# ==========================================

lines = [
    "# Lab 3 - Model Comparison",
    "",
    "| Model | Parameters | Accuracy | Latency (ms) |",
    "|---|---:|---:|---:|"
]

for item in comparison:

    lines.append(
        f"| {item['model']} "
        f"| {item['parameters']:,} "
        f"| {item['accuracy']:.2f}% "
        f"| {item['latency']:.4f} |"
    )

report = "\n".join(lines)

(results_folder / "model_comparison.md").write_text(
    report,
    encoding="utf-8"
)

print("\nLAB 3 - TASK 5 RESULTS")
print(report)

print("\nGraphs saved successfully!")
print("TASK 5 COMPLETED!")




