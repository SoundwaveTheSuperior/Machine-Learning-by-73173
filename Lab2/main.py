
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 1. Veri setini yukle
digits = load_digits()

# 2. Piksel degerlerini 0-1 arasina getir
X = digits.data / 16.0

# 3. Gercek rakamlari al
y = digits.target

# 4. Etiketleri one-hot formatina cevir
Y = np.eye(10)[y]

# 5. Verileri %70 egitim, %30 test olarak ayir
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42,
    stratify=y
)

print("Toplam resim:", len(X))
print("Bir resimdeki piksel sayisi:", X.shape[1])
print("Egitim resimleri:", len(X_train))
print("Test resimleri:", len(X_test))
print("Ilk rakam:", y[0])
print("One-hot etiketi:", Y[0])

# 6. Ilk 10 rakami ekranda goster
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title("Digit: " + str(y[i]))
    ax.axis("off")

plt.tight_layout()
plt.show()














# 4. Cross-Entropy Loss

def cross_entropy_loss(y_true, y_pred):
    # Log(0) hatasini onle
    y_pred = np.clip(y_pred, 1e-12, 1.0)

    # Her resmin hatasini hesapla
    loss = -np.sum(y_true * np.log(y_pred), axis=1)

    # Ortalama hatayi hesapla
    return np.mean(loss)

# Ilk 5 resim icin esit baslangic olasiliklari
probabilities = np.full_like(Y_test[:5], 1 / 10, dtype=float)

# Ilk 5 resmin tahmin hatasi
loss = cross_entropy_loss(
    Y_test[:5],
    probabilities
)

print("\nCross-Entropy Loss Results")
print("Initial loss:", loss)
print("Expected initial loss:", np.log(10))












# 6. Sinir agini kur ve Numerical Gradient Checking yap

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    # Buyuk sayilarda tasmayi onle
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Gradyan kontrolu icin ilk 5 egitim resmini kullan
X_batch_train = X_train[:5]
Y_batch_train = Y_train[:5]

# Ag boyutlari: 64 piksel -> 32 -> 16 -> 10 rakam
rng = np.random.default_rng(42)

W1 = rng.normal(0, np.sqrt(2 / 64), size=(64, 32))
b1 = np.zeros((1, 32))

W2 = rng.normal(0, np.sqrt(2 / 32), size=(32, 16))
b2 = np.zeros((1, 16))

W3 = rng.normal(0, np.sqrt(2 / 16), size=(16, 10))
b3 = np.zeros((1, 10))


def gradient_check_loss():
    # Ileri hesaplama
    Z1 = X_batch_train @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    probabilities = softmax(Z3)

    return cross_entropy_loss(Y_batch_train, probabilities)


def calculate_gradients():
    # Ileri hesaplama
    Z1 = X_batch_train @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    probabilities = softmax(Z3)

    # Geri yayilim
    m = len(X_batch_train)

    dZ3 = (probabilities - Y_batch_train) / m
    dW3 = A2.T @ dZ3

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X_batch_train.T @ dZ1

    return dW1, dW2, dW3


# Geri yayilimdan gelen gradyanlar
dW1, dW2, dW3 = calculate_gradients()

weights = [W1, W2, W3]
gradients = [dW1, dW2, dW3]

epsilon = 1e-5
rng_check = np.random.default_rng(2026)
passed = 0

print("\nNumerical Gradient Checking")

# 10 rastgele agirligi kontrol et
for k in range(10):
    layer = int(rng_check.integers(0, 3))

    W = weights[layer]
    dW = gradients[layer]

    i = int(rng_check.integers(0, W.shape[0]))
    j = int(rng_check.integers(0, W.shape[1]))

    original = W[i, j]

    # Sayisal gradyan icin agirligi kucuk miktarda degistir
    W[i, j] = original + epsilon
    loss_plus = gradient_check_loss()

    W[i, j] = original - epsilon
    loss_minus = gradient_check_loss()

    # Orijinal agirligi geri koy
    W[i, j] = original

    numerical = (loss_plus - loss_minus) / (2 * epsilon)
    analytical = dW[i, j]

    relative_error = abs(numerical - analytical) / max(
        1e-8, abs(numerical) + abs(analytical)
    )

    if relative_error < 1e-6:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL"

    print(
        f"Check {k + 1}: W{layer + 1} "
        f"Error = {relative_error:.8e} "
        f"{status}"
    )

print(f"\nGradient Check Result: {passed}/10")

if passed >= 9:
    print("GRADIENT CHECK PASSED")
else:
    print("GRADIENT CHECK FAILED")



















# =====================================================
# 7. COMPLETE LAB 2 - CONTINUATION
# =====================================================

from pathlib import Path
import time

Path("results").mkdir(exist_ok=True)


# =====================================================
# NEURAL NETWORK FUNCTIONS
# =====================================================

def initialize_network(width):
    rng = np.random.default_rng(42)

    params = {
        "W1": rng.normal(0, 0.01, (64, width)),
        "W2": rng.normal(0, 0.01, (width, width)),
        "W3": rng.normal(0, 0.01, (width, 10)),

        "b1": np.zeros((1, width)),
        "b2": np.zeros((1, width)),
        "b3": np.zeros((1, 10))
    }

    return params


def forward_network(x, params):

    z1 = x @ params["W1"] + params["b1"]
    a1 = relu(z1)

    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)

    z3 = a2 @ params["W3"] + params["b3"]
    p = softmax(z3)

    return z1, a1, z2, a2, p


def backward_network(x, y, params, cache):

    z1, a1, z2, a2, p = cache
    n = len(x)

    # Output layer
    dz3 = (p - y) / n
    dw3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0, keepdims=True)

    # Hidden layer 2
    dz2 = (dz3 @ params["W3"].T) * (z2 > 0)
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Hidden layer 1
    dz1 = (dz2 @ params["W2"].T) * (z1 > 0)
    dw1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    gradients = {
        "W1": dw1,
        "W2": dw2,
        "W3": dw3,
        "b1": db1,
        "b2": db2,
        "b3": db3
    }

    return gradients


# =====================================================
# 8. GRADIENT CHECK - REQUIRED ARCHITECTURE
# =====================================================

print("\nGRADIENT CHECK - HIDDEN WIDTH 16")

params_check = initialize_network(16)

x_check = X_train[:32]
y_check = Y_train[:32]

cache_check = forward_network(x_check, params_check)

gradients_check = backward_network(
    x_check, y_check, params_check, cache_check
)

epsilon = 1e-5
rng_check = np.random.default_rng(2026)

passed = 0
check_report = []

for k in range(10):

    layer = int(rng_check.integers(1, 4))
    name = f"W{layer}"

    W = params_check[name]

    i = int(rng_check.integers(W.shape[0]))
    j = int(rng_check.integers(W.shape[1]))

    original = W[i, j]

    # Positive perturbation
    W[i, j] = original + epsilon

    p_plus = forward_network(
        x_check, params_check
    )[-1]

    loss_plus = cross_entropy_loss(y_check, p_plus)

    # Negative perturbation
    W[i, j] = original - epsilon

    p_minus = forward_network(
        x_check, params_check
    )[-1]

    loss_minus = cross_entropy_loss(y_check, p_minus)

    # Restore original weight
    W[i, j] = original

    numerical = (loss_plus - loss_minus) / (2 * epsilon)

    analytical = gradients_check[name][i, j]

    relative_error = abs(
        numerical - analytical
    ) / max(
        1e-8,
        abs(numerical) + abs(analytical)
    )

    if relative_error < 1e-6:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL"

    line = (
        f"Check {k+1}: {name} "
        f"Error={relative_error:.8e} {status}"
    )

    print(line)
    check_report.append(line)

print(f"\nGradient Check Result: {passed}/10")

if passed >= 9:
    print("GRADIENT CHECK PASSED")
else:
    print("GRADIENT CHECK FAILED")

with open("results/gradient_check.txt", "w") as file:
    file.write("\n".join(check_report))
    file.write(f"\nResult: {passed}/10\n")


# =====================================================
# 9. MODEL TRAINING
# =====================================================

def train_network(
        width,
        learning_rate,
        epochs=50,
        batch_size=32
):

    params = initialize_network(width)

    shuffle_rng = np.random.default_rng(123)

    loss_history = []

    for epoch in range(epochs):

        # Shuffle training data
        indices = shuffle_rng.permutation(len(X_train))

        # Mini-batch training
        for start in range(0, len(X_train), batch_size):

            batch_indices = indices[start:start + batch_size]

            xb = X_train[batch_indices]
            yb = Y_train[batch_indices]

            # Forward propagation
            cache = forward_network(xb, params)

            # Backpropagation
            gradients = backward_network(
                xb, yb, params, cache
            )

            # Gradient descent
            for name in params:
                params[name] -= learning_rate * gradients[name]

        # Calculate loss after every epoch
        train_p = forward_network(X_train, params)[-1]

        epoch_loss = cross_entropy_loss(Y_train, train_p)

        loss_history.append(epoch_loss)

        print(
            f"Epoch {epoch+1}: Loss = {epoch_loss:.6f}"
        )

    # Final test evaluation
    test_p = forward_network(X_test, params)[-1]

    predictions = np.argmax(test_p, axis=1)
    actual = np.argmax(Y_test, axis=1)

    accuracy = np.mean(predictions == actual)

    return loss_history, accuracy, params


# =====================================================
# 10. SIX MODEL EXPERIMENTS
# =====================================================

widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]

results = {}
trained_models = {}

print("\nSTARTING SIX MODEL EXPERIMENTS")

for width in widths:

    for lr in learning_rates:

        print(f"\nTraining Width={width}, LR={lr}")

        history, accuracy, params = train_network(
            width=width,
            learning_rate=lr,
            epochs=50,
            batch_size=32
        )

        results[(width, lr)] = (history, accuracy)
        trained_models[(width, lr)] = params

        print(f"Final Accuracy: {accuracy * 100:.2f}%")


# =====================================================
# 11. SAVE TEST ACCURACY RESULTS
# =====================================================

with open("results/test_accuracies.txt", "w") as file:

    file.write("LAB 2 - TEST ACCURACY RESULTS\n\n")

    for (width, lr), (history, accuracy) in results.items():

        line = (
            f"Width={width}, "
            f"Learning Rate={lr}, "
            f"Accuracy={accuracy * 100:.2f}%"
        )

        print(line)
        file.write(line + "\n")


# =====================================================
# 12. FIRST TRAINING LOSS GRAPH
# =====================================================

first_history = results[(64, 0.1)][0]

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 51),
    first_history,
    label="Width=64, LR=0.1"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Neural Network Training Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/first_training_loss.png",
    dpi=150
)

plt.close()


# =====================================================
# 13. SIX LOSS CURVES GRAPH
# =====================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharey=True
)

for ax, width in zip(axes, widths):

    for lr in learning_rates:

        history, accuracy = results[(width, lr)]

        ax.plot(
            range(1, 51),
            history,
            label=f"LR={lr}"
        )

    ax.set_title(f"Hidden Width = {width}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True)

plt.tight_layout()

plt.savefig(
    "results/loss_curves.png",
    dpi=150
)

plt.close()

print("\nLOSS CURVES SAVED SUCCESSFULLY")


# =====================================================
# 14. COMPUTATIONAL TIMING
# =====================================================

def benchmark_network(width):

    # Use trained model
    params = trained_models[(width, 0.1)]

    x = X_train[:32]
    y = Y_train[:32]

    # Forward and backward functions
    def forward():
        return forward_network(x, params)

    cache = forward()

    def backward():
        return backward_network(x, y, params, cache)

    # Warm-up
    for _ in range(10):
        forward()
        backward()

    forward_times = []
    backward_times = []

    # Measure forward 100 times
    for _ in range(100):

        start = time.perf_counter_ns()

        forward()

        elapsed = time.perf_counter_ns() - start

        forward_times.append(elapsed / 1_000_000)

    # Measure backward 100 times
    for _ in range(100):

        start = time.perf_counter_ns()

        backward()

        elapsed = time.perf_counter_ns() - start

        backward_times.append(elapsed / 1_000_000)

    forward_median = np.median(forward_times)
    backward_median = np.median(backward_times)

    return forward_median, backward_median


# =====================================================
# 15. TIMING BREAKDOWN REPORT
# =====================================================

print("\nCOMPUTATIONAL TIMING RESULTS")

timing_report = "Computational Timing Results\n"

for width in [16, 64]:

    forward_ms, backward_ms = benchmark_network(width)

    line = (
        f"\nHidden Width: {width}\n"
        f"Forward Median: {forward_ms:.6f} ms\n"
        f"Backward Median: {backward_ms:.6f} ms\n"
    )

    print(line)

    timing_report += line

    if backward_ms > forward_ms:
        explanation = (
            "Backward pass took longer. "
            "It requires additional matrix multiplications "
            "to calculate gradients.\n"
        )
    else:
        explanation = (
            "Forward pass took longer in this measurement. "
            "For small models, Python overhead and timing "
            "variation can affect measured performance.\n"
        )

    timing_report += explanation

    print(explanation)

with open("results/timing_breakdown.txt", "w") as file:
    file.write(timing_report)


# =====================================================
# 16. FINAL LAB 2 RESULTS
# =====================================================

print("\n===================================")
print("LAB 2 - FINAL RESULTS")
print("===================================")

print(f"Gradient Checking: {passed}/10")
print("Experiments Completed:", len(results))

best_config = max(
    results,
    key=lambda config: results[config][1]
)

best_accuracy = results[best_config][1]

print("Best Hidden Width:", best_config[0])
print("Best Learning Rate:", best_config[1])
print(f"Best Test Accuracy: {best_accuracy * 100:.2f}%")

print("\nResults saved in the results folder")
print("LAB 2 COMPLETED")

# Display saved loss curves
image = plt.imread("results/loss_curves.png")
plt.figure(figsize=(12, 5))
plt.imshow(image)
plt.axis("off")
plt.show()
