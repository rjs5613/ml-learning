import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    X, Y = np.loadtxt("data/pizza.txt", skiprows=1, unpack=True)
    print(f"X: {X}, Y: {Y}")
    w, b = train(X, Y, lr=0.001, epochs=10000)
    print(f"w: {w}, b: {b}")
    predicted_Y = predict(20, w, b)
    print(f"Predicted price for 20cm pizza: {predicted_Y}")
    plot_regression(X, Y, w, b)


def plot_regression(X, Y, w, b):
    sns.scatterplot(x=X, y=Y)
    x_vals = np.array([min(X), max(X)])
    y_vals = predict(x_vals, w, b)
    plt.plot(x_vals, y_vals, color="red")
    plt.xlabel("Diameter (cm)")
    plt.ylabel("Price ($)")
    plt.title("Pizza Price vs Diameter")
    plt.show()


def train(X, Y, lr, epochs):
    """Train a linear regression model using gradient descent."""
    w, b = 0.0, 0.0
    for epoch in range(epochs):
        dw, db = gradient(X, Y, w, b)
        w -= dw * lr
        b -= db * lr
        if epoch % 1000 == 0:
            current_loss = loss(X, Y, w, b)
            print(f"Epoch {epoch}, Loss: {current_loss}, w: {w}, b: {b}")
    return w, b


def gradient(X, Y, w, b):
    """Compute the gradients of the loss with respect to w and b."""
    N = len(X)
    y_pred = predict(X, w, b)
    print(f"y_pred: {y_pred}")
    error = y_pred - Y
    dw = (2 / N) * np.dot(error, X)
    db = (2 / N) * np.sum(error)
    return dw, db


def loss(X, Y, w, b):
    """Compute the mean squared error loss."""
    y_pred = predict(X, w, b)
    return np.average((Y - y_pred) ** 2)


def predict(X, w, b):
    """Make predictions using the linear model."""
    print(f"X in predict: {X}")
    print(f"w in predict: {w}, b in predict: {b}")
    if not (np.isfinite(w) and np.isfinite(b)):
        raise RuntimeError(
            "Model parameters are non-finite (NaN/Inf). Training diverged."
        )
    return w * X + b


if __name__ == "__main__":
    main()
