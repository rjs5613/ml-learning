import numpy as np


def main() -> None:
    X1, X2, X3, Y = np.loadtxt("data/pizza_multivar.txt", skiprows=1, unpack=True)
    print(f"X1: {X1}, X2: {X2}, X3: {X3}, Y: {Y}")
    X = np.column_stack((X1, X2, X3))
    w, b = train(X, Y, lr=0.001, epochs=10000)
    print(f"W: {w}, b: {b}")
    predicted_Y = predict(np.array([[20, 0.5, 1]]), w, b)
    print(
        f"Predicted price for 20cm pizza with 0.5cm crust and 1 topping: {predicted_Y}"
    )


def train(
    X: np.ndarray, Y: np.ndarray, lr: float, epochs: int
) -> tuple[np.ndarray, float]:
    """Train a linear regression model using gradient descent."""
    n_features = X.shape[1]
    W = np.zeros(n_features)
    b = 0.0
    for epoch in range(epochs):
        dW, db = gradient(X, Y, W, b)
        W -= dW * lr
        b -= db * lr
        if epoch % 1000 == 0:
            current_loss = loss(X, Y, W, b)
            print(f"Epoch {epoch}, Loss: {current_loss}, W: {W}, b: {b}")
    return W, b


def gradient(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """Compute the gradients of the loss with respect to W and b."""
    y_pred = predict(X, W, b)
    error = y_pred - Y
    # dW = -(2/n)*sum(X*(Y - y_pred)) sum of multiplications of 2 vectors are represented by dot product
    dW = 2 * np.dot(X.T, error) / X.shape[0]
    # db = -(2/n)*sum(Y - y_pred)
    db = 2 * np.average(error)
    return dW, db


def loss(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: float) -> np.floating[any]:
    """Compute the mean squared error loss."""
    y_pred = predict(X=X, W=W, b=b)
    error = Y - y_pred
    error_squared = (error) ** 2
    return np.average(error_squared)


def predict(X: np.ndarray, W: np.ndarray, b: float) -> np.ndarray:
    """Make predictions using the linear model."""
    return np.dot(X, W) + b


if __name__ == "__main__":
    main()
