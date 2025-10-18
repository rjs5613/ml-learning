import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    X, Y = np.loadtxt("data/pizza.txt", skiprows=1, unpack=True)
    sns.scatterplot(x=X, y=Y)
    plt.xlabel("Diameter (inches)")
    plt.ylabel("Price (dollars)")
    plt.title("Pizza Price vs Diameter")
    plt.show()


if __name__ == "__main__":
    main()
