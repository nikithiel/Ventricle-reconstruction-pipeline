import matplotlib.pyplot as plt
import numpy as np

def a():
    fig = plt.figure(figsize=(5, 5))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, color='blue', linewidth=2)
    plt.title("Sine Wave")
    plt.xlabel("X")
    plt.ylabel("Y")
    return fig

fig = a()
fig.show()
