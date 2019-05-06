from matplotlib import pyplot as plt
import numpy as np

metrics = ["loss", "boundary_cond", "max_error", "mean_absolute_error"]
labels = ["cost function", r"$|G(0) + G(\beta) + 1|$", "max error", "mean absolute error"]

for m, l in zip(metrics, labels):
    data_train = np.loadtxt("train_" + m + ".txt")
    data_test = np.loadtxt("test_" + m + ".txt")

    # Take into account rescaling: those metrics should correspond to G(tau)
    if m in ["max_error", "mean_absolute_error"]:
        data_train /= 2.
        data_test /= 2.

    plt.yscale("log")
    plt.plot(data_train, label="train")
    #plt.plot(data_test, label="test")
    plt.ylabel(l)
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(m + ".png", dpi=300)
    plt.show()
    plt.close()



