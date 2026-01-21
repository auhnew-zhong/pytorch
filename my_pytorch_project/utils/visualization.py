import matplotlib.pyplot as plt


def plot_losses(losses, save_path=None):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
