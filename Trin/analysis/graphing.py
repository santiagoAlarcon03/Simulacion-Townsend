from matplotlib.figure import Figure


def particle_count_figure(times, counts):
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot(times, counts, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Particles")
    ax.grid(True, alpha=0.3)
    return fig
