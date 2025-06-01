import matplotlib.pyplot as plt

filetag = ["maddpg-vanilla", "maddpg-commloss", "masac-commloss"]#, "mappo-commloss"]
tasknames = ["direct_2agents", "direct_3agents", "broadcast_2agents", "broadcast_3agents"]
for taskname in tasknames:
    files = [ f"results/{tag}_{taskname}_3000/log_grep" for tag in filetag ]
    files2 = [ f"results/{tag}_{taskname}_3000/log_episodes" for tag in filetag ]
    labels = ["MADDPG baseline", "MADDPG", "MASAC"]#, "MAPPO"]

    for file, file2, label in zip(files, files2, labels):
        x_vals, y_vals = [], []
        y_vals2 = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                x_str, y_str = line.split(',', 1)
                x, y = float(x_str), float(y_str)
                x_vals.append(x)
                y_vals.append(y)

        with open(file2, 'r') as f2:
            for line in f2:
                line = line.strip()
                y2_str = line
                y2 = float(y2_str)
                y_vals2.append(y2)

        print(len(x_vals), len(y_vals), label)
        x_vals2 = list(range(len(y_vals2)))
        print(len(x_vals2), len(y_vals2), label)
        plt.plot(x_vals, y_vals, label=label)
        color = plt.gca().lines[-1].get_color() if plt.gca().lines else None
        plt.plot(x_vals2, y_vals2, color=color, alpha=0.5, linestyle='--')


    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.legend()
    plt.title(taskname + ' - Average Return per Episode')
    plt.grid(True)
    plt.savefig(f"results/plot_{taskname}_avg_return.png", dpi=300, bbox_inches='tight')
    plt.close()