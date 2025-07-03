import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def read_rewards_from_file(file_path):
    """
    Čita srednje nagrade po modelu iz .txt fajla.

    Args:
        file_path (str): Putanja do .txt fajla.

    Returns:
        list: Lista prosečnih nagrada po modelu (float).
    """
    rewards = []
    with open(file_path, "r") as f:
        for line in f:
            if ":" in line:
                try:
                    reward = float(line.strip().split(":")[1])
                    rewards.append(reward)
                except ValueError:
                    continue
    return rewards


def plot_multiple_results(file_paths, output_image="vise_na_jednom.png", labels=None):
    """
    Crta više krivih na istoj slici i čuva graf.

    Args:
        file_paths (list): Lista putanja do .txt fajlova sa rezultatima.
        output_image (str): Ime slike za čuvanje.
        labels (list, optional): Lista imena za legende.
    """
    plt.figure(figsize=(12, 6))

    for idx, file_path in enumerate(file_paths):
        rewards = read_rewards_from_file(file_path)
        
        label = labels[idx] if labels and idx < len(labels) else f"Fajl {idx+1}"
        plt.plot(range(len(rewards)), rewards, marker='o', linestyle='-', label=label)

    plt.title("Poređenje modela - svi rezultati na jednoj slici")
    plt.xlabel("Broj modela")
    plt.ylabel("Prosečna nagrada")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_image)
    plt.close()

    print(f"[INFO] Svi grafovi sačuvani na jednoj slici: {output_image}")



numb=6

file_paths=[]
file_paths.append("output6.txt")
file_paths.append("output7.txt")
file_paths.append("output8.txt")

imena = ["Small","Mid","Big"]

plot_multiple_results(file_paths, output_image="graf_rezultati"+str(numb)+".png",labels=imena)


