import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import glob
import re
from os.path import join


def plot_concatinate_files(files_dir, log_file_list):
    best_in_pop_ptrn = re.compile("best in the population ---->")
    stabilize_ptrn = re.compile("best in the population after stabilization")
    activation_ptrn = re.compile("activation average")
    float_ptrn = re.compile("-*\d+.\d*")

    best_val_generation = []
    best_stabile_generation = []
    best_activation_generation = []

    for log_file in log_file_list:
        src_path = join(files_dir, log_file)

        with open(src_path, "r") as src_file:
            for log_line in src_file:
                best_line = best_in_pop_ptrn.search(log_line)
                if best_line is not None:
                    val = float_ptrn.findall(log_line)[0]
                    best_val_generation.append(float(val))

                best_stabile_line = stabilize_ptrn.search(log_line)
                if best_stabile_line is not None:
                    val = float_ptrn.findall(log_line)[0]
                    best_stabile_generation.append(float(val))

                best_activation_line = activation_ptrn.search(log_line)
                if best_activation_line is not None:
                    val = float_ptrn.findall(log_line)[0]
                    best_activation_generation.append(float(val))

    return best_val_generation, best_stabile_generation, best_activation_generation


def plot_graph(graphs, labels, color_list, x_lbl, y_lbl, title):
    for gr, clr, i in zip(graphs, color_list, range(10)):
        if len(gr) > 250:
            gr = gr[:250]
        else:
            print(i)
            print(len(gr))
            print("----------")
        if i == 0:
            plt.plot(range(len(gr)), gr, label="MLIN", color=clr)
        if i == 6:
            plt.plot(range(len(gr)), gr, label=" no MLIN", color=clr)
        else:
            plt.plot(range(len(gr)), gr, color=clr)

    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def main():

    files_dir = "/Users/djgr/code/instincts/modular_rl/trained_models/pulled_from_server/second_phase_instinct/n_deterministic_goals/models4paper/logs/"

    best_vals = []
    best_activations = []
    numbers = [
        ### Get IMN fitness
        plot_concatinate_files(
            files_dir, [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_1_part{prt}.log" for prt in [1, 2, 3]]
        ),
        plot_concatinate_files(
            files_dir,
            [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_2_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir,
            [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_3_part{prt}.log" for prt in [1, 2, 3]]
        ),
        plot_concatinate_files(
            files_dir,
            [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_4_part{prt}.log" for prt in [1]]
        ),
        plot_concatinate_files(
            files_dir,
            [f"instinctual_network_module/EVOLUTION_lidar_smallExp_run_5_part{prt}.log" for prt in [1,2]]
        ),

        # Get CTRL fitness
        plot_concatinate_files(
            files_dir, [f"control/EVOLUTION_lidar_control_run_1_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir, [f"control/EVOLUTION_lidar_control_run_2_part{prt}.log" for prt in [1, 2, 3, 4]]
        ),
        plot_concatinate_files(
            files_dir, [f"control/EVOLUTION_lidar_control_run_3_part{prt}.log" for prt in [1, 2]]
        ),
        plot_concatinate_files(
            files_dir, [f"control/EVOLUTION_lidar_control_run_4_part{prt}.log" for prt in [1, 2, 3]]
        ),
        plot_concatinate_files(
            files_dir, [f"control/EVOLUTION_lidar_control_run_5_part{prt}.log" for prt in [1, 2]]
        ),
    ]

    numbers = list(zip(*numbers))

    plot_graph(
        numbers[0],
        ["MLIN fitness", "instinct run2", "instinct run3", "no instinct run1", "no instinct run2", "no instinct run3"],
        ["blue", "blue", "blue", "blue", "blue", "orange", "orange", "orange", "orange", "orange"],
        "generation",
        "fitness",
        "best individual fitness",
    )

if __name__ == "__main__":
    main()

