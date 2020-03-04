import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import glob
import re
from os.path import join


def plot_concatinate_files(files_dir, log_file_list):
    generation_ptrn = re.compile("Generation")
    distance_fit_ptrn = re.compile("Distance fitness")
    violation_fit_ptrn= re.compile("NOGO fitness")
    float_ptrn = re.compile("-*\d+.\d*")
    int_ptrn = re.compile("-*\d+")

    generations = []
    distance_fits = []
    violation_fits = []

    for log_file in log_file_list:
        src_path = join(files_dir, log_file)

        with open(src_path, "r") as src_file:
            for log_line in src_file:
                gen_line = generation_ptrn.search(log_line)
                if gen_line is not None:
                    val = int_ptrn.findall(log_line)[0]
                    generations.append(int(val))

                distance_fit_line = distance_fit_ptrn.search(log_line)
                if distance_fit_line is not None:
                    val = float_ptrn.findall(log_line)[0]
                    distance_fits.append(float(val))

                violation_fit_line = violation_fit_ptrn.search(log_line)
                if violation_fit_line is not None:
                    val = int_ptrn.findall(log_line)[0]
                    violation_fits.append(float(val))

    return generations, distance_fits, violation_fits


def plot_graph(graphs, labels, color_list, x_lbl, y_lbl, title):
    for gr, lbl, clr in zip(graphs, labels, color_list):
        if len(gr) > 250:
            gr = gr[:250]
        plt.plot(range(len(gr)), gr, label=lbl, color=clr)

    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def main():

    files_dir = "/Users/djgr/code/instincts/modular_rl/trained_models/pulled_from_server/second_phase_instinct/n_deterministic_goals/4goals_lidar_ctrl_noCoordiantes/balance_plot_visualisation/"
    file_name = "output.txt"

    gens, dist_fits, violation_fits = plot_concatinate_files(files_dir, [file_name])

    plt.plot(gens, dist_fits, label="Fitness associated with distance to goal")
    plt.plot(gens, violation_fits, label="Fitness associated with hazard zone violations")

    plt.xlabel("Generations")
    plt.ylabel("")
    plt.title("Decoupled fitness over generations")
    plt.ylim(plt.ylim()[::-1])
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()
