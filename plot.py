import matplotlib.pyplot as plt
import re
import numpy as np

def read_last_line(filename):
    with open(filename, 'rb') as file:
        file.seek(-2, 2)  # Go to the second last byte.
        while file.read(1) != b'\n':
            file.seek(-2, 1)  # Step backwards until a newline is found.
        last_line = file.readline().decode()
    return last_line

def extract_accuracy_pairs(filename):
    log_line = read_last_line(filename)

    # Regular expression to find the accuracy array and extract pairs of floats
    accuracy_pattern = re.compile(r"'accuracy': \[\((.*?)\)\]")

    # Find the accuracy array
    accuracy_match = accuracy_pattern.search(log_line)

    if accuracy_match:
        accuracy_str = accuracy_match.group(1)
        # Regular expression to extract pairs of floats
        pair_pattern = re.compile(r'\(([^)]+)\)')
        pairs = pair_pattern.findall(accuracy_str)
        # Convert extracted strings to tuples of floats
        accuracy_pairs = [tuple(map(float, pair.split(','))) for pair in pairs]
        return accuracy_pairs
    else:
        return []

# filenames = [
#     'outputs/2024-06-18/14-27-03/main.log',
#     'outputs/2024-06-18/14-27-03/main.log',
#     'outputs/2024-06-18/13-51-33/main.log',
#     'outputs/2024-06-18/14-06-24/main.log',
#     'outputs/2024-06-18/14-27-03/main.log',
#     'outputs/2024-06-18/15-01-23/main.log',
#     'outputs/2024-06-18/15-12-40/main.log',
#     'outputs/2024-06-18/15-35-19/main.log'
# ]

colours = ['b', 'y', 'g', 'r', 'orange', 'pink', 'c', 'm']
markers = ['x', '^', '*', '+', 's']

def plot_data(filenames, title):
    data = [extract_accuracy_pairs(filename) for filename in filenames]
    timestamps_arr = [[t[0] for t in data_for_one] for data_for_one in data]
    accuracies_arr = [[a[1] for a in data_for_one] for data_for_one in data]

    # print(accuracies_arr)
    acc = [accuracies[-1] for accuracies in accuracies_arr]
    print(acc)
    
    print(f"{np.mean(acc):.3f}")
    print(f"{np.std(acc):.3f}")

    for timestamps in timestamps_arr:
        timestamps.insert(0, 0)
    for accuracies in accuracies_arr:
        accuracies.insert(0, 0)

    plt.figure(figsize=(12, 6))
    plt.ylim(0, 1)

    plt.tick_params(axis='both', which='major', labelsize=14)  # Tick labels font size

    c = 1

    for (timestamps, accuracies, colour, marker) in zip(timestamps_arr, accuracies_arr, colours * 2, markers):
        plt.plot(timestamps, accuracies, marker=marker, linestyle='-', color=colour, label=f"Simulation {c}")
        c = c + 1
    # Format the plot
    # plt.title(f'Accuracy over Time {title}', fontsize=24)
    plt.xlabel('Timestamp', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.grid(True)
    plt.legend()

    plt.savefig(f'{title}.pdf', format='pdf')
    # Show the plot
    plt.show()

# filenames = [
#     'outputs/2024-06-18/16-33-58/main.log',
#     'outputs/2024-06-18/16-42-35/main.log',
#     'outputs/2024-06-18/16-50-45/main.log',
#     'outputs/2024-06-18/16-59-57/main.log',
#     'outputs/2024-06-18/17-08-19/main.log',
#     'outputs/2024-06-18/17-16-40/main.log',

# # filenames = [
#     'outputs/2024-06-18/17-16-40/main.log',
#     # 'outputs/2024-06-18/17-23-50/main.log',
#     'outputs/2024-06-18/17-31-24/main.log',
#     'outputs/2024-06-18/17-51-05/main.log',
#     'outputs/2024-06-18/18-02-12/main.log',
#     'outputs/2024-06-18/18-13-12/main.log',
#     'outputs/2024-06-18/18-23-39/main.log',
#     'outputs/2024-06-18/18-32-58/main.log'
# ]

# filenames = [
#     'outputs/2024-06-18/19-15-15/main.log',
#     'outputs/2024-06-18/19-24-41/main.log',
#     'outputs/2024-06-18/19-30-48/main.log',
#     'outputs/2024-06-18/20-23-10/main.log',
#     'outputs/2024-06-18/20-32-04/main.log',
#     'outputs/2024-06-18/20-44-16/main.log',

# # filenames = [
#     'outputs/2024-06-18/21-16-27/main.log',
#     # 'outputs/2024-06-18/17-23-50/main.log',
#     'outputs/2024-06-18/21-25-55/main.log'
#     # 'outputs/2024-06-18/17-51-05/main.log',
#     # 'outputs/2024-06-18/18-02-12/main.log',
#     # 'outputs/2024-06-18/18-13-12/main.log',
#     # 'outputs/2024-06-18/18-23-39/main.log',
#     # 'outputs/2024-06-18/18-32-58/main.log'
# ]

# filenames = [
#     'outputs/2024-06-19/13-27-13/main.log',
#     'outputs/2024-06-19/13-33-41/main.log',
#     'outputs/2024-06-19/13-40-34/main.log',
#     'outputs/2024-06-19/13-50-45/main.log',
#     'outputs/2024-06-19/14-01-42/main.log'
# ]

# filenames = [
#     'outputs/2024-06-19/14-25-02/main.log',
#     'outputs/2024-06-19/14-43-06/main.log',
#     'outputs/2024-06-19/14-55-40/main.log',
#     'outputs/2024-06-19/15-02-37/main.log',
#     'outputs/2024-06-19/15-11-05/main.log',
#     'outputs/2024-06-19/15-22-08/main.log',
#     'outputs/2024-06-19/15-29-21/main.log'
# ]


### eval every 5, end at 150/175

## no schedule, epoch=1, lr = lr / (1.05) ** s_age
# filenames = [
#     'outputs/2024-06-19/15-43-28/main.log',
#     'outputs/2024-06-19/15-47-16/main.log',
#     'outputs/2024-06-19/15-51-01/main.log',
#     'outputs/2024-06-19/15-54-41/main.log',
#     'outputs/2024-06-19/15-58-30/main.log',
#     'outputs/2024-06-19/15-58-30/main.log'
# ]

## fixed schedule, epoch=1, lr = lr / (1.05) ** s_age
filenames = [
    'outputs/2024-06-19/16-06-11/main.log',
    'outputs/2024-06-19/16-19-26/main.log',
    'outputs/2024-06-19/16-22-51/main.log',
    'outputs/2024-06-19/16-26-12/main.log',
    'outputs/2024-06-19/16-29-34/main.log'
]

# plot_data(filenames, "other")


### eval every 2, end at 175    17.14
## fixed schedule, epoch=1, lr=....
# filenames = [
#     'outputs/2024-06-19/17-14-50/main.log',
#     'outputs/2024-06-19/17-18-31/main.log',
#     'outputs/2024-06-19/17-22-26/main.log',
#     'outputs/2024-06-19/17-26-13/main.log',
#     'outputs/2024-06-19/17-30-04/main.log'
# ]

filenames_no_prec = [
    'outputs/2024-06-19/17-37-42/main.log',
    'outputs/2024-06-19/17-41-24/main.log',
    'outputs/2024-06-19/17-45-12/main.log',
    'outputs/2024-06-19/17-49-03/main.log',
    'outputs/2024-06-19/17-52-50/main.log'
]

filenames_prec = [
    'outputs/2024-06-19/19-03-14/main.log',
    'outputs/2024-06-19/19-37-47/main.log',
    'outputs/2024-06-19/19-41-32/main.log',
    'outputs/2024-06-19/19-45-09/main.log',
    'outputs/2024-06-19/19-48-54/main.log',
    'outputs/2024-06-19/19-58-37/main.log',
    'outputs/2024-06-20/11-41-42/main.log',
    'outputs/2024-06-20/11-49-28/main.log'
]

# slight shuffle 
filenames_prec = [
    'outputs/2024-06-20/12-00-10/main.log',
    'outputs/2024-06-20/12-05-00/main.log',
    'outputs/2024-06-20/12-09-52/main.log',
    'outputs/2024-06-20/12-14-11/main.log',
    'outputs/2024-06-20/12-18-08/main.log'
]

# more shuffle 
filenames_prec = [
    'outputs/2024-06-20/12-34-53/main.log',
    'outputs/2024-06-20/12-38-25/main.log',
    'outputs/2024-06-20/12-42-07/main.log',
    'outputs/2024-06-20/12-45-47/main.log',
    'outputs/2024-06-20/12-49-42/main.log'
]

# 13.04
# func shuffle 
filenames_prec = [
    'outputs/2024-06-20/13-03-59/main.log',
    'outputs/2024-06-20/13-09-30/main.log',
    'outputs/2024-06-20/13-13-55/main.log',
    'outputs/2024-06-20/13-18-21/main.log',
    'outputs/2024-06-20/13-22-24/main.log'
]

filenames_no_prec_iid = [
    'outputs/2024-06-22/16-13-40/main.log',
    'outputs/2024-06-22/16-17-32/main.log',
    'outputs/2024-06-22/16-21-18/main.log',
    'outputs/2024-06-22/16-26-07/main.log',
    'outputs/2024-06-22/16-30-59/main.log'

]

filenames_prec_what = [
    'outputs/2024-06-22/17-09-29/main.log',
    'outputs/2024-06-22/17-13-18/main.log',
    'outputs/2024-06-22/17-17-12/main.log',
    'outputs/2024-06-22/17-21-04/main.log',
    'outputs/2024-06-22/17-25-33/main.log'

]

plot_data(filenames_prec_what, "IID")
# exit()

plot_data(filenames_no_prec, "Under Precedence Constraints")
# plot_data(filenames_prec, "With Precedence Constraints")

