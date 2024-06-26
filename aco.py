import numpy as np
import math

# Define problem-specific parameters
num_machines = 2

# num_jobs = 5
# processing_times = np.random.randint(1, 10, size=(num_jobs, num_machines))
# precedence_constraints = [
#     (0, 1),  # Job 0 must precede job 1
#     (1, 2),  # Job 1 must precede job 2
#     (1, 3),  # Job 2 must precede job 3
#     (2, 4),  # Job 3 must precede job 4
#     (3, 4),  # Job 3 must precede job 4
#     # Add more constraints as needed
# ]


with open('dag_output.txt', 'r') as file:
    lines = file.readlines()

# Extract number of jobs
num_jobs = int(lines[0].split(':')[1].strip())
print(num_jobs)

# Extract processing times
processing_times_str = lines[1].strip().split(', ')
processing_times_str[-1] = processing_times_str[-1][:-1]

# Remove any empty strings or trailing commas
processing_times_str = [time for time in processing_times_str if time and time.isdigit()]
processing_times = np.array([int(time) for time in processing_times_str], dtype=int)

# Reshape the processing_times array to have 'num_jobs' rows and '1' column (since each job has only one processing time)
# processing_times = processing_times.reshape(num_jobs, 1)

# Extract precedence constraints
precedence_constraints_str = lines[4].strip().split('), (')
precedence_constraints = []
precedence_constraints_str[0] = precedence_constraints_str[0][1:]
precedence_constraints_str[-1] = precedence_constraints_str[-1][:-2]
# print(precedence_constraints_str)
for constraint in precedence_constraints_str:
    constraint = constraint.replace('(', '').replace(')', '').strip()
    if constraint:
        precedence_constraints.append(tuple(map(int, constraint.split(', '))))

# Print the extracted values for verification
# print(f"Number of jobs: {num_jobs}")
# print(f"Processing times:\n{processing_times}")
# print(f"Precedence constraints: {precedence_constraints}")

# exit()


# Initialize ACO parameters
num_ants = 10
num_iterations = 10
evaporation_rate = 0.5
alpha = 1
beta = 2
initial_pheromone = 1

# Initialize pheromone matrix
pheromone = np.ones((num_jobs)) * initial_pheromone

# Heuristic information (e.g., inverse of processing times)
heuristic = 1 / (processing_times + 1e-6)

def get_ready_jobs(end_times, solution):
    ready = []
    
    for machine, jobs in enumerate(solution):
        for job in jobs:
            if (job in end_times.keys()):
                continue

            flag = True

            last_job = None

            for (job_a, job_b) in precedence_constraints:
                if (job_b == job):
                    if (job_a not in end_times.keys()):
                        flag = False
                        break
                        # we can't execute this job yet

                    else:
                        if last_job is None:
                            last_job = job_a
                        if end_times[job_a] > end_times[last_job]:
                            last_job = job_a
            if (flag):
                # ready.append(job)
                ready.append((last_job, job))
                if len(ready) == num_machines:
                    return ready
    
    return ready

def get_last(job, end_times):
    time = 0
    for (job_a, job_b) in precedence_constraints:
        if (job_b == job and job_a in end_times.keys()):
            time = max(end_times[job_a], time)
    
    return time

def get_machine(job, solution):
    for (machine, j) in enumerate(solution):
        if job in j:
            return machine
    return -1

def calculate_makespan(solution):
    if solution is None:
        return float('inf'), {}, {}

    start_times = {}
    end_times = {}
    machine_finish_times = [0] * num_machines

    while(True):
        ready = get_ready_jobs(end_times, solution)
        # print(ready)
        if (len(ready) == 0):
            break
        
        for (last_job, job) in ready:
        # for job in ready:
            # last_job = get_last(job, end_times)
            if last_job == None:
                last_job = 0
            else:
                last_job = end_times[last_job]

            start_times[job] = last_job
            end_times[job] = last_job + processing_times[job]

            # if last_job is None:
            #     start_times[job] = 0
            #     end_times[job] = processing_times[job]
            # else:
            #     start_times[job] = processing_times[last_job]
            #     end_times[job] = processing_times[last_job] + processing_times[job]
    
    makespan = 0

    for end_time in end_times.values():
        makespan = max(makespan, end_time)

    return makespan, start_times, end_times

    # for machine, jobs in enumerate(solution):
    #     current_time = machine_finish_times[machine]
    #     for i, job in enumerate(jobs):
    #         if i > 0:
    #             prev_job = jobs[i - 1]
    #             current_time = max(current_time + setup_times[prev_job, job, machine], machine_finish_times[machine])
    #         start_times[job] = current_time
    #         current_time += processing_times[job, machine]
    #         end_times[job] = current_time
    #         machine_finish_times[machine] = current_time

    # makespan = max(machine_finish_times)
    # return makespan, start_times, end_times

def is_feasible(solution):
    return True
    # Check precedence constraints
    _, start_times, end_times = calculate_makespan(solution)
    # print(start_times)
    # print(end_times)
    for (job_a, job_b) in precedence_constraints:
        if end_times[job_a] is not None and start_times[job_b] is not None:
            if end_times[job_a] > start_times[job_b]:
                return False
    return True

def construct_solution():
    solution = [[] for _ in range(num_machines)]
    available_jobs = list(range(num_jobs))
    while available_jobs:
        # print("i")
        for machine in range(num_machines):
            if not available_jobs:
                break
            probabilities = np.zeros(len(available_jobs))
            for i, job in enumerate(available_jobs):
                probabilities[i] = pheromone[job] ** alpha * heuristic[job] ** beta
            probabilities /= probabilities.sum()
            chosen_job = np.random.choice(available_jobs, p=probabilities)
            solution[machine].append(chosen_job)
            available_jobs.remove(chosen_job)
    return solution

def local_search(solution):
    if solution is None:
        return None

    # Implement a simple local search (e.g., swap two jobs on the same machine)
    best_solution = solution
    # print("aw")
    best_makespan, _, _ = calculate_makespan(solution)
    # print("awd")
    for machine in range(num_machines):
        for i in range(len(solution[machine]) - 1):
            for j in range(i + 1, len(solution[machine])):
                # print("l")
                new_solution = [list(jobs) for jobs in solution]
                new_solution[machine][i], new_solution[machine][j] = new_solution[machine][j], new_solution[machine][i]
                if is_feasible(new_solution):
                    new_makespan, _, _ = calculate_makespan(new_solution)
                    if new_makespan < best_makespan:
                        best_solution = new_solution
                        best_makespan = new_makespan
    return best_solution

# ACO main loop
best_solution = None
best_makespan = float('inf')
best_start_times = None
best_end_times = None

# end_times = {}
# print(get_ready_jobs(end_times, [[0, 2, 3], [1, 4]]))
# end_times = {0: 12}
# print(get_ready_jobs(end_times, [[0, 2, 3], [1, 4]]))
# end_times = {0: 12, 1: 24}
# print(get_ready_jobs(end_times, [[0, 2, 3], [1, 4]]))
# end_times = {0: 12, 1: 24, 2: 25, 3: 34}
# print(get_ready_jobs(end_times, [[0, 2, 3], [1, 4]]))

# exit()

for iteration in range(num_iterations):
    print(iteration)
    solutions = []
    for ant in range(num_ants):
        # print(ant)
        solution = construct_solution()
        # print(solution)
        if is_feasible(solution):
            solutions.append(solution)
            makespan, start_times, end_times = calculate_makespan(solution)
            # print("j")
            # print(start_times)
            # print(end_times)
            if makespan < best_makespan:
                best_solution = solution
                best_makespan = makespan
                best_start_times = start_times
                best_end_times = end_times
                print(best_solution)
                print(best_makespan)

    if solutions:
        # print("s")
        for solution in solutions:
            # print(solution)
            makespan, _, _ = calculate_makespan(solution)
            for machine, jobs in enumerate(solution):
                for job in jobs:
                    pheromone[job] = (1 - evaporation_rate) * pheromone[job] + 1 / makespan
        if best_solution is not None:
            # print("local")
            best_solution = local_search(best_solution)
            # print("out")
    else:
        print(f"Iteration {iteration}: No feasible solutions found.")

print("Processing times:\n", processing_times)
print("\nBest solution:", best_solution)
print("Best makespan:", best_makespan)
print("\nStart times:", best_start_times)
print("End times:", best_end_times)
