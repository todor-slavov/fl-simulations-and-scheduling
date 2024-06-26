import random
import numpy as np
import time

class Job:
    def __init__(self, job_id, duration, predecessors=[]):
        self.job_id = job_id
        self.duration = duration
        self.predecessors = predecessors
        self.start_time = None
        self.end_time = None
        self.machine = None
    
    def __str__(self):
        return f'{self.job_id}: {self.duration} {self.predecessors}'
class Scheduler:
    def __init__(self, num_machines, jobs):
        self.num_machines = num_machines
        self.jobs = jobs
        self.population_size = 500
        self.generations = 10
        self.mutation_rate = 0.1

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self.generate_valid_individual()
            population.append(individual)
        return population

    def generate_valid_individual(self):
        individual = []
        available_jobs = list(self.jobs)
        scheduled_jobs = set()
        
        while available_jobs:
            valid_jobs = [job for job in available_jobs if all(pred in scheduled_jobs for pred in job.predecessors)]
            if not valid_jobs:
                break
            job = random.choice(valid_jobs)
            individual.append(job)
            available_jobs.remove(job)
            scheduled_jobs.add(job.job_id)
        
        return individual

    def fitness(self, individual):
        schedule = {machine: [] for machine in range(self.num_machines)}
        time_tracker = {machine: 0 for machine in range(self.num_machines)}

        job_dict = {job.job_id: job for job in individual}

        for job in individual:
            earliest_start = 0
            for pred in job.predecessors:
                pred_job = job_dict.get(pred)
                if not pred_job or pred_job.end_time is None:
                    return float('inf'), {}

                earliest_start = max(earliest_start, pred_job.end_time)

            min_machine = min(time_tracker, key=time_tracker.get)
            job.start_time = max(earliest_start, time_tracker[min_machine])
            job.end_time = job.start_time + job.duration
            job.machine = min_machine

            time_tracker[min_machine] = job.end_time
            schedule[min_machine].append(job)

        makespan = max(time_tracker.values())
        return makespan, schedule

    def select_parents(self, population):
        sorted_population = sorted(population, key=lambda ind: self.fitness(ind)[0])
        return sorted_population[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        max_attempts = 10
        for _ in range(max_attempts):
            cut = random.randint(0, len(parent1) - 1)
            child = parent1[:cut] + [job for job in parent2 if job not in parent1[:cut]]
            if self.is_valid_schedule(child):
                return child
        return parent1  # Return one of the parents if a valid child is not produced after max_attempts

    def mutate(self, individual):
        max_retries = 10
        for _ in range(max_retries):
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            if self.is_valid_schedule(individual):
                break

    def is_valid_schedule(self, individual):
        scheduled_jobs = set()
        for job in individual:
            if all(pred in scheduled_jobs for pred in job.predecessors):
                scheduled_jobs.add(job.job_id)
            else:
                return False
        return True

    def evolve_population(self, population):
        new_population = []
        parents = self.select_parents(population)
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        return new_population

    def solve(self):
        start_time = time.time()
        population = self.initialize_population()
        best_solution = None
        best_makespan = float('inf')

        for generation in range(self.generations):
            population = self.evolve_population(population)
            for individual in population:
                makespan, schedule = self.fitness(individual)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_solution = schedule

        print(f"EA took {time.time() - start_time} to finish.")
        return best_solution

def read_jobs(file):
    with open(f'{file}', 'r') as file:
        lines = file.readlines()

    # Extract number of jobs
    num_jobs = int(lines[0].split(':')[1].strip())
    print(f"Number of jobs: {num_jobs}")

    # Extract processing times
    processing_times_str = lines[1].strip().split(', ')
    processing_times_str[-1] = processing_times_str[-1][:-1]

    # Remove any empty strings or trailing commas
    processing_times_str = [time for time in processing_times_str if time and time.isdigit()]
    processing_times = np.array([int(time) for time in processing_times_str], dtype=int)

    # Extract precedence constraints
    precedence_constraints_str = lines[4].strip().split('), (')
    precedence_constraints = []
    precedence_constraints_str[0] = precedence_constraints_str[0][1:]
    precedence_constraints_str[-1] = precedence_constraints_str[-1][:-2]

    for constraint in precedence_constraints_str:
        constraint = constraint.replace('(', '').replace(')', '').strip()
        if constraint:
            precedence_constraints.append(tuple(map(int, constraint.split(', '))))

    # Create jobs array
    jobs = []
    for job_id in range(num_jobs):
        dependencies = [pc[0] + 1 for pc in precedence_constraints if pc[1] == job_id]
        job = Job(job_id + 1, processing_times[job_id], dependencies)
        jobs.append(job)

    return jobs



jobs = read_jobs("1000.pdf")

start_time = time.time()
scheduler = Scheduler(num_machines=3, jobs=jobs)
solution = scheduler.solve()

m = 0
m_ = 1000000

for machine, jobs in solution.items():
    for job in jobs:
        m_ = min(m_, job.start_time)
        m = max(m, job.end_time)
print(f"Makespan: {m - m_}")