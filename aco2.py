import numpy as np
import time

class Job:
    def __init__(self, job_id, duration, predecessors):
        self.job_id = job_id
        self.duration = duration
        self.predecessors = predecessors

    def __str__(self):
        return f'{self.job_id}: {self.duration} {self.predecessors}'

class Ant:
    def __init__(self, num_jobs):
        self.solution = []
        self.total_duration = 0
        self.unvisited_jobs = list(range(num_jobs))
        self.schedule = {}

    def add_job(self, job_id, start_time, end_time, machine_id):
        self.solution.append(job_id)
        self.total_duration = max(self.total_duration, end_time)
        self.unvisited_jobs.remove(job_id)
        self.schedule[job_id] = (start_time, end_time, machine_id)

class AntColonyOptimization:
    def __init__(self, jobs, num_machines, num_ants=10, num_iterations=5, alpha=1.0, beta=1.0, rho=0.1, q=1.0):
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = num_machines
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone = np.ones((self.num_jobs, self.num_jobs))
        self.best_solution = None
        self.best_duration = float('inf')
        self.best_schedule = None
        
    def _probability(self, current_job, next_job):
        pheromone = self.pheromone[current_job, next_job] ** self.alpha
        visibility = (1.0 / self.jobs[next_job].duration) ** self.beta
        return pheromone * visibility
    
    def _choose_next_job(self, ant):
        current_job = ant.solution[-1] if ant.solution else -1
        probabilities = []
        
        for job_id in ant.unvisited_jobs:
            if all(predecessor in ant.solution for predecessor in self.jobs[job_id].predecessors):
                probabilities.append((job_id, self._probability(current_job, job_id)))
        
        if not probabilities:
            return None
        
        jobs, probs = zip(*probabilities)
        total = sum(probs)
        probs = [prob / total for prob in probs]
        
        return np.random.choice(jobs, p=probs)
    
    def _update_pheromone(self, ants):
        self.pheromone *= (1 - self.rho)
        for ant in ants:
            for i in range(len(ant.solution) - 1):
                j1, j2 = ant.solution[i], ant.solution[i + 1]
                self.pheromone[j1, j2] += self.q / ant.total_duration
    
    def _evaluate_solution(self, ant):
        machine_times = [0] * self.num_machines
        job_start_end_machine = [-1] * self.num_jobs
        
        for job_id in ant.solution:
            job = self.jobs[job_id]
            start_time = 0
            if job.predecessors:
                start_time = max(ant.schedule[predecessor][1] for predecessor in job.predecessors)
            machine_id = np.argmin(machine_times)
            start_time = max(start_time, machine_times[machine_id])
            end_time = start_time + job.duration
            machine_times[machine_id] = end_time
            job_start_end_machine[job_id] = (start_time, end_time, machine_id)
        
        ant.schedule = {job_id: job_start_end_machine[job_id] for job_id in ant.solution}
        return max(machine_times)
    
    def run(self):
        start_time_sim = time.time()
        for iteration in range(self.num_iterations):
            ants = [Ant(self.num_jobs) for _ in range(self.num_ants)]
            
            for ant in ants:
                machine_times = [0] * self.num_machines  # Track end times of machines
                while ant.unvisited_jobs:
                    next_job = self._choose_next_job(ant)
                    if next_job is not None:
                        job = self.jobs[next_job]
                        if job.predecessors:
                            start_time = max(ant.schedule[predecessor][1] for predecessor in job.predecessors)
                        else:
                            start_time = 0
                        machine_id = np.argmin(machine_times)
                        start_time = max(start_time, machine_times[machine_id])
                        end_time = start_time + job.duration
                        machine_times[machine_id] = end_time
                        ant.add_job(next_job, start_time, end_time, machine_id)
                    else:
                        break
            
                ant.total_duration = self._evaluate_solution(ant)
                
                if ant.total_duration < self.best_duration:
                    self.best_duration = ant.total_duration
                    self.best_solution = ant.solution
                    self.best_schedule = ant.schedule
            
            self._update_pheromone(ants)
        

        end_time_sim = time.time()
        elapsed_time = end_time_sim - start_time_sim
        print(f"Time it took ACO: {elapsed_time} s")
        return self.best_solution, self.best_duration, self.best_schedule

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
        dependencies = [pc[0] for pc in precedence_constraints if pc[1] == job_id]
        job = Job(job_id, processing_times[job_id], dependencies)
        jobs.append(job)

    return jobs


jobs = read_jobs("1000.pdf")

num_machines = 3

aco = AntColonyOptimization(jobs, num_machines)
best_solution, best_duration, best_schedule = aco.run()

print("Best makespan:", best_duration)