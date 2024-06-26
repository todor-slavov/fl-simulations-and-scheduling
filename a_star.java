import java.util.List;
import java.util.ArrayList;
import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

class Main {

    public static State findOptimalSchedule(Schedule solution, long totalDuration) {
        PriorityQueue<State> queue = new PriorityQueue<>((a, b) -> (int) (a.priority - b.priority));
        State initialState = new State(solution.getNumberOfMachines());
        queue.add(initialState);
        long c = 0;
        while (!queue.isEmpty()) {

            State currentState = queue.remove();
//            if (c < 1000)
            long s = currentState.priority;


//            System.out.println("csp: " + currentState.priority);
//            if (queue.size() < 20) {
//                System.out.println(queue);
//            } else {
//                System.exit(0);
//            }

            c ++;
            List<Job> availableJobs = getAvailableJobs(solution.getJobs(), currentState);
            if (availableJobs.isEmpty()) {
                System.out.println("Expanded nodes: " + c);
                return currentState; // All jobs have been scheduled
            }

            for (Job job : availableJobs) {
                for (int i = 0; i < solution.getNumberOfMachines(); i++) {
                    State newState = currentState.copy();
//                    if (currentState.priority != s)
//                        System.out.println("KURVI I BELO");
                    int startTime = earliestStartTime(newState, job, i);
                    JobSet jobSet = new JobSet(startTime, i, job);
                    newState.getMachineSchedules()[i].add(jobSet);
                    newState.setF(totalDuration);
//                    System.out.println(newState.priority);
//                    System.out.printf("%s -> %s\n", currentState, newState);
//                    System.out.printf("%d -> %d\n", currentState.priority, newState.priority);
                    queue.add(newState);
                }
            }
        }

        return null; // No valid schedule found
    }

    public static int f(State state, long totalDuration) {
        return heuristicLongestChain(state);
//        return heuristic(state, totalDuration);
//        return makespan(state);
    }

    private static int makespan(State state) {
        int maxFinishTime = 0;
        for (List<JobSet> schedule : state.getMachineSchedules()) {
            for (JobSet jobSet : schedule) {
                maxFinishTime = Math.max(maxFinishTime, jobSet.getFinishTime());
            }
        }
        return maxFinishTime;
    }

    private static int heuristicLongestChain(State state) {
//        int heuristicValue = 0;
        int duration = 0;
        int best = 0;
        for (List<JobSet> schedule : state.getMachineSchedules()) {
            if (!schedule.isEmpty()) {
                JobSet jobLast = schedule.get(schedule.size() - 1);
                duration = Math.max(duration, jobLast.getJob().getHeuristic() + jobLast.getFinishTime());
            }
        }

        return duration;
    }

    private static int heuristic(State state, long totalDuration) {
        int heuristicValue = 0;
        int duration = 0;

        int latest = 0;

        for (List<JobSet> schedule : state.getMachineSchedules()) {
            for (JobSet jobSet : schedule) {
                latest = Math.max(latest, jobSet.getFinishTime());

                duration = Math.max(duration, jobSet.getFinishTime());
                heuristicValue = Math.max(heuristicValue, jobSet.getJob().getHeuristic());
            }
        }

        long free = 0;
        long filled = 0;
        for (List<JobSet> schedule : state.getMachineSchedules()) {
//            System.out.println(schedule);
            long latestHere = 0;
            for (JobSet jobSet : schedule) {
                filled += jobSet.getDuration();
                latestHere = Math.max(latestHere, jobSet.getFinishTime());

                if (jobSet.getFinishTime() > duration) {
                    heuristicValue = jobSet.getJob().getHeuristic();
                }
                duration = Math.max(duration, jobSet.getFinishTime());

                //                heuristicValue = Math.max(heuristicValue, jobSet.getJob().getHeuristic());
            }
            free += (latest - latestHere);
        }
//        System.out.println("free: " + free);
//        System.out.println("filled: " + filled);
//        System.out.println(free);
//        System.out.println(Math.max(0, Math.floor((totalDuration - filled - free)) / 2));
        return (int) (duration + Math.max(0, Math.floor((totalDuration - filled - free)) / 2));
//        System.out.println(heuristicValue);
//        return 0;
//        return 0;
//        return duration;
//        return heuristicValue + duration;
    }

//    Expanded nodes: 4341836
//            20103
//            38

    private static List<Job> getAvailableJobs(List<Job> jobs, State state) {
        List<Job> availableJobs = new ArrayList<>();
        for (Job job : jobs) {
            boolean allPrecedencesMet = true;
            for (Job precedence : job.getPrecedences()) {
                if (!isJobScheduled(precedence, state)) {
                    allPrecedencesMet = false;
                    break;
                }
            }
            if (allPrecedencesMet && !isJobScheduled(job, state)) {
                availableJobs.add(job);
            }
        }
        return availableJobs;
    }

    private static boolean isJobScheduled(Job job, State state) {
        for (List<JobSet> schedule : state.getMachineSchedules()) {
            for (JobSet jobSet : schedule) {
                if (jobSet.getId() == job.getId()) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int earliestStartTime(State state, Job job, int machineId) {
        int maxPrecedenceFinishTime = 0;
        for (Job precedence : job.getPrecedences()) {
            int precedenceFinishTime = getJobFinishTime(precedence, state);
            maxPrecedenceFinishTime = Math.max(maxPrecedenceFinishTime, precedenceFinishTime);
        }
        int machineAvailableTime = getMachineAvailableTime(state, machineId);
        return Math.max(maxPrecedenceFinishTime, machineAvailableTime);
    }

    private static int getJobFinishTime(Job job, State state) {
        for (List<JobSet> schedule : state.getMachineSchedules()) {
            for (JobSet jobSet : schedule) {
                if (jobSet.getId() == job.getId()) {
                    return jobSet.getFinishTime();
                }
            }
        }
        return 0;
    }

    private static int getMachineAvailableTime(State state, int machineId) {
        List<JobSet> schedule = state.getMachineSchedules()[machineId];
        if (schedule.isEmpty()) {
            return 0;
        }
        return schedule.get(schedule.size() - 1).getFinishTime();
    }

    public static void main(String[] args) {
         List<Job> jobs = new ArrayList<>();

         try {
//             jobs = JobParser.parseJobs("C:\\Users\\Todor\\AppData\\Roaming\\JetBrains\\IdeaIC2021.2\\scratches\\test.txt");
             jobs = JobParser.parseJobs("C:\\Users\\Todor\\AppData\\Roaming\\JetBrains\\IdeaIC2021.2\\scratches\\4.txt");
//             jobs = JobParser.parseJobs("C:\\Users\\Todor\\AppData\\Roaming\\JetBrains\\IdeaIC2021.2\\scratches\\dag_output (3).txt");
//             jobs = JobParser.parseJobs("C:\\Users\\Todor\\AppData\\Roaming\\JetBrains\\IdeaIC2021.2\\scratches\\dag_output.txt");
         } catch (IOException e) {
             e.printStackTrace();
         }
        System.out.println(jobs.size());

        // Output successors
//        for (Job job : jobs) {
//            System.out.print("Job " + job.getId() + " successors: ");
//            for (Job successor : job.getSuccessors())
//                System.out.print(successor.getId() + " ");
//            }
//            System.out.println();
//        }
//
        calculateHeuristic(jobs);
//
//        for (Job job : jobs) {
//            System.out.println(job.getId() + ": " + job.getHeuristic());
//        }

        long start = System.currentTimeMillis();
        Schedule solution = new Schedule(2, jobs);
        long totalDuration = 0;
        for (Job job : jobs)
            totalDuration += job.getDuration();
        State optimalSchedule = findOptimalSchedule(solution, totalDuration);
        System.out.println(System.currentTimeMillis() - start);

        // Output the schedule

        long maxTime = 0;

        for (int i = 0; i < solution.getNumberOfMachines(); i++) {
//            System.out.println("Machine " + (i + 1) + ":");
            for (JobSet jobSet : optimalSchedule.getMachineSchedules()[i]) {
                maxTime = Math.max(jobSet.getFinishTime(), maxTime);
//                System.out.println("  Job " + jobSet.getId() + " (Start: " + jobSet.getStartTime() + ", Finish: " + jobSet.getFinishTime() + ")");
            }
        }
        System.out.println(maxTime);


    }

    public static void calculateHeuristic(List<Job> jobs) {
        for (int i = jobs.size() - 1; i != -1; i --) {
//            System.out.println(dfs(jobs.get(i)) - jobs.get(i).getDuration());
            jobs.get(i).setHeuristic(dfs(jobs.get(i)));
        }
    }

    public static int dfs(Job job) {
        if (job.getSuccessors().size() == 0) {
            return 0;
        }
        if (job.getHeuristic() != 0) {
            return job.getHeuristic();
        }
        if (job.getSuccessors().size() == 1) {
            return dfs(job.getSuccessors().get(0)) + job.getSuccessors().get(0).getDuration();
        }
        return Math.max(dfs(job.getSuccessors().get(0)) + job.getSuccessors().get(0).getDuration(),
                dfs(job.getSuccessors().get(1)) + job.getSuccessors().get(1).getDuration());
    }
}

class JobParser {

    public static List<Job> parseJobs(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;

        // Parse the number of nodes (jobs)
        line = reader.readLine();
        int numberOfNodes = Integer.parseInt(line.split(": ")[1]);

        // Parse job durations
        line = reader.readLine();
        String[] durations = line.split(", ");//[1].split(", ");

        List<Job> jobs = new ArrayList<>();
        Map<Integer, Job> jobMap = new HashMap<>();
        for (int i = 0; i < numberOfNodes; i++) {
            Job job = new Job(i, Integer.parseInt(durations[i]), new ArrayList<>(), new ArrayList<>());
            jobs.add(job);
            jobMap.put(i, job);
        }

        // Parse precedence constraints
        line = reader.readLine();
        line = reader.readLine(); // skip "Precedence constraints:" line
        line = reader.readLine();
        String[] constraints = line.split("\\), \\(");

        if (constraints.length > 0 && constraints[0].startsWith("(")) {
            constraints[0] = constraints[0].substring(1); // Remove leading '(' from first element
            constraints[constraints.length - 1] = constraints[constraints.length - 1].substring(0, constraints[constraints.length - 1].length() - 2); // Remove trailing ')' from last element
        }
        for (String constraint : constraints) {
//            System.out.println(constraint);
            String[] nodes = constraint.split(", ");
            if (nodes[1].contains(",") || nodes[1].contains("\\)")) {
//                System.out.println("KURVI");
                nodes[1] = nodes[1].split("\\),")[0];
            }
//            System.out.println(nodes[0]);
//            System.out.println(nodes[1]);
            int from = Integer.parseInt(nodes[0]);
            int to = Integer.parseInt(nodes[1]);
            try {
                jobMap.get(to).getPrecedences().add(jobMap.get(from));
                jobMap.get(from).getSuccessors().add(jobMap.get(to));
            } catch (Exception e) {

            }

        }

        reader.close();
        return jobs;
    }
}

class Job {
    private int id;
    private int duration;
    private int heuristic;
    private List<Job> precedences;
    private List<Job> successors;

    public Job(int id, int duration, List<Job> precedences, List<Job> successors) {
        this.id = id;
        this.duration = duration;
        this.precedences = precedences;
        this.successors = successors;
    }

    public int getId() {
        return id;
    }

    public int getDuration() {
        return duration;
    }

    public int getHeuristic() {
        return heuristic;
    }

    public List<Job> getPrecedences() {
        return precedences;
    }

    public List<Job> getSuccessors() {
        return successors;
    }

    public void setHeuristic(int heuristic) {
        this.heuristic = heuristic;
    }
}

class Schedule {
    private int numberOfMachines;
    private List<Job> jobs;

    public Schedule(int numberOfMachines, List<Job> jobs) {
        this.numberOfMachines = numberOfMachines;
        this.jobs = jobs;
    }

    public int getNumberOfMachines() {
        return numberOfMachines;
    }

    public List<Job> getJobs() {
        return jobs;
    }
}

class JobSet {
    // private int id;
    // private int duration;
    private int startTime;
    private int finishTime;
    private int machineId;
    // private List<Job> precedenceConstraints;
    private Job job;

    public JobSet(int startTime, int machineId, Job job) {
        this.job = job;
        this.startTime = startTime;
        this.finishTime = startTime + job.getDuration();
        this.machineId = machineId;
    }
    // public JobSet(int id, int duration, int startTime, int machineId, List<Job> precedenceConstraints) {
    //     this.id = id;
    //     this.duration = duration;
    //     this.startTime = startTime;
    //     this.finishTime = startTime + duration;
    //     this.machineId = machineId;
    //     this.precedenceConstraints = precedenceConstraints;
    // }

    public int getId() {
        return job.getId();
    }

    @Override
    public String toString() {
        return "JobSet{" +
                "startTime=" + startTime +
                ", finishTime=" + finishTime +
                ", machineId=" + machineId +
                ", job=" + job +
                '}';
    }

    public int getDuration() {
        return job.getDuration();
    }

    public int getStartTime() {
        return startTime;
    }

    public int getFinishTime() {
        return finishTime;
    }

    public int getMachineId() {
        return machineId;
    }

    public List<Job> getPrecedenceConstraints() {
        return job.getPrecedences();
    }

    public Job getJob() {
        return job;
    }
}

class State {
    long priority;
    private List<JobSet>[] machineSchedules;

    @SuppressWarnings("unchecked")
    public State(int numberOfMachines) {
        machineSchedules = new ArrayList[numberOfMachines];
        for (int i = 0; i < numberOfMachines; i++) {
            machineSchedules[i] = new ArrayList<>();
        }
    }

    public void setF(long totalDuration) {
        priority = Main.f(this, totalDuration);
    }


    @Override
    public String toString() {
        return "State{" +
                "priority=" + priority +
                ", machineSchedules=" + Arrays.toString(machineSchedules) +
                '}';
    }

    public List<JobSet>[] getMachineSchedules() {
        return machineSchedules;
    }

    public State copy() {
        State newState = new State(machineSchedules.length);
        for (int i = 0; i < machineSchedules.length; i++) {
            newState.machineSchedules[i].addAll(machineSchedules[i]);
        }
        return newState;
    }
}
