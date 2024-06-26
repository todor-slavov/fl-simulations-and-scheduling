import heapq
import random
import numpy as np

# A simple implementation of Priority Queue using heapq
class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i[1]) for i in self.queue])

    def isEmpty(self):
        return len(self.queue) == 0

    def insert(self, node):
        heapq.heappush(self.queue, (node.end_time, node))

    def delete(self):
        if not self.isEmpty():
            return heapq.heappop(self.queue)[1]
        else:
            raise IndexError("Pop from an empty priority queue")

class Node():
    def __init__(self, time, id, start_time, n_type, client):
        self.time = time
        self.id = id
        self.start_time = start_time
        self.end_time = start_time + time
        self.prec = []
        self.succ = []
        self.n_type = n_type
        self.client = client

    def __lt__(self, other):
        return self.end_time < other.end_time

    def __le__(self, other):
        return self.end_time <= other.end_time

    def __eq__(self, other):
        return self.end_time == other.end_time

    def __ne__(self, other):
        return self.end_time != other.end_time

    def __gt__(self, other):
        return self.end_time > other.end_time

    def __ge__(self, other):
        return self.end_time >= other.end_time

    def __str__(self):
        ret = self.str_simple() + "\n"
        ret += "Predecessors: \n"
        for prec_node in self.prec:
            ret += "    " + prec_node.str_simple() + "\n"
        ret += "Successors: \n"
        for succ_node in self.succ:
            ret += "    " + succ_node.str_simple() + "\n"
        return ret
    
    def str_simple(self):
        return f"{self.n_type}-Node {self.id}: ({self.start_time} - {self.end_time}) / {self.time}."

    def add_prec(self, prec_node):
        self.prec.append(prec_node)

    def add_succ(self, succ_node):
        self.succ.append(succ_node)

def generate_dag(num_clients: int, means: [int], vars: [int], number_of_tasks: int):
    pq = PriorityQueue()
    client_nodes = []
    server_nodes = []
    nodes = []

    server_end_time = 0

    for client in range(num_clients):
        server_time = max(1, int(random.gauss(1, 0)))
        server_node = Node(server_time, len(client_nodes) + len(server_nodes), server_end_time, "S", -1)
        server_nodes.append(server_node)
        nodes.append(server_node)

        server_end_time = server_end_time + server_time

        client_time = max(1, int(random.gauss(means[client], vars[client])))
        client_node = Node(client_time, len(client_nodes) + len(server_nodes), server_end_time, "C", client)
        client_nodes.append(client_node)
        nodes.append(client_node)

        server_node.add_succ(client_node)
        client_node.add_prec(server_node)

        pq.insert(client_node)

        if (len(server_nodes) > 1):
            server_node.add_prec(server_nodes[-2])
            server_nodes[-2].add_succ(server_node)

    while not pq.isEmpty():
        current_client_node = pq.delete()
        
        server_time = max(1, int(random.gauss(1, 1)))

        server_node = Node(server_time, len(client_nodes) + len(server_nodes), current_client_node.end_time, "S", -1)
        server_nodes.append(server_node)
        nodes.append(server_node)

        current_client_node.add_succ(server_node)
        server_node.add_prec(current_client_node)

        server_node.add_prec(server_nodes[-2])
        server_nodes[-2].add_succ(server_node)

        if len(nodes) < number_of_tasks:
            client_time = max(1, int(random.gauss(means[current_client_node.client], vars[current_client_node.client])))
            client_node = Node(client_time, len(client_nodes) + len(server_nodes), server_node.end_time, "C", current_client_node.client)
            client_nodes.append(client_node)
            nodes.append(client_node)

            server_node.add_succ(client_node)
            client_node.add_prec(server_node)

            pq.insert(client_node)

    return nodes


num_clients = 3
number_of_tasks = 1000

means = np.random.randint(5, 15, size=(num_clients))
vars = np.random.randint(2, 4, size=(num_clients))

nodes = generate_dag(num_clients, means, vars, number_of_tasks)

# Write to file
with open(f"{number_of_tasks}.pdf", "w") as file:
    file.write(f"Nodes: {len(nodes)}\n")
    for node in nodes:
        file.write(f"{node.time}, ")
    
    file.write("\n\nPrecedence constraints:\n")
    for node in nodes:
        for succ in node.succ:
            file.write(f"({node.id}, {succ.id}), ")

# print(f"means: {means}")
# print(f"vars: {vars}")
# print()

# nodes = generate_dag(num_clients, means, vars, number_of_tasks)

# print("Nodes:")
# for node in nodes:
#     print(f"{node.time}", end=", ")

# print()
# print("Precedence constraints: ")
# for node in nodes:
#     for succ in node.succ:
#         print(f"({node.id}, {succ.id})", end=", ")