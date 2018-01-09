from queue import PriorityQueue

import datetime
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import math
import itertools
import pickle
import random

from copy import deepcopy, copy

from tensorforce.core.networks import Network

from nco.runner.tree_runner import Node

# S2V parameters
EMBEDDING_ITERATIONS = 5
OUTPUT_SIZE = 5

# TSP parameters
INSTANCE_SIZE = "debug"
INSTANCE_SIZE_BOUNDS = dict(
    debug=[5, 5],
    train_xxs=[10, 15],
    train_xs=[15, 20],
    train_s=[40, 50],
    train_m=[50, 100],
    train_l=[100, 200],
    train_xl=[200, 300],
    test_s=[300, 400],
    test_m=[400, 500],
    test_l=[500, 600],
    test_xl=[1000, 1200]
)
SQUARE_SIZE = int(1e6)
ILLEGAL_VALUE = 1e4


class CGNTSPNetwork(Network):
    def graph_embedding_layer(self, x, distances, it):
        with tf.variable_scope("graph_embedding_layer_" + str(it), reuse=False):
            weights = tf.get_variable(name="embedding_layer_weights",
                                      shape=(1, OUTPUT_SIZE, OUTPUT_SIZE),
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

            embedding = tf.matmul(distances, x)
            embedding = tf.reshape(embedding, [1, OUTPUT_SIZE, -1])
            embedding = tf.matmul(weights, embedding)
            embedding = tf.reshape(embedding, [-1, OUTPUT_SIZE, 1])

            mean, variance = tf.nn.moments(x=embedding,
                                           axes=[0, 1],
                                           keep_dims=False)
            embedding = tf.nn.batch_normalization(x=embedding,
                                                  mean=mean,
                                                  variance=variance,
                                                  offset=None,
                                                  scale=None,
                                                  variance_epsilon=1e-8)

            embedding = tf.nn.relu(embedding)
            embedding = tf.add(embedding, x)

            return embedding


    def tf_apply(self, x, internals=(), update=False, return_internals=False):
        vertex_states = x["vertex_states"]
        distances = x["distances"]

        embedding = self.graph_embedding_layer(x=vertex_states,
                                               distances=distances,
                                               it=0)
        for it in range(1, EMBEDDING_ITERATIONS):
            embedding = self.graph_embedding_layer(x=embedding,
                                                   distances=distances,
                                                   it=it)
        embedding = tf.reshape(embedding, [-1, OUTPUT_SIZE])
        if return_internals:
            return embedding, []
        else:
            return embedding


class CGNTSPEnvironment():
    def __init__(self, instance_type="uniform"):
        self.network = CGNTSPNetwork
        self.instance_type = instance_type

        self.lb_instance_size = INSTANCE_SIZE_BOUNDS[INSTANCE_SIZE][0]
        self.ub_instance_size = INSTANCE_SIZE_BOUNDS[INSTANCE_SIZE][1]

        assert self.ub_instance_size <= OUTPUT_SIZE

        self.nr_vertices = 0
        self.vertices = []
        self.nr_edges = 0
        self.edges = []

        self.vertex_states = []
        self.edge_states = []
        self.distances = []
        self.neighbourhoods = []

        self.current_solution = []
        self.current_value = np.inf

        self.best_solution = []
        self.best_value = np.inf

        self.optimal_solution = []
        self.optimal_value = 0

        self.approx_ratio = 0
        self.episodes_approx_ratios = []

        self.states = dict(vertex_states=dict(type="float", shape=(OUTPUT_SIZE, 1)),
                           distances=dict(type="float", shape=(OUTPUT_SIZE, OUTPUT_SIZE)))
        self.actions = dict(type="int", num_actions=OUTPUT_SIZE)

    def get_state(self):
        return dict(vertex_states=self.vertex_states,
                    distances=self.distances + np.identity(OUTPUT_SIZE))

    def draw_random_instance(self, compare_solvers=False):
        self.nr_vertices = random.randint(self.lb_instance_size, self.ub_instance_size)

        if self.instance_type == "clustered":
            cluster_centers = [random.sample(range(SQUARE_SIZE), 2)
                               for cluster in range(max(1, int(self.ub_instance_size / 100)))]

            self.vertices = []
            for vertex in range(OUTPUT_SIZE):
                if vertex < self.nr_vertices:
                    random_center = random.choice(cluster_centers)
                    offsets = [int(random.normalvariate(mu=0, sigma=1) * SQUARE_SIZE / math.sqrt(self.nr_vertices))
                               for dim in range(2)]
                    self.vertices.append([random_center[0] + offsets[0],
                                          random_center[1] + offsets[1]])
                else:
                    self.vertices.append([0, 0])

        elif self.instance_type == "uniform":
            self.vertices = [random.sample(range(SQUARE_SIZE), 2)
                             if var < self.nr_vertices else
                             [0, 0]
                             for var in range(OUTPUT_SIZE)]

        self.distances = np.rint([[(np.linalg.norm(np.asarray(x) - np.asarray(y)))
                                   for y in self.vertices]
                                  for x in self.vertices],
                                 dtype=np.float)
        self.distances[self.nr_vertices:, :] = 0
        self.distances[:, self.nr_vertices:] = 0

        self.distances /= np.max(self.distances)

        # self.distances - np.mean(self.distances)
        # self.distances /= np.std(self.distances)

        self.neighbourhoods = np.zeros_like(self.distances, dtype=np.float)

        self.neighbourhoods[self.distances > 0] = 1

        self.vertex_states = np.zeros(self.states["vertex_states"]["shape"], dtype=np.float)

        self.current_solution = []
        self.current_value = 0

        self.approx_ratio = 0

        # if compare_solvers:
        #     solver_data = dict()
        #
        #     start = datetime.datetime.now()
        #     self.optimal_value, self.optimal_solution = self.solve_tsp_dynamic()
        #     solver_data["dynamic"] = [datetime.datetime.now() - start,
        #                               self.optimal_value]
        #
        #     start = datetime.datetime.now()
        #     self.k_opt_value, self.k_opt_solution = self.solve_tsp_3opt()
        #     solver_data["k_opt"] = [datetime.datetime.now() - start,
        #                             self.k_opt_value]
        #
        #     start = datetime.datetime.now()
        #     self.tree_search_value, self.tree_search_solution = self.solve_tsp_tree_search()
        #     solver_data["tree_search"] = [datetime.datetime.now() - start,
        #                                   self.tree_search_value]
        #
        #     return solver_data
        # else:
        #     self.optimal_value, self.optimal_solution = self.solve_tsp_dynamic()
        #
        #     # plt.plot(*zip(*[self.vars[self.optimal_solution[vertex % self.nr_vars]]
        #     #                 for vertex in range(self.nr_vars + 1)]),
        #     #          linestyle="solid",
        #     #          marker="o")
        #     # plt.show()

    def reset(self):
        self.current_solution = []
        self.current_value = np.inf

        self.best_solution = []
        self.best_value = np.inf

        self.optimal_solution = []
        self.optimal_value = 0

        self.draw_random_instance()
        return self.get_state()

    def execute(self, actions):
        # assert action in range(len(self.cities))
        # assert action not in self.tour

        if not all([actions in range(self.nr_vertices), actions not in self.current_solution]):
            if not self.current_solution:
                actions = np.argmin(np.sum(self.distances[:self.nr_vertices - 1, :self.nr_vertices - 1], axis=1))
            else:
                actions = random.choice([vertex
                                         for vertex in range(self.nr_vertices)
                                         if vertex not in self.current_solution])

        if not self.current_solution:
            self.current_solution.append(actions)
            reward = 0

        else:
            self.current_solution.append(actions)
            self.vertex_states[actions] = 1

            old_value = self.current_value
            self.current_value = self.evaluate_solution(self.current_solution)
            if self.current_value == ILLEGAL_VALUE:
                print()
            reward = -(self.current_value - old_value)

        terminal = len(self.current_solution) == self.nr_vertices

        if terminal:
            if self.optimal_value > 0:
                self.approx_ratio = self.current_value / self.optimal_value
                self.episodes_approx_ratios.append(self.approx_ratio)

        return self.get_state(), actions, terminal, reward

    def evaluate_solution(self, tour):
        tour_length = 0

        for city in range(len(tour)):
            distance = self.distances[tour[city % len(tour)]][tour[(city + 1) % len(tour)]]
            if distance == 0.0:
                return ILLEGAL_VALUE
            else:
                tour_length += distance

        return tour_length

    # https://gist.github.com/mlalevic/6222750
    def solve_tsp_dynamic(self):
        # calc all lengths
        # initial value - just distance from 0 to every other point + keep the track of edges
        a = {(frozenset([0, idx + 1]), idx + 1): (distance, [0, idx + 1]) for idx, distance in enumerate(self.distances[0][1:])}
        for m in range(2, self.nr_vertices):
            b = {}
            for s in [frozenset(C) | {0} for C in itertools.combinations(range(1, self.nr_vertices), m)]:
                for j in s - {0}:
                    # this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                    b[(s, j)] = min([(a[(s - {j}, k)][0] + self.distances[k][j],
                                      a[(s - {j}, k)][1] + [j])
                                     for k in s if k != 0 and k != j])
            a = b
        result = min([(a[d][0] + self.distances[0][d[1]],
                       a[d][1]) for d in iter(a)])
        # 0 for padding
        return result

    def solve_tsp_nearest_neighbour(self, current_solution=None):
        correction_factor = 1
        heuristic_value = 0
        if current_solution is None:
            heuristic_solution = [int(np.argmin(np.sum(self.distances[:self.nr_vertices - 1, :self.nr_vertices - 1], axis=1)))]
            current_solution = self.current_solution
        else:
            heuristic_solution = copy(current_solution)

        for step in range(1, self.nr_vertices):
            if step < len(current_solution):
                distance = self.distances[current_solution[step - 1]][current_solution[step]]
            else:
                nearest_neighbour = sorted([(neighbour, self.distances[heuristic_solution[-1]][neighbour])
                                            for neighbour in range(self.nr_vertices)
                                            if neighbour not in heuristic_solution],
                                           key=lambda x: x[1])[0]
                distance = nearest_neighbour[1]
                heuristic_solution.append(nearest_neighbour[0])
            if distance == 0.0:
                return ILLEGAL_VALUE
            else:
                heuristic_value += distance

        return [correction_factor * heuristic_value, heuristic_solution]

    def solve_tsp_tree_search(self):
        nodes_by_layer = [1]
        for layer in range(0, self.nr_vertices):
            nodes_by_layer.append((self.nr_vertices - layer) * nodes_by_layer[layer])
        total_node_count = sum(nodes_by_layer[1:])

        nodes_created = 0
        nodes_visited = 0
        leafs_created = 0
        leafs_visited = 0

        def print_tree_info(status):
            # solution_str = "[" + \
            #            ", ".join([str(vertex)
            #                       for vertex in node.environment.current_solution]) \
            #            + "]"
            #
            # print(node.environment.nr_vertices,
            #       "{:.4f}".format(node.current_value).replace(".", ","),
            #       "{:.4f}".format(upper_bound).replace(".", ","),
            #       "{:.4f}".format(highest_lower_bound).replace(".", ","),
            #       "{:.4f}".format(node.q_value).replace(".", ","),
            #       "|",
            #       nodes_created,
            #       nodes_visited,
            #       "{:.4f}".format(nodes_created / total_node_count).replace(".", ","),
            #       "{:.4f}".format(nodes_visited / total_node_count).replace(".", ","),
            #       "|",
            #       leafs_created,
            #       leafs_visited,
            #       "{:.4f}".format(leafs_created / math.factorial(self.nr_vertices)).replace(".", ","),
            #       "{:.4f}".format(leafs_visited / math.factorial(self.nr_vertices)).replace(".", ","),
            #       "|",
            #       unopened_nodes.qsize(),
            #       status,
            #       solution_str,
            #       sep="\t")
            pass

        env = deepcopy(self)
        state = env.reset

        counter = itertools.count()

        root = Node(env, state)
        node = None
        unopened_nodes = PriorityQueue()
        unopened_nodes.put((0,
                            0,
                            -next(counter),
                            root))

        best_solution = None
        upper_bound = -np.inf

        while True:
            if node is None:
                if not unopened_nodes.empty():
                    node = unopened_nodes.get()[-1]
                else:
                    break

            highest_lower_bound = node.current_value + node.q_value

            if node.terminal:
                nodes_visited += 1
                leafs_visited += 1
                if node.current_value > upper_bound:
                    upper_bound = node.current_value
                    best_solution = deepcopy(node)
                    print_tree_info("new_bound")
                else:
                    print_tree_info("leaf")
                node = None
                continue

            if node.current_value <= upper_bound or highest_lower_bound <= upper_bound:
                print_tree_info("bounding")
                node = None
                continue
            elif node.layer > 0:
                nodes_visited += 1


            for var in range(env.nr_vertices):
                if var not in node.environment.current_solution:
                    child_environment = deepcopy(node.environment)
                    state, action, terminal, reward = child_environment.execute(actions=var)

                    if action not in node.child_nodes.values():
                        child_node = Node(environment=child_environment, state=state, action=action, terminal=terminal,
                                          parent_node=node,
                                          q_value=-self.solve_tsp_nearest_neighbour(child_environment.current_solution)[0])
                        node.child_nodes[action] = child_node

                        if terminal:
                            leafs_created += 1
                        nodes_created += 1

            print_tree_info("branching")
            sorted_child_nodes = sorted(node.child_nodes.values(), key=lambda x: x.q_value)

            for child_node in sorted_child_nodes[1:]:
                unopened_nodes.put((child_node.q_value,
                                    child_node.current_value,
                                    -next(counter),
                                    child_node))

            node = sorted_child_nodes[0]

        return [best_solution.environment.current_value, best_solution.environment.current_solution]

    # def solve_tsp_2opt(self):
    #     tour = random.sample(range(self.nr_cities), self.nr_cities)
    #
    #     for _ in range(self.k_opt_iterations):
    #         [i, j] = sorted(random.sample(range(self.nr_cities), 2))
    #         new_tour = tour[:i] + tour[j:j+1] + tour [i + 1:j] + tour[i:i + 1] + tour [j + 1:]
    #         new_distances = [self.distances[new_tour[k % self.nr_cities]][new_tour[(k + 1) % self.nr_cities]]
    #                          for k in [j, j - 1, i, i - 1]]
    #         if 0.0 in new_distances:
    #             continue
    #         old_distances = [self.distances[tour[k % self.nr_cities]][tour[(k + 1) % self.nr_cities]]
    #                          for k in [j, j - 1, i, i - 1]]
    #         if sum(new_distances) < sum(old_distances):
    #             tour = new_tour.copy()
    #
    #     return [self.evaluate_tour(tour), tour]

    def solve_tsp_3opt(self):
        iterations = int(1e6)

        current_solution = random.sample(range(self.nr_vertices), self.nr_vertices)
        for _ in range(iterations):
            old_i, old_j, old_k = sorted(random.sample(range(self.nr_vertices), 3))

            candidate_solutions = []
            for candidate in itertools.permutations([[old_i, old_j, old_k]]):
                new_i, new_j, new_k = candidate[0]
                candidate_solution = current_solution[:old_i] \
                                     + current_solution[new_i:new_i + 1] \
                                     + current_solution[old_i + 1:old_j] \
                                     + current_solution[new_j:new_j + 1] \
                                     + current_solution[old_j + 1:old_k] \
                                     + current_solution[new_k:new_k + 1] \
                                     + current_solution[old_k + 1:]

                new_distances = [self.distances[candidate_solution[l % self.nr_vertices]][candidate_solution[(l + 1) % self.nr_vertices]]
                                 for l in [new_i - 1, new_i, new_j - 1, new_j, new_k - 1, new_k]]

                if 0.0 in new_distances:
                    candidate_solutions.append([candidate_solutions.copy(), int(1e4)])
                else:
                    candidate_solutions.append([candidate_solutions.copy(), sum(new_distances)])

            current_distance = sum([self.distances[current_solution[l % self.nr_vertices]][current_solution[(l + 1) % self.nr_vertices]]
                                    for l in [old_i - 1, old_i, old_j - 1, old_j, old_k - 1, old_k]])

            for candidate_solution in candidate_solutions:
                if candidate_solution[1] < current_distance:
                    current_solution = candidate_solution[0].copy()
                    current_distance = candidate_solution[1]

        return [self.evaluate_solution(current_solution), current_solution]

    def generate_data(self, nr_instances, compare_solvers=False):
        file_name = "./data/tsp/{}_{}-{}.p".format(self.instance_type,
                                                   self.lb_instance_size,
                                                   self.ub_instance_size)
        print_body = "#:\t{}\tsize:\t{}\t" \
                     "|\t{}:\t{}\t{:.4f}\t" \
                     "|\t{}:\t{}\t{:.4f}\tratio:\t{:.4f}" \
                     "|\t{}:\t{}\t{:.4f}\tratio:\t{:.4f}"
        try:
            instances = pickle.load(open(file_name, "rb"))
        except (pickle.PickleError, FileNotFoundError, EOFError):
            instances = []

        solver_stats = dict(dynamic=list(),
                            k_opt=list(),
                            tree_search=list())

        while len(instances) < nr_instances:
            solver_data = self.draw_random_instance(compare_solvers)
            if solver_data:
                for key, value in solver_data.items():
                    solver_stats[key].append(value)
                print(print_body.format(len(instances) + 1,
                                        self.nr_vertices,
                                        "dynamic",
                                        solver_stats["dynamic"][-1][0],
                                        solver_stats["dynamic"][-1][1],
                                        "k_opt",
                                        solver_stats["k_opt"][-1][0],
                                        solver_stats["k_opt"][-1][1],
                                        solver_stats["k_opt"][-1][1] / solver_stats["dynamic"][-1][1],
                                        "tree_search",
                                        solver_stats["tree_search"][-1][0],
                                        solver_stats["tree_search"][-1][1],
                                        solver_stats["tree_search"][-1][1] / solver_stats["dynamic"][-1][1])
                      )

            instances.append(deepcopy(self))

            # if instances and len(instances) % 25 == 0:
            #     plt.plot(*zip(*[self.vertices[self.optimal_solution[var % self.nr_vertices]]
            #                     for var in range(self.nr_vertices + 1)]),
            #              label="dynamic",
            #              linestyle="solid",
            #              color="green",
            #              marker="o")
            #     plt.plot(*zip(*[self.vertices[self.tree_search_solution[var % self.nr_vertices]]
            #                     for var in range(self.nr_vertices + 1)]),
            #              label="tree_search",
            #              linestyle="dotted",
            #              linewidth="4.0",
            #              color="black",
            #              marker="nothing")
            #     plt.plot(*zip(*[self.vertices[self.k_opt_solution[var % self.nr_vertices]]
            #                     for var in range(self.nr_vertices + 1)]),
            #              label="k_opt",
            #              linestyle="dotted",
            #              linewidth="4.0",
            #              color="blue",
            #              marker="nothing")
            #     plt.show()
            #
            #     pickle.dump(instances, open(file_name, "wb"))

        pickle.dump(instances, open(file_name, "wb"))

    def close(self):
        pass
