import tempfile

from graphviz import Digraph
from itertools import count
# noinspection PyCompatibility
from queue import PriorityQueue

import math
import numpy as np
import tensorflow as tf
import time

from copy import deepcopy, copy
from tensorforce.execution import Runner


class Node:
    def __init__(self, environment, state=None, action=None, terminal=False, q_value=0.0, parent_node=None):
        self.environment = environment
        self.state = state
        self.action = action
        self.terminal = terminal
        self.current_value = -self.environment.current_value
        self.q_value = q_value

        self.layer = len(environment.current_solution)
        self.parent_node = parent_node
        self.child_nodes = {}

        self.label = "[" + \
                      ", ".join([str(var) for var in self.environment.current_solution]) \
                      + "]"

class TreeRunner(Runner):
    def episode_finished(self):
        if len(self.environment.current_solution) < 10:
            solution_str = "[" + \
                       ", ".join([str(var) for var in self.environment.current_solution]) \
                       + "]"
        else:
            solution_str = "[" + \
                       ", ".join([str(var) for var in self.environment.current_solution[:3]]) \
                       + ", ..., " \
                       + ", ".join([str(var) for var in self.environment.current_solution[-3:]]) \
                       + "]"

        if self.batch_losses:
            last_batch_loss = self.batch_losses[-1]
            mean_batch_loss = np.mean(self.batch_losses[-100:])
        else:
            last_batch_loss = 0.0
            mean_batch_loss = 0.0

        if self.q_values:
            last_q_values = self.q_values[-1]
            mean_q_values = np.mean(self.q_values[-100:])
        else:
            last_q_values = 0.0
            mean_q_values = 0.0

        nodes_by_layer = []

        for layer in range(0, self.environment.nr_vertices):
            nodes_by_layer.append((self.environment.nr_vertices - layer) * nodes_by_layer[layer])
        total_node_count = sum(nodes_by_layer[1:])

        print(self.episode,
              self.episode_timesteps[-1],
              self.timestep // self.agent.batch_size,
              self.environment.nr_vertices,
              solution_str.center(31),
              "{:.4f}".format(self.episode_rewards[-1]).replace(".", ","),
              "|",
              self.nodes_created,
              self.nodes_visited,
              "{:.4f}".format(self.nodes_created / total_node_count).replace(".", ","),
              "{:.4f}".format(self.nodes_visited / total_node_count).replace(".", ","),
              "{:.4f}".format(self.nodes_visited / self.nodes_created).replace(".", ","),
              "|",
              self.leafs_created,
              self.leafs_visited,
              "{:.4f}".format(self.leafs_created / math.factorial(self.environment.nr_vertices)).replace(".", ","),
              "{:.4f}".format(self.leafs_visited / math.factorial(self.environment.nr_vertices)).replace(".", ","),
              "{:.4f}".format(self.leafs_visited / self.leafs_created).replace(".", ","),
              "|",
              "{:.4f}".format(self.environment.approx_ratio).replace(".", ","),
              "{:.4f}".format(np.mean(self.environment.episodes_approx_ratios)).replace(".", ","),
              "|",
              "{:.4f}".format(last_q_values).replace(".", ","),
              "{:.4f}".format(mean_q_values).replace(".", ","),
              "|",
              "{:.4e}".format(last_batch_loss).replace(".", ","),
              "{:.4e}".format(mean_batch_loss).replace(".", ","),
              sep="\t")
        return True

    def print_tree_info(self, node, highest_lower_bound, status):
        # solution_str = "[" + \
        #            ", ".join([str(var) for var in node.environment.current_solution]) \
        #            + "]"
        #
        # if self.batch_losses:
        #     last_batch_loss = self.batch_losses[-1]
        #     mean_batch_loss = np.mean(self.batch_losses)
        # else:
        #     last_batch_loss = 0.0
        #     mean_batch_loss = 0.0
        #
        # if self.q_values:
        #     mean_q_values = np.mean(self.q_values[-100:])
        # else:
        #     mean_q_values = 0.0
        #
        # print(self.episode,
        #       self.timestep // self.agent.batch_size,
        #       node.environment.nr_vars,
        #       self.leafs_created,
        #       "{:.4f}".format(node.current_value).replace(".", ","),
        #       "{:.4f}".format(self.upper_bound).replace(".", ","),
        #       "{:.4f}".format(highest_lower_bound).replace(".", ","),
        #       "{:.4f}".format(node.q_value).replace(".", ","),
        #       "{:.4f}".format(mean_q_values).replace(".", ","),
        #       "{:.4e}".format(last_batch_loss).replace(".", ","),
        #       "{:.4e}".format(mean_batch_loss).replace(".", ","),
        #       status,
        #       self.unopened_nodes.qsize(),
        #       solution_str,
        #       sep="\t")
        pass

    def explore_bounds(self, node):
        if self.episode > 0:
            return [node.current_value + node.q_value,
                    node.current_value <= self.upper_bound or node.current_value + node.q_value <= self.upper_bound]
        else:
            ratio = self.episode / 200
            exploration_current_value = ratio + node.current_value + (1 - ratio) * -node.current_value
            exploration_q_value = ratio + node.q_value + (1 - ratio) * -node.q_value
            return [exploration_current_value + exploration_q_value,
                    (exploration_current_value <= self.upper_bound or exploration_current_value + exploration_q_value <= self.upper_bound)]

    def run(self,
            timesteps=None,
            episodes=None,
            max_episode_timesteps=None,
            deterministic=False,
            episode_finished=None):
        """
        Runs the agent on the environment.

        Args:
            timesteps: Number of timesteps
            episodes: Number of episodes
            max_episode_timesteps: Max number of timesteps per episode
            deterministic: Deterministic flag
            episode_finished: Function handler taking a `Runner` argument and returning a boolean indicating
                whether to continue execution. For instance, useful for reporting intermediate performance or
                integrating termination conditions.
        """

        self.episode = self.agent.episode
        if episodes is not None:
            episodes += self.agent.episode

        self.timestep = self.agent.timestep
        if timesteps is not None:
            timesteps += self.agent.timestep

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()
        self.max_episode_timesteps = max_episode_timesteps
        self.batch_losses = []
        self.q_values = []

        while True:
            search_graph = Digraph()

            state = self.environment.reset()
            self.agent.reset()

            self.counter = count()

            self.nodes_created = 0
            self.nodes_visited = 0
            self.leafs_created = 0
            self.leafs_visited = 0

            root = Node(self.environment, state)
            search_graph.node(root.label, "root")
            node = None
            self.unopened_nodes = PriorityQueue()
            self.unopened_nodes.put((0,
                                0,
                                -next(self.counter),
                                root))

            self.episode_solution = None
            self.upper_bound = -np.inf

            self.episode_timestep = 0
            episode_start_time = time.time()

            while True:
                if node is None:
                    if not self.unopened_nodes.empty():
                        node = self.unopened_nodes.get()[-1]
                        phase = "best_first"
                    else:
                        break
                else:
                    phase = "depth_first"

                highest_lower_bound, node_bound = self.explore_bounds(node)

                if node.terminal:
                    self.nodes_visited += 1
                    self.leafs_visited += 1

                    if node.current_value > self.upper_bound:
                        self.upper_bound = node.current_value
                        self.episode_solution = deepcopy(node)
                        self.print_tree_info(node, highest_lower_bound, phase + ": new bound")
                    else:
                        self.print_tree_info(node, highest_lower_bound, phase + ": leaf")

                    current_solution = copy(node.environment.current_solution)
                    node = root
                    for vertex in current_solution:
                        node = node.child_nodes[vertex]
                        self.agent.current_states = node.state
                        self.agent.current_actions["action"] = node.action

                        updates = self.agent.observe(terminal=node.terminal,
                                                     reward=node.current_value,
                                                     return_loss_per_instance=True)
                        if updates is not None:
                            self.batch_losses.append(np.mean(updates))

                    node = None
                    continue

                if node_bound or (self.max_episode_timesteps is not None and self.episode_timestep == self.max_episode_timesteps):
                    self.print_tree_info(node, highest_lower_bound, phase + ": bounding")
                    # search_graph.node_attr(node.label, color="red")
                    node = None
                    continue
                elif node.layer > 0:
                    self.nodes_visited += 1

                self.agent.act(states=node.state, deterministic=deterministic)
                q_values = copy(self.agent.next_internals[0])
                # q_values /= node.environment.nr_cities
                self.q_values.append(np.mean(q_values))

                for vertex in range(self.environment.nr_vertices):
                    if vertex not in node.environment.current_solution:
                        child_environment = deepcopy(node.environment)
                        state, action, terminal, reward = child_environment.execute(actions=vertex)

                        if action not in node.child_nodes.values():
                            child_node = Node(environment=child_environment, state=state, action=action,
                                              terminal=terminal, parent_node=node, q_value=q_values[action])
                            node.child_nodes[action] = child_node

                            if terminal:
                                self.leafs_created += 1
                            self.nodes_created += 1

                            search_graph.node(child_node.label,
                                              "{}\nub: {:.4f}\nlb: {:.4f}\ncw: {:.4f}\nqv: {:.4f}".format(child_node.label,
                                                                                                          self.upper_bound,
                                                                                                          child_node.current_value + child_node.q_value,
                                                                                                          child_node.current_value,
                                                                                                          child_node.q_value))
                            search_graph.edge(node.label, child_node.label, "{}.".format(self.nodes_created))

                            self.episode_timestep += 1
                            self.timestep += 1
                        else:
                            print()

                self.print_tree_info(node, highest_lower_bound, phase + ": branching")

                sorted_child_nodes = sorted(node.child_nodes.values(), key=lambda x: x.q_value)

                for child_node in sorted_child_nodes[1:]:
                    self.unopened_nodes.put((child_node.q_value,
                                             child_node.current_value,
                                             -next(self.counter),
                                             child_node))

                node = sorted_child_nodes[0]

            if self.episode_solution is not None:
                self.environment = self.episode_solution.environment
                episode_reward = -self.environment.current_value

                time_passed = time.time() - episode_start_time

                self.episode_rewards.append(episode_reward)
                self.episode_timesteps.append(self.episode_timestep)
                self.episode_times.append(time_passed)

                if (timesteps is not None and self.agent.timestep >= timesteps) or \
                        (episodes is not None and self.agent.episode >= episodes):
                    # agent.episode / agent.timestep are globally updated
                    break

                if episode_finished and not self.episode_finished():
                    break

                if self.episode > 500 and self.episode % 500 == 0:
                    # node = self.episode_solution
                    # while node is not None:
                    #     search_graph.attr(node.label, color="green")
                    #     node = node.parent_node

                    search_graph.view(tempfile.mktemp('.gv'))
                self.episode += 1
            else:
                break

        self.agent.close()
        self.environment.close()

    def validate(self,
                 timesteps=None,
                 episodes=None,
                 max_episode_timesteps=None,
                 deterministic=False,
                 episode_finished=None):
        pass
