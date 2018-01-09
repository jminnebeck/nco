import threading
import time
import math
import numpy as np
from scipy import stats
from copy import copy, deepcopy

from tensorforce.execution import Runner

ALPHA = 0.05
C = 1.5
EPS = 0.75

EXPANSION_STEPS = 500

TENSOR_NAMES = dict(action_logits='vpg/actions-and-internals/categorical/parameterize/Log:0',
                    action_probs='vpg/actions-and-internals/categorical/parameterize/Maximum:0',
                    action_state_value='vpg/actions-and-internals/categorical/parameterize/ReduceLogSumExp/Squeeze:0',
                    baseline_prediction='vpg/loss-per-instance/vpg/reward-estimation/mlp-baseline/predict/Squeeze:0')

STATE_VALUE_TENSOR = 'vpg/loss-per-instance/vpg/reward-estimation/mlp-baseline/predict/Squeeze:0'
LOSS_PER_INSTANC_TENSOR = 'vpg/loss-per-instance/vpg/pg-loss-per-instance/mul:0'

PRINT_STEP_INFO = False

class Node:
    def __init__(self, environment, state=None, action=None, terminal=False, visit_count=0, prior_value=0.0, parent_node=None):
        self.environment = environment
        self.state = state
        self.action = action
        self.terminal = terminal

        self.visit_count = visit_count
        self.action_values = []
        self.prior_value = prior_value

        self.virtual_loss = 0

        self.layer = len(self.environment.current_solution)
        self.parent_node = parent_node
        self.child_nodes = {}

        self.label = "[" + \
                      ", ".join([str(var) for var in self.environment.current_solution]) \
                      + "]"

    @property
    def mean_value(self):
        total_action_value = sum(self.action_values)
        virtual_visits = self.visit_count + self.virtual_loss
        if total_action_value == 0 or virtual_visits == 0:
            return 0.0
        else:
            return total_action_value / virtual_visits

    @property
    def upper_bound(self):
        return C * self.prior_value * (math.sqrt(self.parent_node.visit_count) / (1 + self.visit_count + self.virtual_loss))
        # return C * self.prior_value / (1 + self.visit_count + self.virtual_loss)


class MCTSRunner(Runner):
    def episode_finished(self):
        if len(self.environment.current_solution) < 10:
            solution_str = "[" + \
                       ", ".join([str(vertex) for vertex in self.environment.current_solution]) \
                       + "]"
        else:
            solution_str = "[" + \
                       ", ".join([str(vertex) for vertex in self.environment.current_solution[:3]]) \
                       + ", ..., " \
                       + ", ".join([str(vertex) for vertex in self.environment.current_solution[-3:]]) \
                       + "]"

        if self.batch_losses:
            last_batch_loss = self.batch_losses[-1]
            mean_batch_loss = np.mean(self.batch_losses)
        else:
            last_batch_loss = 0.0
            mean_batch_loss = 0.0

        print(self.episode,
              self.timestep,
              self.timestep // self.agent.batch_size,
              self.environment.nr_vertices,
              self.episode_search_steps,
              "{:.4f}".format(self.environment.current_value).replace(".", ","),
              "{:.4f}".format(self.environment.best_value).replace(".", ",").rjust(6),
              # "{:.4f}".format(self.environment.best_value / self.environment.optimal_value).replace(".", ",").rjust(6),
              "{:.4e}".format(last_batch_loss).replace(".", ","),
              "{:.4e}".format(mean_batch_loss).replace(".", ","),
              solution_str,
              sep="\t")
        pass
        return True

    def print_step_info(self, root, visit_counts_before):
        if len(self.environment.current_solution) < 10:
            solution_str = "[" + \
                       ", ".join([str(vertex) for vertex in self.environment.current_solution]) \
                       + "]"
        else:
            solution_str = "[" + \
                       ", ".join([str(vertex) for vertex in self.environment.current_solution[:3]]) \
                       + ", ..., " \
                       + ", ".join([str(vertex) for vertex in self.environment.current_solution[-3:]]) \
                       + "]"

        # search_prob_str = "[" + \
        #                   " ".join([str(root.child_nodes[vertex].visit_count).rjust(4)
        #                             if vertex in root.child_nodes else "    "
        #                             for vertex in range(self.environment.nr_vertices)]) \
        #                   + "]"

        sum_priors = sum([root.child_nodes[vertex].prior_value
                          for vertex in root.child_nodes.keys()])
        normalised_priors = [root.child_nodes[vertex].prior_value / sum_priors
                             if vertex in root.child_nodes else 0
                             for vertex in range(self.environment.nr_vertices)]
        search_prob_str = "[" + \
                          ", ".join(["{:.2f}/{:.2f}".format(root.child_nodes[vertex].visit_count / root.visit_count, normalised_priors[vertex]).rjust(4)
                                     if vertex in root.child_nodes else "         "
                                     for vertex in range(self.environment.nr_vertices)]) \
                          + "]"

        # search_prob_str = "[" + \
        #                   ", ".join(["{}|{:.2f}|{:.2f}".format(root.child_nodes[vertex].visit_count,
        #                                                 root.child_nodes[vertex].mean_value,
        #                                                 root.child_nodes[vertex].mean_value + root.child_nodes[vertex].upper_bound).rjust(4)
        #                              if vertex in root.child_nodes else "         "
        #                              for vertex in range(self.environment.nr_vertices)]) \
        #                   + "]"

        if self.batch_losses:
            last_batch_loss = self.batch_losses[-1]
            mean_batch_loss = np.mean(self.batch_losses)
        else:
            last_batch_loss = 0.0
            mean_batch_loss = 0.0

        print(self.episode,
              self.timestep,
              self.timestep // self.agent.batch_size,
              self.environment.nr_vertices,
              self.episode_search_steps,
              str(root.visit_count - visit_counts_before["root"]).rjust(4),
              str(self.step_leaf_visits,).rjust(4),
              "{:.4f}".format(len(self.step_leafs_found) /
                              math.factorial(self.environment.nr_vertices -
                                             len(self.environment.current_solution))).replace(".", ","),
              "{:.4f}".format(self.environment.current_value).replace(".", ","),
              "{:.4f}".format(self.environment.best_value).replace(".", ",").rjust(6),
              # "{:.4f}".format(self.environment.best_value / self.environment.optimal_value).replace(".", ",").rjust(6),
              "{:.4f}".format(root.action_values[-1]).replace(".", ","),
              "{:.4f}".format(root.mean_value).replace(".", ","),
              search_prob_str,
              "{:.4e}".format(last_batch_loss).replace(".", ","),
              "{:.4e}".format(mean_batch_loss).replace(".", ","),
              solution_str,
              sep="\t")

    def expansion_done(self, root, visit_counts_before):
        if len(root.child_nodes) == 1:
            return True
        elif root.visit_count - visit_counts_before["root"] >= EXPANSION_STEPS:
            return True
        elif root.visit_count - visit_counts_before["root"] < 30 * max(1, len(root.child_nodes)):
            return False
        else:
            # unique_action_values = [len(set(child_node.action_values))
            #                         for child_node in root.child_nodes.values()]

            approximation_condition = [root.child_nodes[action].visit_count - visit_counts_before[action] > 30
                                       for action in root.child_nodes.keys()]
            if not all(approximation_condition):
                return False
            else:
                means = np.asarray([np.mean(child_node.action_values)
                                    for child_node in root.child_nodes.values()])
                stds = np.asarray([np.std(child_node.action_values)
                                   for child_node in root.child_nodes.values()])
                sqrt_visits = np.asarray([math.sqrt(child_node.visit_count)
                                          for child_node in root.child_nodes.values()])
                t_stats = np.asarray([stats.t.ppf(1 - 0.5 * ALPHA, child_node.visit_count - 1)
                                      for child_node in root.child_nodes.values()])
                conf_intervals = t_stats * (stds / sqrt_visits)
                deviations = conf_intervals / means
                precisions_reached = np.where(deviations <= ALPHA,  [True], [False])
                return all(precisions_reached)

    def node_prediction(self, node):
        self.agent.act(states=node.state)

        fetches = [self.agent.model.graph.get_tensor_by_name(TENSOR_NAMES['action_probs']),
                   self.agent.model.graph.get_tensor_by_name(TENSOR_NAMES['baseline_prediction'])]

        feed_dict = {self.agent.model.state_inputs[name]: np.expand_dims(state, axis=0)
                     for name, state in node.state.items()}

        predictions = self.agent.model.monitored_session.run(fetches=fetches, feed_dict=feed_dict)
        return [prediction[0] for prediction in predictions]

    def run(self,
            timesteps=None,
            episodes=None,
            max_episode_timesteps=None,
            deterministic=False,
            episode_finished=None):

        self.batch_losses = []

        self.episode = self.agent.episode
        if episodes is not None:
            episodes += self.agent.episode

        self.timestep = self.agent.timestep
        if timesteps is not None:
            timesteps += self.agent.timestep

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()
        self.max_episode_timesteps = max_episode_timesteps

        self.episode_timestep = 0
        episode_start_time = time.time()

        while True:
            state = self.environment.reset()
            self.agent.reset()
            root = Node(self.environment, state)

            episode_reward = 0

            self.episode_search_steps = 0
            while len(self.environment.current_solution) < self.environment.nr_vertices:
                self.step_leaf_visits = 0
                self.step_leafs_found = []

                node = root
                visit_counts_before = {action: 0
                                       for action in range(self.environment.nr_vertices)}
                if root.child_nodes:
                    for action in root.child_nodes.keys():
                        visit_counts_before[action] = root.child_nodes[action].visit_count
                visit_counts_before["root"] = root.visit_count

                while not self.expansion_done(root, visit_counts_before):
                    self.episode_search_steps += 1

                    if node is None:
                        node = root

                    if not node.child_nodes:
                        if node.terminal:
                            self.step_leaf_visits += 1
                            if node.label not in self.step_leafs_found:
                                self.step_leafs_found.append(node.label)

                            if self.environment.best_value == np.inf:
                                self.environment.best_solution = deepcopy(node.environment.current_solution)
                                self.environment.best_value = node.environment.current_value

                            if node.environment.current_value <= self.environment.best_value:
                                state_value = 1
                                if node.environment.current_value < self.environment.best_value:
                                    self.environment.best_solution = deepcopy(node.environment.current_solution)
                                    self.environment.best_value = node.environment.current_value
                            else:
                                state_value = self.environment.best_value / node.environment.current_value
                        else:
                            prior_pred, state_pred = self.node_prediction(node)
                            # noise = np.random.dirichlet(alpha=[0.03] * self.environment.nr_vertices)

                            prior_noise = np.random.rand(self.environment.states["vertex_states"]["shape"][0])
                            noisy_priors = EPS * prior_pred + (1 - EPS) * prior_noise

                            value_noise = np.random.rand(1)[0]
                            state_value = EPS * state_pred + (1 - EPS) * value_noise

                            for action in range(self.environment.nr_vertices):
                                if action not in node.environment.current_solution:
                                    child_environment = deepcopy(node.environment)
                                    state, action, terminal, reward = child_environment.execute(actions=action)
                                    if action not in node.child_nodes.values():
                                        child_node = Node(environment=child_environment,
                                                          state=state,
                                                          action=action,
                                                          terminal=terminal,
                                                          parent_node=node,
                                                          prior_value=noisy_priors[action]
                                                          )
                                        node.child_nodes[action] = child_node

                        while node is not None:
                            node.visit_count += 1
                            node.action_values.append(state_value)
                            node.virtual_loss = max(0, node.virtual_loss - 1)
                            node = node.parent_node

                    else:
                        evaluated_child_nodes = sorted([[child_node, child_node.mean_value, child_node.upper_bound]
                                                        for child_node in node.child_nodes.values()],
                                                       key=lambda x: x[1] + x[2], reverse=True)
                        node = evaluated_child_nodes[0][0]
                        node.virtual_loss += 1

                # if len(self.environment.current_solution) <= math.ceil(0.2 * self.environment.nr_vertices):
                #     temperature = 1
                # else:
                #     temperature = 0.1

                temperature = 1

                sum_exp_visit_counts = sum([root.child_nodes[action].visit_count ** (1 / temperature)
                                            for action in root.child_nodes.keys()])
                search_probs = [(root.child_nodes[action].visit_count ** (1 / temperature)) / sum_exp_visit_counts
                                if action not in root.environment.current_solution else 0.0
                                for action in range(self.environment.nr_vertices)]
                action = np.random.choice(a=range(self.environment.nr_vertices),
                                          p=search_probs)

                state, action, terminal, _ = self.environment.execute(action)

                for name, state in state.items():
                    self.agent.current_states[name] = state

                self.agent.current_actions["action"] = action

                self.timestep += 1
                self.episode_timestep += 1

                loss_per_instance = self.agent.observe(terminal=terminal,
                                                       reward=root.action_values[-1],
                                                       return_loss_per_instance=True)
                if loss_per_instance is not None:
                    self.batch_losses.append(np.mean(loss_per_instance))

                if PRINT_STEP_INFO:
                    self.print_step_info(root, visit_counts_before)

                root = root.child_nodes[action]
                root.environment = self.environment
                root.state = state
                root.terminal = terminal

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

            self.episode += 1
            # print()

        self.agent.close()
        self.environment.close()