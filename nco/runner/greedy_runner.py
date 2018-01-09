import numpy as np
import time

from tensorforce.execution import Runner


class GreedyRunner(Runner):
    def episode_finished(self):
        tour_str = "[" + \
                   ", ".join([str(city) for city in self.environment.tour[:3]]) \
                   + ", ..., " \
                   + ", ".join([str(city) for city in self.environment.tour[-3:]]) \
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
              self.environment.nr_cities,
              tour_str.center(37),
              "{:.4f}".format(self.episode_rewards[-1]).replace(".", ","),
              "{:.4f}".format(self.environment.approx_ratio).replace(".", ","),
              "{:.4f}".format(np.mean(self.environment.episodes_approx_ratios)).replace(".", ","),
              "{:.4e}".format(last_batch_loss).replace(".", ","),
              "{:.4e}".format(mean_batch_loss).replace(".", ","),
              sep="\t")
        return True

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

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()
        self.batch_losses = []

        while True:

            state = self.environment.reset()
            self.agent.reset()
            episode_reward = 0
            self.episode_timestep = 0
            episode_start_time = time.time()

            while True:

                action = self.agent.act(states=state, deterministic=deterministic)

                if self.repeat_actions > 1:
                    reward = 0
                    for repeat in range(self.repeat_actions):
                        state, action, terminal, step_reward = self.environment.execute(actions=action)
                        reward += step_reward
                        if terminal:
                            break
                else:
                    state, action, terminal, reward = self.environment.execute(actions=action)

                self.agent.current_actions["action"] = action
                loss_per_instance = self.agent.observe(terminal=terminal, reward=reward, return_loss_per_instance=True)
                if loss_per_instance is not None:
                    self.batch_losses.append(np.mean(loss_per_instance))

                self.episode_timestep += 1
                self.timestep += 1
                episode_reward += reward

                if terminal or (max_episode_timesteps is not None and self.episode_timestep == max_episode_timesteps):
                    break

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

        self.agent.close()
        self.environment.close()

    def validate(self,
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

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()
        self.batch_losses = []

        while True:

            state = self.environment.reset()
            self.agent.reset()
            episode_reward = 0
            self.episode_timestep = 0
            episode_start_time = time.time()

            while True:

                action = self.agent.act(states=state, deterministic=deterministic)

                if self.repeat_actions > 1:
                    reward = 0
                    for repeat in range(self.repeat_actions):
                        state, action, terminal, step_reward = self.environment.execute(actions=action)
                        reward += step_reward
                        if terminal:
                            break
                else:
                    state, action, terminal, reward = self.environment.execute(actions=action)

                self.episode_timestep += 1
                self.timestep += 1
                episode_reward += reward

                if terminal or (max_episode_timesteps is not None and self.episode_timestep == max_episode_timesteps):
                    break

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

        self.agent.close()
        self.environment.close()