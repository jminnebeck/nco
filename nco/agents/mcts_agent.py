from tensorforce.agents import VPGAgent
from tensorforce.models import PGLogProbModel


class MCTSAgent(VPGAgent):
    def observe(self, terminal, reward, return_loss_per_instance=False):
        # from Agent
        self.current_terminal = terminal
        if self.reward_preprocessing is None:
            self.current_reward = reward
        else:
            self.current_reward = self.reward_preprocessing.process(reward)

        if self.batched_observe > 0:
            # Batched observe for better performance with Python.
            self.observe_terminal.append(self.current_terminal)
            self.observe_reward.append(self.current_reward)

            if self.current_terminal or len(self.observe_terminal) >= self.batched_observe:
                self.episode = self.model.observe(
                    terminal=self.observe_terminal,
                    reward=self.observe_reward
                )
                self.observe_terminal = list()
                self.observe_reward = list()

        else:
            self.episode = self.model.observe(
                terminal=self.current_terminal,
                reward=self.current_reward
            )

        # from BatchAgent
        for name, batch_state in self.batch_states.items():
            batch_state.append(self.current_states[name])
        for batch_internal, internal in zip(self.batch_internals, self.current_internals):
            batch_internal.append(internal)
        for name, batch_action in self.batch_actions.items():
            batch_action.append(self.current_actions[name])
        self.batch_terminal.append(self.current_terminal)
        self.batch_reward.append(self.current_reward)

        self.batch_count += 1

        if self.batch_count == self.batch_size:
            loss_per_instance = self.model.update(
                                    states=self.batch_states,
                                    internals=self.batch_internals,
                                    actions=self.batch_actions,
                                    terminal=self.batch_terminal,
                                    reward=self.batch_reward,
                                    return_loss_per_instance=return_loss_per_instance
            )
            self.reset_batch()
            if loss_per_instance is not None:
                return loss_per_instance[1]


        # from MemoryAgent
        # self.memory.add_observation(
        #     states=self.current_states,
        #     internals=self.current_internals,
        #     actions=self.current_actions,
        #     terminal=self.current_terminal,
        #     reward=self.current_reward
        # )
        #
        # if self.timestep >= self.first_update and self.timestep % self.update_frequency == 0:
        #     assert self.repeat_update == 1
        #
        #     for _ in range(self.repeat_update):
        #         batch = self.memory.get_batch(batch_size=self.batch_size, next_states=True)
        #         loss_per_instance = self.model.update(
        #             # TEMP: Random sampling fix
        #             states={name: np.stack((batch['states'][name], batch['next_states'][name])) for name in
        #                     batch['states']},
        #             internals=batch['internals'],
        #             actions=batch['actions'],
        #             terminal=batch['terminal'],
        #             reward=batch['reward'],
        #             return_loss_per_instance=True
        #         )
        #         self.memory.update_batch(loss_per_instance=loss_per_instance)
        #
        #         if loss_per_instance is not None:
        #             return loss_per_instance[1]

    def initialize_model(self, states_spec, actions_spec):
        # return MCTSModel(
        return PGLogProbModel(
            states_spec=states_spec,
            actions_spec=actions_spec,
            network_spec=self.network_spec,
            device=self.device,
            scope=self.scope,
            saver_spec=self.saver_spec,
            summary_spec=self.summary_spec,
            distributed_spec=self.distributed_spec,
            optimizer=self.optimizer,
            discount=self.discount,
            normalize_rewards=self.normalize_rewards,
            variable_noise=self.variable_noise,
            distributions_spec=self.distributions_spec,
            entropy_regularization=self.entropy_regularization,
            baseline_mode=self.baseline_mode,
            baseline=self.baseline,
            baseline_optimizer=self.baseline_optimizer,
            gae_lambda=self.gae_lambda
        )
