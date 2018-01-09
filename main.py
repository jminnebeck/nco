from nco.environments.cgn_tsp_environment import CGNTSPEnvironment, OUTPUT_SIZE
from nco.agents.mcts_agent import MCTSAgent
from nco.runner.mcts_runner import MCTSRunner

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    print("creating environment")
    environment = CGNTSPEnvironment(instance_type="uniform")

    optimizer = dict(type='adam',
                     learning_rate=1e-6)

    agent = MCTSAgent(states_spec=environment.states,
                      actions_spec=environment.actions,
                      network_spec=environment.network,
                      preprocessing=dict(type="standardize"),
                      optimizer=optimizer,
                      batch_size=64,
                      batched_observe=64,
                      baseline_mode='network',
                      baseline=dict(type='mlp',
                                    sizes=[OUTPUT_SIZE, 1]
                                    ),
                      baseline_optimizer=optimizer,
                      # summary_spec=dict(directory=".\summaries",
                      #                   save_secs=60,
                      #                   summary_labels=['total-loss',
                      #                                   'losses',
                      #                                   'variables',
                      #                                   'activations',
                      #                                   'relu']
                      #                  )
                      )

    print("creating runner")
    runner = MCTSRunner(agent=agent, environment=environment)

    print("starting runner")
    runner.run(episodes=int(1e5), episode_finished=runner.episode_finished)


if __name__ == '__main__':
    main()
