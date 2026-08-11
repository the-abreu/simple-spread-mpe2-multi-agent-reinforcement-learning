from environment.simple_spread import create_environment
from agents.independent_q_learning_agent import IndependentQLearningAgent
from agents.random_agent import RandomAgent


def main():

    # ==========================================================
    # Training configuration
    # ==========================================================

    num_episodes = 100

    # ==========================================================
    # Environment
    # ==========================================================

    env = create_environment()

    # ==========================================================
    # Agents
    # ==========================================================

    agents = {}

    for agent_name in env.possible_agents:

        agents[agent_name] = IndependentQLearningAgent(
            env.action_space(agent_name)
        )

    print("=" * 60)
    print("SIMPLE SPREAD - TRAINING")
    print("=" * 60)

    print(f"\nAgents: {env.possible_agents}")
    print(f"Number of agents: {len(env.possible_agents)}")
    print(f"Episodes: {num_episodes}")

    # ==========================================================
    # Training metrics
    # ==========================================================

    episode_rewards = []
    episode_steps = []
    epsilon_history = []
    q_table_sizes = []

    # ==========================================================
    # Training loop
    # ==========================================================

    for episode in range(1, num_episodes + 1):

        observations, infos = env.reset()

        total_reward = 0.0
        step = 0

        while True:

            actions = {}

            # --------------------------------------------------
            # Select actions
            # --------------------------------------------------

            for agent_name in env.agents:

                observation = observations[agent_name]

                action = agents[agent_name].choose_action(
                    observation
                )

                actions[agent_name] = action

            # --------------------------------------------------
            # Environment transition
            # --------------------------------------------------

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            # --------------------------------------------------
            # Q-Learning update
            # --------------------------------------------------

            for agent_name in env.agents:

                done = (
                    terminations[agent_name]
                    or truncations[agent_name]
                )

                agents[agent_name].update(
                    observation=observations[agent_name],
                    action=actions[agent_name],
                    reward=rewards[agent_name],
                    next_observation=next_observations[agent_name],
                    done=done,
                )

            # --------------------------------------------------
            # Update state
            # --------------------------------------------------

            observations = next_observations

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------

            total_reward += sum(rewards.values())
            step += 1

            # --------------------------------------------------
            # Episode termination
            # --------------------------------------------------

            if all(terminations.values()) or all(truncations.values()):
                break

        # ======================================================
        # Epsilon decay
        # ======================================================

        for agent_name in agents:

            agents[agent_name].decay_epsilon()

        # ======================================================
        # Store metrics
        # ======================================================

        episode_rewards.append(total_reward)
        episode_steps.append(step)

        epsilon_history.append(
            agents["agent_0"].epsilon
        )

        q_table_sizes.append(
            sum(
                len(agent.q_table)
                for agent in agents.values()
            )
        )

        # ======================================================
        # Training progress
        # ======================================================

        if episode == 1 or episode % 10 == 0:

            print(
                f"Episode {episode:3d} | "
                f"Reward: {total_reward:8.3f} | "
                f"Steps: {step:3d} | "
                f"Epsilon: {agents['agent_0'].epsilon:.3f} | "
                f"Q-States: {q_table_sizes[-1]}"
            )

    # ==========================================================
    # Training summary
    # ==========================================================

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(f"\nEpisodes: {num_episodes}")

    print(
        f"Initial reward: {episode_rewards[0]:.3f}"
    )

    print(
        f"Final reward: {episode_rewards[-1]:.3f}"
    )

    print(
        f"Initial epsilon: {epsilon_history[0]:.3f}"
    )

    print(
        f"Final epsilon: {epsilon_history[-1]:.3f}"
    )

    print(
        f"Final Q-table states: {q_table_sizes[-1]}"
    )

    # ==========================================================
    # Evaluation
    # ==========================================================

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    evaluation_episodes = 100
    evaluation_rewards = []

    # ----------------------------------------------------------
    # Disable exploration
    # ----------------------------------------------------------

    for agent in agents.values():
        agent.epsilon = 0.0

    # ----------------------------------------------------------
    # Evaluation loop
    # ----------------------------------------------------------

    for episode in range(evaluation_episodes):

        observations, infos = env.reset(seed=42 + episode)

        episode_reward = 0.0
        step = 0

        while True:

            actions = {}

            # --------------------------------------------------
            # Select greedy actions
            # --------------------------------------------------

            for agent_name in env.agents:

                action = agents[agent_name].choose_action(
                    observations[agent_name]
                )

                actions[agent_name] = action

            # --------------------------------------------------
            # Environment transition
            # --------------------------------------------------

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            # --------------------------------------------------
            # Evaluation reward
            # --------------------------------------------------

            episode_reward += sum(rewards.values())

            observations = next_observations

            step += 1

            # --------------------------------------------------
            # Episode termination
            # --------------------------------------------------

            if all(terminations.values()) or all(truncations.values()):
                break

        evaluation_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:

            print(
                f"Episode {episode + 1:3d} | "
                f"Reward: {episode_reward:8.3f} | "
                f"Steps: {step:3d}"
            )

    # ==========================================================
    # Evaluation summary
    # ==========================================================

    average_reward = (
        sum(evaluation_rewards)
        / len(evaluation_rewards)
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Evaluation episodes: {evaluation_episodes}"
    )

    print(
        f"Average reward: {average_reward:.3f}"
    )

    print(
        f"Best reward: {max(evaluation_rewards):.3f}"
    )

    print(
        f"Worst reward: {min(evaluation_rewards):.3f}"
    )

    print(
        f"Q-table states: {len(agents['agent_0'].q_table)}"
    )

    print(
        f"Final epsilon: {agents['agent_0'].epsilon:.3f}"
    )

    # ==========================================================
    # Random Agent Baseline
    # ==========================================================

    print("\n" + "=" * 60)
    print("RANDOM AGENT BASELINE")
    print("=" * 60)

    random_agents = {}

    for agent_name in env.possible_agents:

        random_agents[agent_name] = RandomAgent(
            env.action_space(agent_name)
        )

    baseline_episodes = 100
    baseline_rewards = []

    for episode in range(baseline_episodes):

        observations, infos = env.reset(seed=42 + episode)

        episode_reward = 0.0
        step = 0

        while True:

            actions = {}

            # --------------------------------------------------
            # Select random actions
            # --------------------------------------------------

            for agent_name in env.agents:

                action = random_agents[agent_name].choose_action(
                    observations[agent_name]
                )

                actions[agent_name] = action

            # --------------------------------------------------
            # Environment transition
            # --------------------------------------------------

            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = env.step(actions)

            # --------------------------------------------------
            # Global reward
            # --------------------------------------------------

            episode_reward += sum(rewards.values())

            observations = next_observations

            step += 1

            # --------------------------------------------------
            # Episode termination
            # --------------------------------------------------

            if all(terminations.values()) or all(truncations.values()):
                break

        baseline_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:

            print(
                f"Episode {episode + 1:3d} | "
                f"Reward: {episode_reward:8.3f} | "
                f"Steps: {step:3d}"
            )

    # ==========================================================
    # Random Agent Summary
    # ==========================================================

    average_random_reward = (
        sum(baseline_rewards) / len(baseline_rewards)
    )

    print("\n" + "=" * 60)
    print("RANDOM AGENT SUMMARY")
    print("=" * 60)

    print(f"Baseline episodes: {baseline_episodes}")
    print(f"Average reward: {average_random_reward:.3f}")
    print(f"Best reward: {max(baseline_rewards):.3f}")
    print(f"Worst reward: {min(baseline_rewards):.3f}")

    env.close()


if __name__ == "__main__":
    main()