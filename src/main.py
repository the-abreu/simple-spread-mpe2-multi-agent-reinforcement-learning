from environment.simple_spread import create_environment
from agents.independent_q_learning_agent import IndependentQLearningAgent
from utils.discretization import discretize


def main():

    env = create_environment()

    observations, infos = env.reset(seed=42)

    agents = {}

    for agent_name in env.possible_agents:
        agents[agent_name] = IndependentQLearningAgent(
            action_space=env.action_space(agent_name)
        )

    print("=" * 50)
    print("Simple Spread Environment")
    print("=" * 50)

    print(f"\nAgents: {env.possible_agents}")
    print(f"\nNumber of agents: {len(env.possible_agents)}")

    # ==========================================================
    # Experiment 1 - Observations
    # ==========================================================

    print("\n" + "=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)

    for agent in env.possible_agents:

        observation = observations[agent]

        print(f"\nAgent: {agent}")
        print(f"Shape: {observation.shape}")
        print(f"Observation:\n{observation}")

    # ==========================================================
    # Experiment 2 - Action Spaces
    # ==========================================================

    print("\n" + "=" * 60)
    print("ACTION SPACES")
    print("=" * 60)

    for agent in env.possible_agents:

        action_space = env.action_space(agent)

        print(f"\nAgent: {agent}")
        print(f"Action space: {action_space}")

    # ==========================================================
    # Experiment 0 - Discretization Test
    # ==========================================================

    print("\n" + "=" * 60)
    print("DISCRETIZATION TEST")
    print("=" * 60)

    observation = observations["agent_0"]

    state = discretize(observation)

    print(f"\nOriginal observation shape: {observation.shape}")
    print(f"Discrete state: {state}")
    print(f"State size: {len(state)}")

    # ==========================================================
    # Experiment 3 - Multi-Agent Loop
    # ==========================================================

    print("\n" + "=" * 60)
    print("MULTI-AGENT LOOP")
    print("=" * 60)

    step = 0

    while True:

        actions = {}

        # ------------------------------------------------------
        # Escolha das ações
        # ------------------------------------------------------

        for agent_name in env.agents:

            observation = observations[agent_name]

            action = agents[agent_name].choose_action(
                observation
            )

            actions[agent_name] = action

        # ------------------------------------------------------
        # Executa ações no ambiente
        # ------------------------------------------------------

        (
            next_observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

        # ------------------------------------------------------
        # Atualiza cada agente
        # ------------------------------------------------------

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

        observations = next_observations

        # ------------------------------------------------------
        # Debug temporário da Q-Table
        # ------------------------------------------------------

        print("\n" + "=" * 60)
        print("Q-TABLE SIZE")
        print("=" * 60)

        agent = agents["agent_0"]

        print(f"Number of states: {len(agent.q_table)}")

        if len(agent.q_table) > 0:

            first_state = next(iter(agent.q_table))

            print(f"Example state: {first_state}")

            print(
                f"Q-values: {agent.q_table[first_state]}"
            )

            print(f"Visited states: {len(agent.q_table)}")

            for state, values in list(agent.q_table.items())[:5]:
                print(state, values)

        step += 1

        print(f"\nStep: {step}")
        print(f"Rewards: {dict(rewards)}")

        if all(terminations.values()) or all(truncations.values()):
            print("\nEpisode finished!")
            break

    env.close()


if __name__ == "__main__":
    main()