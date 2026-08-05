from environment.simple_spread import create_environment

from agents.random_agent import RandomAgent

from utils.discretization import discretize

def main():
    env = create_environment()

    observations, infos = env.reset(seed=42)

    agents = {}

    for agent_name in env.possible_agents:
        agents[agent_name] = RandomAgent(
            env.action_space(agent_name)
        )

    print("=" * 50)
    print("Simple Spread Environment")
    print("=" * 50)

    print(f"\nAgents: {env.possible_agents}")
    print(f"\nNumber of agents: {len(env.agents)}")

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

    for agent in env.agents:
        action_space = env.action_space(agent)

        print(f"\nAgent: {agent}")
        print(f"Action space: {action_space}")

    # ==========================================================
    # Experiment 3 - Action Steps 
    # ==========================================================

    print("\n" + "=" * 60)
    print("MULTI-AGENT LOOP")
    print("=" * 60)

    observations, infos = env.reset(seed=42)

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

    step = 0

    while True:

        actions = {}

        for agent_name in env.agents:
            observation = observations[agent_name]

            action = agents[agent_name].choose_action(
                observation
            )

            actions[agent_name] = action

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

        step += 1

        print(f"\nStep: {step}")
        print(f"Rewards: {dict(rewards)}")

        if all(terminations.values()) or all(truncations.values()):
            print("\nEpisode finished!")
            break

if __name__ == "__main__":
    main()