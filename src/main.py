from environment.simple_spread import create_environment


def main():
    env = create_environment()

    observations, infos = env.reset(seed=42)

    print("=" * 50)
    print("Simple Spread Environment")
    print("=" * 50)

    print(f"\nAgents: {env.agents}")
    print(f"\nNumber of agents: {len(env.agents)}")

    # ==========================================================
    # Experiment 1 - Observations
    # ==========================================================

    print("\n" + "=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)

    for agent in env.agents:
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
    # Experiment 3 - Actions
    # ==========================================================

    print("\n" + "=" * 60)
    print("FIRST STEP")
    print("=" * 60)

    actions = {
        "agent_0": 0,
        "agent_1": 0,
        "agent_2": 0,
    }

    print("\nActions:")
    print(actions)

    (
        next_observations,
        rewards,
        terminations,
        truncations,
        infos,
    ) = env.step(actions)

    print("\nRewards:")
    print(rewards)

    print("\nTerminations:")
    print(terminations)

    print("\nTruncations:")
    print(truncations)

    print("\nNext observation keys:")
    print(next_observations.keys())

    # ==========================================================
    # Observation Keys
    # ==========================================================

    print("\n" + "=" * 60)
    print("OBSERVATION KEYS")
    print("=" * 60 + "\n")

    for agent in observations.keys():
        print(agent)


if __name__ == "__main__":
    main()