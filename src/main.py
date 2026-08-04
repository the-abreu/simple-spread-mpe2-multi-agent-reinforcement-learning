from environment.simple_spread import create_environment


def main():

    env = create_environment()

    observations, infos = env.reset(seed=42)

    print("=" * 50)
    print("Simple Spread Environment")
    print("=" * 50)

    print(f"\nAgents: {env.agents}")

    print(f"\nNumber of agents: {len(env.agents)}")

    print("\nObservations keys:")

    for agent in observations.keys():
        print(agent)


if __name__ == "__main__":
    main()