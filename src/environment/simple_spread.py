from mpe2 import simple_spread_v3

def create_environment():
    """
    Creates and returns the Simple Spread environment.
    """

    env = simple_spread_v3.parallel_env(
        render_mode="rgb_array"
    )

    return env