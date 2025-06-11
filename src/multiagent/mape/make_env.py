"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, num_agents=2, dist_threshold=0.1, arena_size=1, identity_size=0, 
             discrete_action=True, cam_range=1, mask_obs_dist=None):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                           (without the .py extension)
        num_agents     :    total number of agents in environment
        dist_threshold :    reward threshold for task completion
        arena_size     :    arena size
        identity_size  :    identity size
        discrete_action:    whether to use discrete action space
        cam_range     :     camera view range
        mask_obs_dist :     observation range visualization distance
    Returns:
        env            :   MultiAgentEnv object
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario(num_agents=num_agents, 
                                                            dist_threshold=dist_threshold,
                                                            arena_size=arena_size,
                                                            identity_size=identity_size)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, 
                        scenario.info, scenario.done, discrete_action=discrete_action, 
                        cam_range=cam_range, mask_obs_dist=mask_obs_dist)
    return env
