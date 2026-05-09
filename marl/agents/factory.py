import torch

from marl.models.mpnn import MPNN
from marl.agents.rlagent import Neo
from marl.utils import make_multiagent_env


def setup_master(args, learner_cls, env=None, return_env=False):
    """Build the learner and its shared team policies.

    This keeps the old setup_master behavior, but removes construction details
    from learner.py. learner_cls is passed in to avoid a circular import.
    """
    if env is None:
        env = make_multiagent_env(
            args.env_name,
            num_agents=args.num_agents,
            dist_threshold=args.dist_threshold,
            arena_size=args.arena_size,
            identity_size=args.identity_size,
            mask_obs_dist=args.mask_obs_dist if hasattr(args, "mask_obs_dist") else None,
        )

    team1, team2 = [], []
    policy1 = None
    policy2 = None

    num_adversary = 0
    num_friendly = 0
    for agent in env.world.policy_agents:
        if hasattr(agent, "adversary") and agent.adversary:
            num_adversary += 1
        else:
            num_friendly += 1

    action_space = env.action_space[-1]
    entity_mp = args.entity_mp
    num_entities = _num_entities_for_env(args.env_name, args.num_agents)
    obs_dim_for_policy = _policy_obs_dim(env, entity_mp, num_entities)
    pos_index = args.identity_size + 2

    for i, agent in enumerate(env.world.policy_agents):
        obs_dim = env.observation_space[i].shape[0]

        if hasattr(agent, "adversary") and agent.adversary:
            if policy1 is None:
                policy1 = MPNN(
                    input_size=obs_dim_for_policy,
                    num_agents=num_adversary,
                    num_entities=num_entities,
                    action_space=action_space,
                    pos_index=pos_index,
                    mask_dist=args.mask_dist,
                    entity_mp=entity_mp,
                    is_recurrent=args.is_recurrent,
                ).to(args.device)
            team1.append(Neo(args, policy1, (obs_dim,), action_space))
        else:
            if policy2 is None:
                policy2 = MPNN(
                    input_size=obs_dim_for_policy,
                    num_agents=num_friendly,
                    num_entities=num_entities,
                    action_space=action_space,
                    pos_index=pos_index,
                    mask_dist=args.mask_dist,
                    mask_obs_dist=args.mask_obs_dist,
                    entity_mp=entity_mp,
                    is_recurrent=args.is_recurrent,
                ).to(args.device)
            team2.append(Neo(args, policy2, (obs_dim,), action_space))

        _load_optional_pretrained_modules(args, policy1, policy2)

    master = learner_cls(args, [team1, team2], [policy1, policy2], env=env)

    if args.continue_training:
        print("Loading pretrained model")
        checkpoint = torch.load(args.load_dir, map_location=torch.device("cpu"))
        master.load_models(checkpoint["models"])

    if return_env:
        return master, env
    return master


def _num_entities_for_env(env_name, num_agents):
    if env_name == "simple_spread":
        return num_agents
    if env_name == "simple_formation":
        return 1
    if env_name == "simple_line":
        return 2
    raise NotImplementedError("Unknown environment, define entity_mp for this.")


def _policy_obs_dim(env, entity_mp, num_entities):
    obs_dim = env.observation_space[-1].shape[0]
    if entity_mp:
        return obs_dim - 2 * (2 * num_entities - 1)
    return obs_dim


def _load_optional_pretrained_modules(args, policy1, policy2):
    if hasattr(args, "load_low_level_path") and args.load_low_level_path is not None:
        print(f"Loading pretrained low-level model from: {args.load_low_level_path}")
        if policy1 is not None:
            policy1.load_pretrained_low_level(args.load_low_level_path, freeze=True)
        if policy2 is not None:
            policy2.load_pretrained_low_level(args.load_low_level_path, freeze=True)

    if hasattr(args, "load_high_level_path") and args.load_high_level_path is not None:
        print(f"Loading pretrained high-level model from: {args.load_high_level_path}")
        if policy1 is not None:
            policy1.load_pretrained_high_level(args.load_high_level_path, freeze=False)
        if policy2 is not None:
            policy2.load_pretrained_high_level(args.load_high_level_path, freeze=False)

    if hasattr(args, "load_high_critic_path") and args.load_high_critic_path is not None:
        print(f"Loading pretrained high-level critic from: {args.load_high_critic_path}")
        if policy1 is not None:
            policy1.load_pretrained_high_level(args.load_high_critic_path, freeze=False)
        if policy2 is not None:
            policy2.load_pretrained_high_level(args.load_high_critic_path, freeze=False)
