import numpy as np


def update_policies(teams_list, trainers_list, high_trainers_list,
                    use_pretrained_low_level):
    return_high_vals = []
    return_vals = []

    for i, trainer in enumerate(high_trainers_list):
        rollouts_list = [agent.rollouts for agent in teams_list[i]]
        high_vals = trainer.update(rollouts_list)
        return_high_vals.append([np.array(high_vals)] * len(rollouts_list))

    if use_pretrained_low_level:
        return_vals = [
            [np.array([0.0, 0.0, 0.0])] * len(teams_list[i])
            for i in range(len(teams_list))
        ]
    else:
        for i, trainer in enumerate(trainers_list):
            rollouts_list = [agent.rollouts for agent in teams_list[i]]
            vals = trainer.update(rollouts_list)
            return_vals.append([np.array(vals)] * len(rollouts_list))

    low_arr = np.stack([x for v in return_vals for x in v])
    high_arr = np.stack([x for v in return_high_vals for x in v])
    return np.concatenate([low_arr, high_arr], axis=1)
