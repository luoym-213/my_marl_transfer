try:
    from multiagent.maps.belief_map import GlobalBeliefMap
except ModuleNotFoundError as exc:
    if exc.name != "multiagent":
        raise
    from maps.belief_map import GlobalBeliefMap

__all__ = ["GlobalBeliefMap"]
