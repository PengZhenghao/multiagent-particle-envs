import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='routing2D-v0',
    entry_point='gym_routing.envs:routing2D',
    # timestep_limit=1000,
    # reward_threshold=1.0,
    # nondeterministic = True,
)