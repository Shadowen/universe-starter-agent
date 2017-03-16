import gym
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_env(env_id, client_id, remotes, **kwargs):
    return gym.make(env_id)
