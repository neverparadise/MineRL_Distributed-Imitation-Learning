import ray
import unittest
from unittest.mock import Mock, MagicMock, call

ray.init()

@ray.remote
def make_env(index):
    env = Mock()
    env.return_value = "Environment %d is created" %index
    port_number = int("12340")+index
    return env()
num_envs = 4

class EnvTests(unittest.TestCase):
    def test1(self):
        make_env()

    def test2(self):
        num_envs = 4
        env_list = [make_env.remote(i) for i in range(num_envs)]
        print(ray.get(env_list))

if __name__=='__main__':
    unittest.main()