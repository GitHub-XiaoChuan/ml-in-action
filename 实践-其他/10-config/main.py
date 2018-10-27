from basic.config import cfg_from_file
from basic.config import cfg
import basic.config2 as config2

cfg_from_file('config.yml')
print(cfg.dev.name)
print(cfg.test.name)

cfg2 = config2.Config()
print(cfg2.name)
print(cfg2.age)