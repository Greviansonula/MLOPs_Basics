from omegaconf import OmegaConf

# # Loading
# config = OmegaConf.load('config.yaml')

# # accessing
# print(config.preferences.user)
# print(config["preferences"]["trait"])

# Hydra
import hydra
from hydra import initialize, compose

# @hydra.main(config_name="config.yaml")
# def main(cfg):
#     # print the config file using `to_yaml` method
#     print(OmegaConf.to_yaml(cfg))
#     print(cfg.preferences.user)

# if __name__ == "__main__":
#     main()

initialize(".") # Assumes the config file is in current dir
cfg = compose(config_name="config.yaml")
print(OmegaConf.to_yaml(cfg))