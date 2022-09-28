"""
__author__ = "Siem de Jong"
Project structure inspired by Hager Rady and Mo'men AbdelRazek (moemen95/Pytorch-Project-Template).

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse

from agents.agent import THGStrainStressAgent
from utils.config import *


def main():
    print("test")
    print(globals())
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "config",
        metavar="config_json_file",
        default="None",
        help="The Configuration file in json format",
    )
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    # Create the Agent and pass all the configuration to it then run it..
    # print(config.agent, type(config.agent))
    # assert config.agent in [THGStrainStressAgent]
    # print(config)
    agent = THGStrainStressAgent(config)
    agent.run()
    agent.finalize()


if __name__ == "__main__":
    main()
