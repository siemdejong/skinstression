import json
import logging
import os
from logging import Formatter
from logging.handlers import RotatingFileHandler
from pprint import pprint

from easydict import EasyDict

from utils.dirs import create_dirs


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        "{}exp_debug.log".format(log_dir), maxBytes=10**6, backupCount=5
    )
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        "{}exp_error.log".format(log_dir), maxBytes=10**6, backupCount=5
    )
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(json_file: str):
    """Processes configuration file.

    Sets up output directory structure.
    Sets up logging.

    Args:
        json_file: path to json file containing configurations.

    Returns:
        configuration object.
    """
    config, _ = get_config_from_json(json_file)
    print(" The experiment configuration:")
    pprint(config)

    # Ming sure that exp_name is provided.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("Please provide exp_name in json config file.")
        exit(-1)

    # Create experiment directory with logging childs.
    config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    config.out_dir = os.path.join("experiments", config.exp_name, "out/")
    config.log_dir = os.path.join("experiments", config.exp_name, "logs/")
    create_dirs(
        [config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir]
    )

    # Setup logging.
    setup_logging(config.log_dir)
    logging.getLogger().info("Logging directories are succesfully created.")

    return config
