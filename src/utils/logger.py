import logging, logging.config, yaml, pathlib

def setup_logging(cfg_path: str = None):
    cfg_path = cfg_path or str(pathlib.Path(__file__).parents[2] / "config" / "logging_config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    logging.config.dictConfig(cfg)
    return logging.getLogger("moe")
