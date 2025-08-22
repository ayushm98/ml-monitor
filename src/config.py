"""Configuration management."""
import yaml
from pathlib import Path

def load_config():
    """Load configuration from YAML."""
    config_path = Path("config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()
