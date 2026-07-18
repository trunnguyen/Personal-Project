from pathlib import Path
import yaml

class Config:
    def __init__(self, config_path="configs/config.yml"):
        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def get(self,*keys):

        value = self.config
        for key in keys:
            value = value[key]

        return value

if __name__ == "__main__":
    config = Config()
    print(config.get("paths","input_dir"))