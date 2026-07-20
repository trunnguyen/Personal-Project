from pathlib import Path
import yaml


class Config:

    def __init__(self, config_path="configs/config.yml"):

        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        paths = self.config["paths"]

        self.input_dir = Path(paths["input_dir"])

        self.output_dir = Path(paths["output_dir"])

        self.knowledge_dir = Path(paths["knowledge_dir"])

        self.drug_dictionary_path = Path(paths["drug_dictionary"])

        self.symptom_dictionary_path = Path(paths["symptom_dictionary"])

        self.diagnosis_dictionary_path = Path(paths["diagnosis_dictionary"])

    def get(self, *keys):

        value = self.config

        for key in keys:
            value = value[key]

        return value


if __name__ == "__main__":

    config = Config()

    print(config.input_dir)

    print(config.drug_dictionary_path)

    print(config.symptom_dictionary_path)

    print(config.diagnosis_dictionary_path)