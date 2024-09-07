from argparse import ArgumentParser
from .pipeline import GraphPipe, Response
import json, yaml

def load_config_from_file(config_file: str) -> GraphPipe:
    pass

# for cli
def get_options():
    parser = ArgumentParser(description="Synthlab CLI")

    parser.add_argument("-c", "--config", 
                        help="Configuration file", required=True)

    # action to the config file
    parser.add_argument("-s", "--summary", help="Summary of the pipeline", 
                        action="store_true", )
    parser.add_argument("-vv", "--verify", help="Verify the config file", 
                        action="store_true", required=False)
    parser.add_argument("-vis", "--visualize", type=str, help="Produce a visualization of the pipeline", 
                        required=False, default=None)
    parser.add_argument("-e", "--execute", type=str, help="Execute the pipeline with the given input file(s)", 
                        required=False, default=None, nargs='+')

    return parser.parse_args()

def cli():
    pass

if __name__ == "__main__":
    cli()