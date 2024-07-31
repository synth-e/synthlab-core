import json
from .registry import get_meta_of, get_cls, has_cls
from typing import Union
import os
import yaml

class GraphPipe(object):
    def __init__(self, config=None) -> None:
        self.config = config

        if config is not None:
            self.construct(config)

        self.vertices = {}
        self.edges = {}
        self.objects = {}

    def export(self):
        return {
            "vertices": [v for v in self.vertices.values()],
            "edges": [e for e in self.edges.values()]
        }

    @classmethod
    def load(cls, data: Union[str, dict]):
        if isinstance(data, str):            
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = yaml.safe_load(data)
            except yaml.YAMLError:
                if not os.path.exists(data):
                    raise ValueError("Invalid data. Not a valid json string or file path")
                
                with open(data, 'r') as f:
                    data = json.load(f)
            
            return cls(data)
        
        elif not isinstance(data, dict):
            raise ValueError("Invalid data. Not a valid json string or file path")

        return cls(data)
            
    
    def construct(self, config: dict):
        if 'edges' not in config or 'vertices' not in config:
            raise ValueError("Invalid config. At least one of 'edges' or 'vertices' is missing")

        for edge in config['edges']:
            self.add_edge(edge)

        for vertice in config['vertices']:
            self.add_vertice(vertice) 

    def add_edge(self, data):
        pass

    def add_vertice(self, data):
        vid, module_name = data['id'], data['module']

        if not has_cls(module_name):
            raise ValueError(f"Invalid module name #{vid}:{module_name}")

        self.vertices[vid] = get_cls(module_name)()
        self.objects[vid] = None

    def remove_edge(self, data):
        pass

    def remove_vertice(self, data):
        pass