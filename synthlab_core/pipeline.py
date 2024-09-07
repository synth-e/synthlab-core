import json
from .registry import get_cls, has_cls
from typing import Union
import os
import yaml
from .utilities.misc.decltype import type_compatible
from typing import List, Dict
from synthlab_core.atomic import AtomicType
from synthlab_core.node import INode
from synthlab_core.registry import get_cls, ClassType

class Response(object):
    module: str
    data: AtomicType

class Edge(object):
    src: str
    dest: str
    gate: str

class GraphPipe(object):
    def __init__(self, config=None, batch_size=64, preload=False) -> None:
        self.config = config

        if config is not None:
            self.construct(config, load=preload)

        self.dependencies, self.topo = self._compile()

        self.vertices: Dict[str, INode] = {}
        self._objects = {}
        self._params = {}
        self.edges: List[Edge] = []
        self.batch_size = batch_size
        
    def _compile(self):
        """
        Return the dependencies graph and topo order
        """

        if len(self.vertices) == 0:
            return None, None

        dependencies_graph = {
            id: [None] * len(vertice.in_specs()) for id, vertice in self.vertices.items()
        }

        depends = {id: [] for id in self.vertices}

        depends_cnt = {id: 0 for id in self.vertices}

        for e in self.edges:
            specs = self.vertices[e.dest].in_specs()

            for i, field_spec in enumerate(specs):
                if e.gate == field_spec[0]:
                    dependencies_graph[e.dest][i] = e.src
                    depends[e.src].append(e.dest)
                    depends_cnt[e.dest] += 1
                    break

        que = [id for id in self.vertices if depends_cnt[id] == 0]
        topo = []

        while len(que) > 0:
            cur = que.pop(0)
            topo.append(cur)

            for d in depends[cur]:
                depends_cnt[d] -= 1
                if depends_cnt[d] == 0:
                    que.append(d)

        return dependencies_graph, topo

    def io(self, _dependencies=None, _topo=None):
        if _dependencies is None or _topo is None:
            dependencies, topo = self._compile()

        else:
            dependencies, topo = _dependencies, _topo

        if dependencies is None or topo is None:
            return {"inputs": [], "outputs": []}

        in_specs, out_specs = [], []

        for k, v in dependencies.items():
            specs = self.vertices[k].in_specs()

            for i, d in enumerate(v):
                if d is None:
                    in_specs.append([k, specs[i]])

        return {
            "inputs": in_specs,
            "outputs": out_specs,
            "dependencies": dependencies,
            "topo": topo,
        }

    def summary(self):
        return {
            "vertices": [v for v in self.vertices.values()],
            "edges": [e for e in self.edges.values()],
            "io": {
                **self.io()
            }
        }

    @classmethod
    def from_file(cls, data: Union[str, dict], load=False):
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
            
            return cls(data, load=load)
        
        elif not isinstance(data, dict):
            raise ValueError("Invalid data. Not a valid json string or file path")

        return cls(data, load=load)
            
    def load(self):
        for e in self.vertices:
            if self._objects[e] is None:
                self._objects[e] = self.vertices[e](**self._params[e])
    
    def construct(self, config: dict):
        if 'vertices' not in config:
            raise ValueError("Invalid config. At least one of 'edges' or 'vertices' is missing")

        for vertice in config['vertices']:
            self.add_vertice(vertice) 

        for edge in config['edges']:
            self.add_edge(edge)

    def add_edge(self, data):
        if any(e not in data for e in ['src', 'dest']):
            raise ValueError("Invalid edge data. 'src' and 'dest' keys are required")

        src, dest = data['src'], data['dest']
        dest_gate = dest.get('field', None)

        if dest_gate is None:
            raise ValueError("Invalid edge data. 'field' key is required for destination")

        src_module: INode = self.vertices.get(src['id'], None)
        dest_module: INode = self.vertices.get(dest['id'], None)

        if src_module is None:
            raise ValueError(f"Invalid source module #{src['id']}")

        if dest_module is None:
            raise ValueError(f"Invalid destination module #{dest['id']}")

        out_dtype = src_module.out_dtype()
        in_dtype = dest_module.in_dtype(key=dest_gate)

        if not type_compatible(out_dtype, in_dtype):
            raise ValueError(f"Invalid edge. Incompatible data types: {out_dtype} -> {in_dtype}")
        
        for edge in self.edges:
            if edge.dest == dest['id'] and edge.gate == dest_gate:
                raise ValueError(f"Edge already exists: {src['id']} -> {dest['id']}:{dest_gate}")

        self.edges.append(Edge(src=src['id'], dest=dest['id'], gate=dest_gate))

    def add_vertice(self, data, load=False):
        vid, module_name, params = data['id'], data['module'], data['params']

        if not has_cls(module_name):
            raise ValueError(f"Invalid module name #{vid}:{module_name}")

        self.vertices[vid] = get_cls(module_name)
        self._params[vid] = params
        self._objects[vid] = None if not load else self.vertices[vid](**params)

    def remove_edge(self, data):
        '''
        @TODO: Implement this method
        '''
        raise NotImplementedError("Method not implemented")

    def remove_vertice(self, data):
        '''
        @TODO: Implement this method
        '''
        raise NotImplementedError("Method not implemented")
    
    def __call__(self, data: dict, *args, **kwargs):
        dependencies, topo = self._compile()
    
        if dependencies is None or topo is None:
            return
        
        io_cfg = self.io(dependencies, topo)
        inp_cfg = io_cfg["inp_cfg"]
        inp_pool, init_pool = {}, {}        
        
        for k, c, _ in inp_cfg:
            key = f"{k}.{c[0]}"

            if key not in data:
                raise ValueError(f"Missing input data {key}")
            else:
                init_pool[key] = get_cls(ClassType.DATA_TYPE, c[1])
                
                if init_pool[key] == None:
                    raise ValueError(f"Invalid data type {c[1]}")
                
                init_pool[key](data[key])
        
        for t in topo:
            dependency = dependencies[t]
            meta = self._meta[t]

            for i, v in enumerate(dependency):
                if v is not None:
                    continue

                field_name, _ = meta["in_specs"][i]
                dependency[i] = f"{t}.{field_name}"

            inps =[inp_pool.get(d, None) for d in dependency]
            inps  = [x if x is not None else init_pool[y] 
                     for x, y in zip(inps, dependency)]
            
            t_output = self.objects[t].__call__(*inps)
            k, _ = meta["out_specs"][0]
            
            yield t, t_output

    def prepare_input(self, data):
        init_pool = {}

        io_cfg = self.io(self.dependencies, self.topo)
        inp_cfg = io_cfg["inp_cfg"]

        for k, c, _ in inp_cfg:
            key = f"{k}.{c[0]}"

            if key not in data:
                raise ValueError(f"Missing input data {key}")
            else:
                init_pool[key] = get_cls(ClassType.DATA_TYPE, c[1])

                if init_pool[key] == None:
                    raise ValueError(f"Invalid data type {c[1]}")

                init_pool[key](data[key])

        return init_pool
 
    def __batch_call__(self, data, *args, **kwargs):
        if self.dependencies is None or self.topo is None:
            return
        
        inp_pool, init_pool = {}, self.prepare_input(data)
        
        for t in self.topo:
            dependency = self.dependencies[t]
            inps  = [
                inp_pool.get(d, None) 
                if x is not None else init_pool[d] 
                for x, d in zip(inps, dependency)
            ]

            t_output = self.objects[t].__batch_call__(*inps)

            yield t, t_output
