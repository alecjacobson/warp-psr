"""Run the established numerical suite against the relocated private implementation."""
import ast
import importlib
from pathlib import Path
import sys
import types

import pytest

root = Path('/home/horde/projects/warp-psr')
source = root / 'src/warp_psr'
package = types.ModuleType('warp_psr')
package.__path__ = ['/home/horde/projects/warp-field-composition/warp/_src/geometry/poisson']
sys.modules['warp_psr'] = package
for path in source.glob('*.py'):
    if path.name == '__init__.py':
        continue
    module = importlib.import_module('warp._src.geometry.poisson.' + path.stem)
    sys.modules['warp_psr.' + path.stem] = module
    setattr(package, path.stem, module)
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and hasattr(module, node.name + '_kernel'):
            setattr(module, node.name, getattr(module, node.name + '_kernel'))
package.reconstruct = package.reconstruction.reconstruct
package.Reconstruction = package.reconstruction.Reconstruction
package.reconstruct_differentiable = package.reconstruction_adjoint.reconstruct_differentiable
package.LevelField = package.field.LevelField
for old, new in [('Field', 'PoissonField'), ('Field64', 'PoissonField64'), ('evaluate_field', 'poisson_evaluate')]:
    setattr(package, old, getattr(package.device_field, new))
    setattr(package.device_field, old, getattr(package.device_field, new))
sys.path.insert(0, str(root/'tests'))
raise SystemExit(pytest.main([str(root/'tests'), '-q', '--disable-warnings', '--maxfail=5', *sys.argv[1:]]))
