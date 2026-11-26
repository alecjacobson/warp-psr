"""Audit ordered kernel expressions after namespace, annotation and naming changes."""
import ast
import copy
import hashlib
import json
from pathlib import Path

source = Path('/home/horde/projects/warp-psr/src/warp_psr')
target = Path('/home/horde/projects/warp-field-composition/warp/_src/geometry/poisson')


def identifier(name):
    name = name.removesuffix('_kernel').lstrip('_')
    return {'evaluate_field': 'poisson_evaluate'}.get(name, name)


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = identifier(node.id)
        return node

    def visit_Attribute(self, node):
        node.attr = identifier(node.attr)
        return self.generic_visit(node)

    def visit_arg(self, node):
        node.arg = identifier(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.name = identifier(node.name)
        node.decorator_list = []
        node.returns = None
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return self.generic_visit(node)


def functions(path):
    result = {}

    def walk(node, parents=()):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            parents = (*parents, identifier(node.name))
        if isinstance(node, ast.FunctionDef):
            decorators = [ast.unparse(d) for d in node.decorator_list]
            if any(d.startswith(('wp.kernel', 'wp.func')) for d in decorators):
                normalized = Normalize().visit(copy.deepcopy(node))
                result['.'.join(parents)] = ast.dump(normalized, include_attributes=False)
        for child in ast.iter_child_nodes(node):
            walk(child, parents)
    walk(ast.parse(path.read_text()))
    return result


rows = []
for path in source.glob('*.py'):
    if path.name == '__init__.py':
        continue
    original, port = functions(path), functions(target/path.name)
    for name, before in original.items():
        after = port.get(name)
        rows.append({'module': path.stem, 'function': name, 'match': before == after,
                     'original_sha256': hashlib.sha256(before.encode()).hexdigest(),
                     'port_sha256': hashlib.sha256(after.encode()).hexdigest() if after else None})
report = {'scope': 'Compare ordered expression ASTs for kernel/device functions. Normalize renamed identifiers, omit type annotations, decorators and docstrings. Complements numerical tests; not a proof of compilation or host-path equivalence.',
          'functions': rows, 'matched': sum(r['match'] for r in rows), 'total': len(rows)}
Path(__file__).with_name('kernel-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print('Matched',report['matched'],'of',report['total'])
for row in rows:
    if not row['match']:
        print('DIFFERENCE',row['module'],row['function'])
