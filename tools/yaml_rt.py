"""Round-trip editing of figshare_metadata.yaml (keeps comments and formatting)."""

from ruamel.yaml import YAML


def _yaml():
    y = YAML()
    y.preserve_quotes = True
    y.width = 110
    y.indent(mapping=2, sequence=2, offset=0)
    return y


def load(path):
    with open(path) as f:
        return _yaml().load(f)


def dump(data, path):
    with open(path, "w") as f:
        _yaml().dump(data, f)
