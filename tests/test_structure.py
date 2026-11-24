"""Keep application dependencies acyclic and GUI validation independent."""

import ast
import builtins
from graphlib import TopologicalSorter
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))


class StructureTests(unittest.TestCase):
    def test_application_imports_are_acyclic_including_lazy_imports(self):
        modules = {path.stem: path for path in SOURCE.glob("*.py")}
        dependencies = {}
        for name, path in modules.items():
            imports = set()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            dependencies[name] = imports & modules.keys()
        tuple(TopologicalSorter(dependencies).static_order())

    def test_gui_validation_does_not_import_dispatch_or_media(self):
        from gui import build_arguments

        original_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name in ("cli", "player", "export", "av", "sounddevice", "tkinter"):
                raise AssertionError(f"Unexpected dependency during validation: {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=SOURCE.parent / "tests") as directory:
            source = Path(directory) / "input.mp4"
            source.touch()
            with patch("builtins.__import__", side_effect=guard):
                _, options = build_arguments("play", {"input": str(source)})
                self.assertEqual(options.command, "play")
                with self.assertRaises(ValueError):
                    build_arguments("play", {"input": str(source), "gamma": "nan"})


if __name__ == "__main__":
    unittest.main()
