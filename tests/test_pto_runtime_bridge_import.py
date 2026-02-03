import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def test_pto_runtime_bridge_imports_runtime_builder():
    import pto_wsp.pto_runtime_bridge as b

    rb = b.import_runtime_builder()
    assert hasattr(rb, "RuntimeBuilder")


def test_pto_runtime_bridge_imports_modules():
    import pto_wsp.pto_runtime_bridge as b

    assert hasattr(b.import_runtime_builder(), "RuntimeBuilder")
    assert hasattr(b.import_pto_compiler(), "PTOCompiler")
    assert hasattr(b.import_bindings(), "bind_host_binary")
    assert hasattr(b.import_elf_parser(), "extract_text_section")
