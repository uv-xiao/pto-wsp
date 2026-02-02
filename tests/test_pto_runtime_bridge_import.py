import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def test_pto_runtime_bridge_imports_runtime_builder():
    import pto_wsp.pto_runtime_bridge as b

    rb = b.import_runtime_builder()
    assert hasattr(rb, "RuntimeBuilder")

