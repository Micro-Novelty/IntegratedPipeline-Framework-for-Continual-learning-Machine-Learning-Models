"""Run the interactive smoke test: ``python -m AbstractIntegratedModule``."""
import traceback

from .cli_tests import PermissiveTest

if __name__ == "__main__":
    try:
        PermissiveTest()
    except Exception as e:
        print(f'|| Program Crashed...,  Error: {e}')
        traceback.print_exc()
        pass
