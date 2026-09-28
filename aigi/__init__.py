"""Additive AIGI01 research extension; importing does not download any asset."""
import sys
sys.dont_write_bytecode = True  # Original upstream tracks .pyc; never overwrite those files.
__version__ = '0.2.0'
UPSTREAM_COMMIT = 'cd43f5d9761fb34f5224622145629d3ff2b89ca1'
