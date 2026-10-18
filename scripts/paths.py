"""Locations of the public source data. Override with environment variables if needed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DANDI_000774 = Path(os.environ.get('DANDI_000774', ROOT / 'data' / 'dandi_000774'))   # mouse stimulation NWB files
DANDI_000021 = Path(os.environ.get('DANDI_000021', ROOT / 'data' / 'dandi_000021'))   # Allen Neuropixels NWB files
ORGANOID_EVENTS = Path(os.environ.get('ORGANOID_EVENTS', ROOT / 'data' / 'organoid_events'))
