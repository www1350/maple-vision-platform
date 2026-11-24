"""Offline visual research and deterministic simulation.

Disable ONNX Runtime's process-level uploader before any optional native import.
Embedding applications must set the same variable before importing ORT themselves.
"""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
