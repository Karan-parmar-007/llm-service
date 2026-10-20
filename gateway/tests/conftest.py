import os
import sys

# Tests run both halves with a stubbed speech model and no token.
os.environ["SPEECH_ENABLED"] = "1"
os.environ["LLM_ENABLED"] = "1"
os.environ["TARA_EAGER"] = "0"
os.environ["TARA_STUB"] = "1"
os.environ["FLOW_TOKEN"] = ""
os.environ["KEEPALIVE_ENABLED"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
