"""The Windows distribution includes the standard PortAudio backend only."""

import os

# sounddevice selects an ASIO-only filename whenever this variable is present.
# Keep the packaged backend usable even when the parent shell enables ASIO.
os.environ.pop("SD_ENABLE_ASIO", None)
