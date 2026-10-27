"""Stand-in for the ``ezcolorlog`` package (coloured logging), which the Cambrian source imports at module level and
the StarVLA env does not have.  Only model_matrix/downstream/cambrian_entry.py puts this folder on the path, at its
end, so an installed package would win.  The Cambrian code uses the logger for messages only."""
import logging
import types

root_logger = logging.getLogger("cambrian")
_seen = set()


def _warning_once(self, msg, *args, **kwargs):  # the transformers-style helper the Cambrian source calls
    if msg not in _seen:
        _seen.add(msg)
        self.warning(msg, *args, **kwargs)


root_logger.warning_once = types.MethodType(_warning_once, root_logger)
