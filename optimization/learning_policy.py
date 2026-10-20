"""Personal-learning dispatch policy: train locally (native) or defer (external).

A leaf module (stdlib + ``config_resolve`` only) so every trainer, the retrain
scheduler and the comparison routes can import it at module top without a cycle.
"""

import logging
import sys

logger = logging.getLogger(__name__)

_BACKENDS = ('native', 'external')
# Distinct invalid policy values already warned about, so a per-request caller
# does not repeat the warning on every call.
_warned_values: set = set()


def _warn_once(value, reason, exc_info=False):
    key = repr(value)
    if key in _warned_values:
        return
    _warned_values.add(key)
    logger.warning(
        "Personal learning deferred: auto_retrain %s (%s); failing closed to 'external'",
        reason, key, exc_info=exc_info,
    )


def learning_backend(config_path=None):
    """Explicit dispatch policy: native(default) or retained external work.

    Malformed/unreadable policy never silently enables local learned inference.
    A valid config with no backend keeps the historical native behavior.
    """
    try:
        # The viewer can relocate its library config; use its authoritative
        # already-loaded path without importing API/auth side effects in a CLI.
        if config_path is None:
            server = sys.modules.get('api.config')
            server_path = getattr(server, 'server_config_path', None)
            if callable(server_path):
                config_path = server_path()
        from config_resolve import load_resolved
        block = load_resolved(config_path).get('auto_retrain', {})
        if not isinstance(block, dict):
            _warn_once(block, 'block is not an object')
            return 'external'
        backend = block.get('backend', 'native')
        if backend in _BACKENDS:
            return backend
        _warn_once(backend, 'backend must be "native" or "external"')
        return 'external'
    except Exception as ex:  # noqa: BLE001 -- policy failure defers, never CPU fallback
        _warn_once(type(ex).__name__, 'backend policy could not be read', exc_info=True)
        return 'external'


def external_learning_result():
    """Honest deferred result, not a claim an external job was dispatched."""
    return {'error': 'Preferences retained; external GPU learning has not been dispatched here',
            'deferred': True, 'backend': 'external', 'written': 0}


def deferred_if_external(config_path=None):
    """The deferred result when the policy is not native, else None."""
    if learning_backend(config_path) != 'native':
        return external_learning_result()
    return None


def unavailable_if_external(config_path=None):
    """Compare-endpoint form of ``deferred_if_external``: ``available: False``."""
    deferred = deferred_if_external(config_path)
    if deferred is None:
        return None
    return {'available': False, 'message': deferred['error']}
