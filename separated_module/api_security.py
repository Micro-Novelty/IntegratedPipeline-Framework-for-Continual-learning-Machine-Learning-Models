"""Request-level security helpers: RateLimiter, InputSanitizer, ApiKeyManager."""

from datetime import datetime
from typing import Dict
import hashlib
import logging
import re
import secrets
import threading
import time
from .security_config import SecurityError

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.

    Supports both:
      - a single shared bucket (original behavior, backward compatible)
      - independent per-key buckets (e.g. per peer IP/agent_id),
        so one noisy peer cannot starve rate-limit capacity from others
    """

    def __init__(self, requests_per_minute: int = 60, per_key: bool = False,
                max_keys: int = 1000):
        #validate config, refuse a limiter that can never refill
        if requests_per_minute <= 0:
            raise ValueError(
                f"[-] requests_per_minute must be > 0, got {requests_per_minute}"
            )

        self.requests_per_minute = requests_per_minute
        self.per_key = per_key
        self.max_keys = max_keys
        self._lock = threading.Lock()

        if per_key:
            # one bucket per key, so peers don't share capacity here
            self._buckets: Dict[str, dict] = {}
        else:
            self.tokens      = float(requests_per_minute)
            self.last_refill = time.time()

    def acquire(self, key: str = None) -> bool:
        """Try to take a token from the global bucket (or the per-key bucket when `per_key`). Returns True if allowed."""
        with self._lock:
            if self.per_key:
                return self._acquire_keyed(key or "_default")
            return self._acquire_global()

    def _acquire_global(self) -> bool:
        now     = time.time()
        elapsed = now - self.last_refill

        # guarded against clock skew producing negative elapsed
        if elapsed < 0:
            logger.warning(
                f'[!] RateLimiter: system clock moved backward by '
                f'{-elapsed:.3f}s — ignoring this interval for refill'
            )
            elapsed = 0.0

        new_tokens   = elapsed * (self.requests_per_minute / 60.0)
        self.tokens  = min(self.requests_per_minute, self.tokens + new_tokens)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def _acquire_keyed(self, key: str) -> bool:
        now = time.time()

        if key not in self._buckets:
            # cap number of tracked keys to prevent unbounded
            # growth from an attacker cycling through many fake peer IDs
            if len(self._buckets) >= self.max_keys:
                self._evict_oldest_bucket()

            self._buckets[key] = {
                'tokens': float(self.requests_per_minute),
                'last_refill': now,
            }

        bucket  = self._buckets[key]
        elapsed = now - bucket['last_refill']

        if elapsed < 0:
            logger.warning(
                f'[!] RateLimiter[{key}]: clock moved backward by '
                f'{-elapsed:.3f}s — ignoring interval'
            )
            elapsed = 0.0

        new_tokens = elapsed * (self.requests_per_minute / 60.0)
        bucket['tokens'] = min(self.requests_per_minute, bucket['tokens'] + new_tokens)
        bucket['last_refill'] = now

        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        return False

    def _evict_oldest_bucket(self):
        """Evict the least-recently-refilled bucket to bound memory use."""
        if not self._buckets:
            return
        oldest_key = min(
            self._buckets.keys(),
            key=lambda k: self._buckets[k]['last_refill']
        )
        del self._buckets[oldest_key]

    def get_wait_time(self, key: str = None) -> float:
        """
        Seconds until at least 1 token will be available.
        Useful for callers that want to back off.
        """
        with self._lock:
            if self.per_key:
                bucket = self._buckets.get(key or "_default")
                tokens = bucket['tokens'] if bucket else self.requests_per_minute
            else:
                tokens = self.tokens

            if tokens >= 1:
                return 0.0
            tokens_needed = 1 - tokens
            return tokens_needed / (self.requests_per_minute / 60.0)

    def get_stats(self) -> Dict:
        """Visibility into limiter state"""
        with self._lock:
            if self.per_key:
                return {
                    'mode'          : 'per_key',
                    'tracked_keys'  : len(self._buckets),
                    'max_keys'      : self.max_keys,
                    'requests_per_minute': self.requests_per_minute,
                }
            return {
                'mode'          : 'global',
                'current_tokens': round(self.tokens, 2),
                'requests_per_minute': self.requests_per_minute,
            }


class InputSanitizer:
    """Sanitize and validate inputs."""

    # comprehensive control-char stripping including
    # newline/CR, since this class explicitly exists to protect
    # logging/serialization from injection from unknown source.
    _CONTROL_CHARS_PATTERN = re.compile(
        r'[\x00-\x1f\x7f]'   
    )

    # OPT-IN for genuinely
    # multi-line legitimate text (e.g. free-form descriptions).
    _CONTROL_CHARS_ALLOW_WHITESPACE = re.compile(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'   
    )

    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000,
                      allow_newlines: bool = False) -> str:
        """Validate and clean a text input: type and length checks (before any transformation), control-character removal, optional newline handling. Raises SecurityError on invalid input."""
        if not isinstance(text, str):
            raise SecurityError("[-] Input must be a string")

        # check length BEFORE any transformation, so length
        # limits reflect what was actually submitted.
        if len(text) > max_length:
            raise SecurityError(f"[-] Text exceeds maximum length of {max_length}")

        if allow_newlines:
            text = InputSanitizer._CONTROL_CHARS_ALLOW_WHITESPACE.sub('', text)
            # even when newlines are allowed, escape them for log safety in here although
        else:
            text = InputSanitizer._CONTROL_CHARS_PATTERN.sub('', text)

        escape_patterns = [
            r'\\x[0-9a-fA-F]{2}',   # literal "\x41" style escapes.
            r'\\u[0-9a-fA-F]{4}',   # literal "\u0041" style escapes
        ]
        for pattern in escape_patterns:
            if re.search(pattern, text):
                logger.warning(
                    f'[!] sanitize_text: literal escape sequence pattern '
                    f'detected and neutralized (pattern={pattern})'
                )
                text = re.sub(pattern, '?', text)

        return text.strip()

    @staticmethod
    def validate_batch_size(size: int, max_batch: int = 100) -> bool:
        """Ensure `size` is an int in [1, max_batch]; raises SecurityError otherwise."""
        # validated type before comparison to avoid a confusing
        # TypeError if a non-int slips through
        if not isinstance(size, int):
            raise SecurityError(f"[-] Batch size must be an integer, got {type(size).__name__}")
        if size <= 0 or size > max_batch:
            raise SecurityError(f"[-] Batch size must be between 1 and {max_batch}")
        return True


class ApiKeyManager:
    """Manage API keys with rotation and bounded storage."""

    MIN_KEY_LENGTH = 16   # minimum acceptable length for caller-supplied keys

    def __init__(self, rotation_days: int = 30, max_keys: int = 10000):
        self.keys: Dict[str, dict] = {}
        self.rotation_days = rotation_days
        self.max_keys = max_keys  
        self._lock = threading.Lock()

    def generate_key(self, metadata: dict = None, key_value: str = None) -> str:
        """Create (or register a caller-supplied, minimum-length) API key; only its SHA-256 hash and metadata are stored. Returns the plain key."""
        if key_value:
            # reject obviously weak caller-supplied keys
            if not isinstance(key_value, str) or len(key_value) < self.MIN_KEY_LENGTH:
                raise SecurityError(
                    f"[-] Provided key must be a string of at least "
                    f"{self.MIN_KEY_LENGTH} characters"
                )
            raw_key = key_value
        else:
            raw_key = secrets.token_urlsafe(32)

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with self._lock:
            # warn on collision 
            if key_hash in self.keys and self.keys[key_hash].get('is_active'):
                logger.warning(
                    f'[!] generate_key: key hash collision with an '
                    f'already-active key — overwriting metadata'
                )

            # evict oldest inactive entries if at capacity,
            if len(self.keys) >= self.max_keys:
                self._evict_oldest_inactive()

            self.keys[key_hash] = {
                'created_at': datetime.now(),
                'last_used' : None,
                'metadata'  : metadata or {},
                'is_active' : True
            }

        return raw_key

    def validate_key(self, api_key: str) -> bool:
        """Return True if the key's hash is known and still active; False otherwise (including non-string input)."""
        # explicit type check before .encode(), avoids an
        # uncontrolled AttributeError on non-string input
        if not api_key or not isinstance(api_key, str):
            return False

        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        except (UnicodeEncodeError, AttributeError):
            return False

        with self._lock:
            key_info = self.keys.get(key_hash)
            if not key_info or not key_info.get('is_active', False):
                return False

            age = (datetime.now() - key_info['created_at']).days
            if age >= self.rotation_days:
                key_info['is_active'] = False
                return False

            key_info['last_used'] = datetime.now()
            return True

    def revoke_key(self, api_key: str) -> bool:
        """Mark the key as inactive. Returns True if the key existed."""
        if not api_key or not isinstance(api_key, str):
            return False

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._lock:
            if key_hash in self.keys:
                self.keys[key_hash]['is_active'] = False
                return True
        return False

    def cleanup_expired(self, grace_period_days: int = 7):
        """
        periodic sweep to actually remove long-inactive keys.
        """
        with self._lock:
            now = datetime.now()
            to_remove = [
                key_hash for key_hash, info in self.keys.items()
                if not info.get('is_active', False)
                and (now - info['created_at']).days >= (self.rotation_days + grace_period_days)
            ]
            for key_hash in to_remove:
                del self.keys[key_hash]

            if to_remove:
                logger.info(f'[=] Cleaned up {len(to_remove)} expired API key entries')

            return len(to_remove)

    def _evict_oldest_inactive(self):
        """Evict the oldest inactive key to make room, called under lock."""
        inactive = [
            (h, info) for h, info in self.keys.items()
            if not info.get('is_active', False)
        ]
        if inactive:
            oldest_hash = min(inactive, key=lambda x: x[1]['created_at'])[0]
            del self.keys[oldest_hash]
        else:
            # no inactive keys to evict and still at capacity 
            logger.warning(
                f'[!] ApiKeyManager at max_keys={self.max_keys} capacity '
                f'with no inactive keys to evict — consider raising max_keys '
                f'or auditing why so many keys remain active'
            )

    def get_stats(self) -> Dict:
        """Visibility into key store health — same pattern as WorkerPool.get_health()."""
        with self._lock:
            active   = sum(1 for k in self.keys.values() if k.get('is_active'))
            inactive = len(self.keys) - active
            return {
                'total_keys'   : len(self.keys),
                'active_keys'  : active,
                'inactive_keys': inactive,
                'at_capacity'  : len(self.keys) >= self.max_keys,
            }
