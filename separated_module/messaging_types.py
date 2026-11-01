"""Message, queue and request data types, enums and priorities shared across the async/agent layers."""

import asyncio
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from enum import Enum, IntEnum


class MessagePriority(Enum):
    """Priority levels for queued messages (LOW < NORMAL < HIGH < CRITICAL)."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class WrapperState(Enum):
    """Wrapper state machine."""
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AsyncTask:
    """Track async tasks for proper cleanup."""
    id: str
    future: asyncio.Future
    created_at: float
    callback: Optional[Callable] = None
    timeout: float = 30.0


class TrustLevel(IntEnum):
    """Trust levels for peer agents"""
    UNTRUSTED = 0      # No trust - will be rejected
    BASIC = 1          # Basic trust - limited operations
    STANDARD = 2       # Standard trust - most operations
    HIGH = 3           # High trust - sensitive operations
    FULL = 4           # Full trust - administrative access


class RequestStatus(Enum):
    """Lifecycle states of an async prediction request: pending, processing, completed, failed."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AsyncRequest:
    """Track an async prediction request"""
    request_id: str
    texts: Any
    api_key: Optional[str]
    client_ip: Optional[str]
    callback: Optional[Callable] = None
    webhook_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None
    
    @property
    def age(self) -> float:
        """Seconds elapsed since the request was created."""
        return time.time() - self.created_at


    @property
    def is_expired(self, timeout: int = 30) -> bool:
        """True when the request is older than `timeout` seconds (default 30)."""
        return self.age > timeout


@dataclass
class SecureMessage:
    """Lightweight message envelope (id, type, payload, timestamp) for secure peer exchanges."""
    # Secure message wrapper
    id: str
    type: str
    payload: Any
    timestamp: float
    signature: str = ""


@dataclass
class Message:
    """Message passed between agents / queues.

    Carries sender, recipient, payload, priority, an optional callback,
    retry/timeout settings and a `trust` score in [0, 1] that is lowered when a
    peer misbehaves.
    """
    id: str
    type: str
    sender: str
    recipient: str
    payload: Any
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    callback: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    trust: float = 1.0    

    @property
    def age(self) -> float:
        """Age of message in seconds."""
        return time.time() - self.created_at

    @property
    def proper_trust(self) -> bool:
        """True while the sender's trust score is above 0.3."""
        return self.trust > 0.3

    @property
    def degrade_trust(self) -> bool:
        """Lower the sender's trust score by 0.1 (called after a suspicious/failed exchange)."""
        self.trust = self.trust - 0.1 

    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        expired = self.age > self.timeout
        if expired:
            self.degrade_trust()
        return expired
    
    # ============ COMPARISON METHODS FOR PRIORITY QUEUE ============
    
    def __lt__(self, other):
        """Less than comparison for priority queue."""
        if not isinstance(other, Message):
            return NotImplemented
        
        # Compare by priority value first (lower number = higher priority)
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        
        # If same priority, compare by creation time (older messages get processed first)
        return self.created_at < other.created_at
    
    def __le__(self, other):
        """Less than or equal."""
        if not isinstance(other, Message):
            return NotImplemented
        return self.__lt__(other) or self.__eq__(other)
    
    def __eq__(self, other):
        """Equality comparison."""
        if not isinstance(other, Message):
            return NotImplemented
        return self.id == other.id
    
    def __ne__(self, other):
        """Not equal."""
        if not isinstance(other, Message):
            return NotImplemented
        return not self.__eq__(other)
    
    def __gt__(self, other):
        """Greater than."""
        if not isinstance(other, Message):
            return NotImplemented
        return not self.__lt__(other) and not self.__eq__(other)
    
    def __ge__(self, other):
        """Greater than or equal."""
        if not isinstance(other, Message):
            return NotImplemented
        return not self.__lt__(other)
    
    def __hash__(self):
        """Message hashable"""
        return hash(self.id)
