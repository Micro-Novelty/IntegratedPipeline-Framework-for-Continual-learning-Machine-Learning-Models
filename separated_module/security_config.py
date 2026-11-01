"""Security configuration objects: security levels, admin roles, SecurityConfig and SecurityError."""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


@dataclass
class SecurityConfig:
    """Security configuration for async manager"""
    max_text_length: int = 10000
    max_queue_size: int = 100
    max_pending_tasks: int = 50
    rate_limit_requests: int = 60  # per minute
    rate_limit_window: int = 60  # seconds
    request_timeout: float = 30.0
    max_concurrent: int = 10
    enable_auth: bool = True
    allowed_ips: List[str] = field(default_factory=list)  # Empty = allow all
    blocklisted_ips: List[str] = field(default_factory=list)
    require_api_key: bool = True
    api_key_rotation_days: int = 30

    # Admin-specific settings
    admin_bypass_rate_limit: bool = True  # Admins bypass rate limiting
    admin_bypass_ip_check: bool = False   # Admins still need IP whitelist
    enforce_admin_ip_whitelist: bool = True  # Separate admin IP whitelist
    admin_allowed_ips: List[str] = field(default_factory=list)  # Admin-specific IPs
    admin_rate_limit: int = 300  # Higher limit for admins (per minute)
    log_all_admin_actions: bool = True
    
    # Start protection (NOT authentication)
    min_start_interval: float = 5.0  # Seconds between start attempts
    max_consecutive_failures: int = 3  # Before circuit breaker
    max_cpu_percent: float = 99.0  # Don't start if CPU > 99%
    max_memory_percent: float = 95.0  # Don't start if memory > 95%
    min_disk_space_mb: int = 100  # Minimum 100MB free
    
    # Per-request security (REAL authentication)
    rate_limit_per_ip: bool = True
    
    # Optional: Bootstrap only for critical deployments
    require_bootstrap_auth: bool = False  # Default OFF for flexibility
    bootstrap_token_hash: Optional[str] = None  # Only if above is True    


class SecurityLevel(Enum):
    """Deployment security levels"""
    DEVELOPMENT = "dev"      # No security, max flexibility
    STAGING = "staging"      # API keys only
    PRODUCTION = "prod"      # API keys + rate limiting
    HARDENED = "hardened"    # Everything + bootstrap token


class SecurityError(Exception):
    """Raised when a security check (validation, auth, rate limit, ...) fails."""
    pass


class AdminRole(Enum):
    """Admin roles: ADMIN (full access), OPERATOR (view only) and AUDITOR (audit log only)."""
    ADMIN = "admin"
    OPERATOR = "operator"  # Can view but not modify
    AUDITOR = "auditor"     # Can only view audit log
