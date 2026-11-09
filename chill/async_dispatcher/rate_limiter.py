"""
Mô phỏng cơ chế phòng vệ và giới hạn tần suất (Rate Limiter)
Bao gồm:
- Token Bucket Rate Limiter
- Sliding Window Log Rate Limiter
Dùng để đo lường khả năng phòng vệ và benchmark độ trễ.
"""

import time
import asyncio
from typing import Tuple, Dict

class TokenBucketRateLimiter:
    """
    Thuật toán Token Bucket Rate Limiter:
    - capacity: Số lượng token tối đa trong thùng
    - refill_rate: Số token được nạp lại mỗi giây (tokens/sec)
    """
    def __init__(self, capacity: int = 50, refill_rate: float = 20.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

    async def acquire(self, cost: int = 1) -> bool:
        """Thử lấy token không chặn (non-blocking). Trả về True nếu thành công, False nếu bị chặn (429)."""
        async with self._lock:
            self._refill()
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

class MockTargetService:
    """
    Dịch vụ đích mô phỏng (Target Server) có cơ chế Rate Limiting:
    - Nhận chuỗi ứng viên
    - Nếu bị vượt ngưỡng rate limit -> Trả về HTTP 429 Too Many Requests
    - Nếu khớp mật khẩu bí mật -> Trả về HTTP 200 OK
    - Nếu sai mật khẩu -> Trả về HTTP 401 Unauthorized
    """
    def __init__(self, secret_target: str, rate_limiter: TokenBucketRateLimiter, simulated_network_latency_ms: float = 2.0):
        self.secret_target = secret_target
        self.rate_limiter = rate_limiter
        self.latency_sec = simulated_network_latency_ms / 1000.0
        self.total_requests = 0
        self.passed_requests = 0
        self.throttled_requests = 0
        self.match_found = False

    async def verify(self, candidate: str) -> Dict[str, any]:
        """Xử lý xác thực bất đồng bộ phi chặn."""
        self.total_requests += 1

        # Kiểm tra rate limit
        allowed = await self.rate_limiter.acquire(1)
        if not allowed:
            self.throttled_requests += 1
            return {
                "status_code": 429,
                "candidate": candidate,
                "message": "Too Many Requests (Rate limit exceeded)",
                "matched": False
            }

        self.passed_requests += 1

        # Mô phỏng độ trễ I/O phi chặn
        if self.latency_sec > 0:
            await asyncio.sleep(self.latency_sec)

        matched = (candidate == self.secret_target)
        if matched:
            self.match_found = True
            return {
                "status_code": 200,
                "candidate": candidate,
                "message": "Authentication Success",
                "matched": True
            }
        else:
            return {
                "status_code": 401,
                "candidate": candidate,
                "message": "Authentication Failed",
                "matched": False
            }
