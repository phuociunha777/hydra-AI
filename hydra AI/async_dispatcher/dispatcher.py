"""
Tầng kiểm thử bất đồng bộ (Asynchronous Verification Dispatcher)
Sử dụng vòng lặp I/O phi chặn (Non-blocking I/O Event Loop)
để kiểm thử đồng thời các mẫu thuộc Candidate Pool đã vượt qua ngưỡng phân loại của mô hình.
"""

import asyncio
import time
from typing import List, Tuple, Dict, Any, Optional
from async_dispatcher.rate_limiter import MockTargetService

class AsyncVerificationDispatcher:
    """
    Tầng điều phối kiểm thử bất đồng bộ:
    - Tiếp nhận các ứng viên đã được Tầng phân tích lọc vào Candidate Pool
    - Sử dụng mô hình xử lý phi chặn (Non-blocking Coroutines)
    - Đảm bảo tính toán đồng thời cao mà không gây nghẽn tài nguyên CPU/Network
    """
    def __init__(self, target_service: MockTargetService, max_concurrency: int = 20):
        self.target_service = target_service
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.dispatched_count = 0
        self.success_result: Optional[Dict[str, Any]] = None

    async def _worker_verify(self, candidate_info: Tuple[str, float, str]) -> Dict[str, Any]:
        """Worker thực hiện gọi hàm verify với điều phối Semaphore."""
        candidate_str, score, cluster_name = candidate_info
        
        async with self.semaphore:
            self.dispatched_count += 1
            start_t = time.perf_counter()
            resp = await self.target_service.verify(candidate_str)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            result = {
                "candidate": candidate_str,
                "score": score,
                "cluster": cluster_name,
                "status_code": resp["status_code"],
                "matched": resp["matched"],
                "latency_ms": round(elapsed_ms, 3)
            }

            if resp["matched"]:
                self.success_result = result
            return result

    async def dispatch_candidate_pool(self, candidate_pool: List[Tuple[str, float, str]]) -> Dict[str, Any]:
        """
        Điều phối kiểm thử đồng thời các ứng viên vào dịch vụ mô phỏng (Mock Service) thông qua vòng lặp phi chặn.
        """
        if not candidate_pool:
            return {
                "total_sent": 0,
                "matched": False,
                "duration_sec": 0.0,
                "results": []
            }

        start_time = time.perf_counter()
        self.success_result = None

        # Khởi tạo các tác vụ phi chặn
        tasks = [asyncio.create_task(self._worker_verify(item)) for item in candidate_pool]
        
        # Chờ tất cả hoàn thành hoặc dừng ngay khi có kết quả match
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        status_200 = sum(1 for r in results if r["status_code"] == 200)
        status_429 = sum(1 for r in results if r["status_code"] == 429)
        status_401 = sum(1 for r in results if r["status_code"] == 401)

        summary = {
            "total_sent": len(results),
            "status_200_count": status_200,
            "status_429_throttled": status_429,
            "status_401_unauthorized": status_401,
            "matched_found": self.success_result is not None,
            "match_details": self.success_result,
            "duration_sec": round(total_time, 4),
            "throughput_req_per_sec": round(len(results) / max(total_time, 1e-6), 2),
            "results": results
        }
        return summary
