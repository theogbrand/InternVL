"""
Optimal RAVEN Rollout Strategy for Azure OpenAI Rate Limits

Key Optimizations:
1. Token Bucket Rate Limiter (simpler than sliding window)
2. Request Queue with Priority Processing  
3. Optimal Batch Sizing (math-driven)
4. Simplified Pipeline (no complex overlapping)
5. Smart Token Estimation
"""

import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

@dataclass
class APIRequest:
    id: str
    messages: List[Dict]
    max_tokens: int
    temperature: float
    estimated_tokens: int
    priority: int = 0

class OptimalRateLimiter:
    """
    Token bucket rate limiter optimized for Azure limits:
    - 1K RPM = 16.67 requests/second
    - 1M TPM = 16,667 tokens/second
    
    Key insight: With 1K avg tokens per request, RPM is the bottleneck
    """
    
    def __init__(self, rpm_limit=950, tpm_limit=950000):
        # Use 95% of limits for safety buffer
        self.requests_per_second = rpm_limit / 60.0  # ~15.8 req/s
        self.tokens_per_second = tpm_limit / 60.0    # ~15,833 tokens/s
        
        # Token buckets (allow 2-second bursts)
        self.request_tokens = self.requests_per_second * 2
        self.data_tokens = self.tokens_per_second * 2
        
        self.max_request_tokens = self.request_tokens
        self.max_data_tokens = self.data_tokens
        
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def _refill_buckets(self):
        """Refill token buckets based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            self.request_tokens = min(
                self.max_request_tokens,
                self.request_tokens + self.requests_per_second * elapsed
            )
            self.data_tokens = min(
                self.max_data_tokens,
                self.data_tokens + self.tokens_per_second * elapsed
            )
            self.last_refill = now
    
    def acquire(self, estimated_tokens: int = 1000, timeout: float = 300) -> bool:
        """Acquire permission to make request (blocking with timeout)"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                self._refill_buckets()
                
                if self.request_tokens >= 1.0 and self.data_tokens >= estimated_tokens:
                    # Consume tokens
                    self.request_tokens -= 1.0
                    self.data_tokens -= estimated_tokens
                    return True
                
                # Calculate optimal wait time
                wait_for_request = max(0, (1.0 - self.request_tokens) / self.requests_per_second)
                wait_for_tokens = max(0, (estimated_tokens - self.data_tokens) / self.tokens_per_second)
                wait_time = min(1.0, max(wait_for_request, wait_for_tokens))
            
            time.sleep(wait_time)
        
        return False  # Timeout

class RequestProcessor:
    """Optimal request processor with controlled concurrency"""
    
    def __init__(self, rate_limiter: OptimalRateLimiter, max_concurrent: int = 50):
        self.rate_limiter = rate_limiter
        self.max_concurrent = max_concurrent
        self.request_queue = queue.Queue()
        self.results = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.active_count = 0
        self.lock = threading.Lock()
    
    def submit_batch(self, requests: List[APIRequest]) -> List[str]:
        """Submit batch and wait for all results"""
        # Submit all requests
        for req in requests:
            self.request_queue.put(req)
        
        # Process requests with rate limiting
        futures = []
        for req in requests:
            # Wait for rate limit clearance
            if self.rate_limiter.acquire(req.estimated_tokens):
                future = self.executor.submit(self._execute_request, req)
                futures.append((req.id, future))
        
        # Collect results
        results = {}
        for req_id, future in futures:
            try:
                content = future.result(timeout=300)  # 5 min timeout
                results[req_id] = content
            except Exception:
                results[req_id] = ""
        
        # Return in original order
        return [results.get(req.id, "") for req in requests]
    
    def _execute_request(self, request: APIRequest) -> str:
        """Execute single API request"""
        try:
            # Simulate API call (replace with actual Azure OpenAI call)
            time.sleep(0.1)  # Simulate network latency
            return f"Response for {request.id}"
        except Exception:
            return ""

def calculate_optimal_batch_size(
    rpm_limit: int = 950,
    avg_tokens_per_request: int = 1000,
    num_return_sequences: int = 2,
    num_mc_sequences: int = 8,
    avg_steps_per_response: int = 10
) -> Dict[str, int]:
    """
    Calculate optimal batch sizes based on rate limits
    
    Math:
    - RPM is bottleneck: 950 requests/minute = ~15.8 req/s
    - For sustainable throughput: aim for ~12 req/s (75% utilization)
    - Safety buffer prevents rate limit violations
    """
    
    # Target sustainable rate (75% of limit)
    target_rps = (rpm_limit * 0.75) / 60  # ~11.9 req/s
    
    # Calculate requests per sample
    initial_requests_per_sample = num_return_sequences
    mc_requests_per_sample = num_return_sequences * avg_steps_per_response * num_mc_sequences
    total_requests_per_sample = initial_requests_per_sample + mc_requests_per_sample
    
    # Optimal batch size for initial rollouts
    initial_batch_size = max(1, int(target_rps * 60 / initial_requests_per_sample))  # requests per minute
    
    # Optimal batch size for MC evaluation  
    mc_batch_size = max(1, int(target_rps * 10 / mc_requests_per_sample))  # 10-second batches
    
    return {
        'initial_batch_size': min(initial_batch_size, 100),  # Cap at 100 for memory
        'mc_batch_size': min(mc_batch_size, 20),             # Cap at 20 for MC
        'requests_per_sample': total_requests_per_sample,
        'target_rps': target_rps,
        'estimated_samples_per_hour': int(3600 / (total_requests_per_sample / target_rps))
    }

def optimal_strategy_recommendations():
    """
    Return the optimal strategy recommendations
    """
    
    # Calculate optimal parameters
    optimal_params = calculate_optimal_batch_size()
    
    strategy = {
        # Core rate limiting
        'rate_limiter': 'Token bucket (simpler than sliding window)',
        'rpm_limit': 950,  # 95% of 1K limit
        'tpm_limit': 950000,  # 95% of 1M limit
        
        # Optimal batch sizes
        'initial_batch_size': optimal_params['initial_batch_size'],
        'mc_batch_size': optimal_params['mc_batch_size'],
        'num_return_sequences': 2,  # Keep as is
        'num_mc_sequences': 6,      # Reduced from 8 for better throughput
        
        # Concurrency settings
        'max_workers': 50,          # Optimal for I/O bound tasks
        'request_timeout': 300,     # 5 minutes per request
        
        # Processing strategy
        'pipeline_approach': 'Sequential batches (not overlapping)',
        'early_stopping': True,     # Stop on incorrect steps
        'incremental_saving': True, # Save after each batch
        
        # Expected performance
        'estimated_throughput': f"{optimal_params['estimated_samples_per_hour']} samples/hour",
        'memory_usage': 'Low (no large request queues)',
        'complexity': 'Low (simple, debuggable)'
    }
    
    return strategy

def print_strategy_analysis():
    """Print detailed strategy analysis"""
    
    print("=== OPTIMAL AZURE OPENAI RATE LIMITING STRATEGY ===\n")
    
    strategy = optimal_strategy_recommendations()
    optimal_params = calculate_optimal_batch_size()
    
    print("1. CORE PROBLEM ANALYSIS:")
    print(f"   • Current: batch_size=32 × num_sequences=2 × mc_sequences=8 = 512+ requests/step")
    print(f"   • Problem: Far exceeds 1K RPM limit, causes complex scheduling")
    print(f"   • Solution: Math-driven batch sizing\n")
    
    print("2. OPTIMAL PARAMETERS:")
    for key, value in strategy.items():
        if isinstance(value, str) and len(value) > 50:
            print(f"   • {key}: {value[:50]}...")
        else:
            print(f"   • {key}: {value}")
    print()
    
    print("3. MATHEMATICAL JUSTIFICATION:")
    print(f"   • Target rate: {optimal_params['target_rps']:.1f} requests/second (75% of limit)")
    print(f"   • Requests per sample: {optimal_params['requests_per_sample']}")
    print(f"   • Initial batch size: {optimal_params['initial_batch_size']} (sustainable)")
    print(f"   • MC batch size: {optimal_params['mc_batch_size']} (allows 10s processing)")
    print(f"   • Expected throughput: {optimal_params['estimated_samples_per_hour']} samples/hour\n")
    
    print("4. KEY IMPROVEMENTS OVER CURRENT:")
    improvements = [
        "Replace sliding window with token bucket (simpler, more predictable)",
        "Remove complex pipeline overlapping (source of bugs)",
        "Use math-driven batch sizing instead of guessing",
        "Implement proper request queuing with timeouts", 
        "Add comprehensive error handling and recovery",
        "Provide clear performance metrics and debugging"
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f"   {i}. {improvement}")
    
    print("\n5. IMPLEMENTATION PRIORITY:")
    print("   1. Implement token bucket rate limiter (simple, reliable)")
    print("   2. Replace parallel pipeline with sequential batched processing")  
    print("   3. Use calculated optimal batch sizes")
    print("   4. Add proper timeout and error handling")
    print("   5. Test with small dataset to validate throughput")

if __name__ == "__main__":
    print_strategy_analysis() 