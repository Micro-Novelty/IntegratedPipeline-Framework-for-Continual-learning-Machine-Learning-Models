"""AutoBatcherAutomation: automatic batch-size selection."""

from collections import deque
import logging
import threading
import time

logger = logging.getLogger(__name__)


# The AutoBatcherAutomation class manages the batching of incoming prediction requests to optimize processing efficiency. 
# It collects requests over a short time window or until a maximum batch size is reached, then processes them together through the pipeline. This allows for improved throughput while still providing timely responses to individual requests.
class AutoBatcherAutomation:
    """Collects single prediction requests and runs them through the pipeline in batches (up to `max_batch_size` or after `max_wait_ms`)."""
    def __init__(self, pipeline, max_batch_size=32, max_wait_ms=50):
        self.pipeline       = pipeline
        self.max_batch_size = max_batch_size
        self.max_wait_ms    = max_wait_ms

        self.request_queue  = deque()
        self.processing     = False
        self.results        = {}
        self.result_events  = {}   # per-request Event 
        self.next_id        = 0

        self._state_lock = threading.Lock()   # guards processing flag + next_id

    def add_request(self, text, callback=None):
        """Queue a text for batched prediction; returns a request id (optionally calls `callback` with the result)."""
        with self._state_lock:
            request_id    = self.next_id
            self.next_id += 1

            event = threading.Event()
            self.result_events[request_id] = event

            self.request_queue.append({
                'id'       : request_id,
                'text'     : text,
                'callback' : callback,
                'timestamp': time.time()
            })

            # check-and-set happens atomically under the same lock
            should_start = not self.processing
            if should_start:
                self.processing = True

        if should_start:
            self._start_processing()

        result = self.get_result(request_id)
        self.cleanup_stale()

        return request_id

    def _start_processing(self):
        thread = threading.Thread(target=self._process_batches, daemon=True)
        thread.start()

    def _process_batches(self):
        try:
            while True:
                with self._state_lock:
                    if not self.request_queue:
                        break

                time.sleep(self.max_wait_ms / 1000)

                batch = []
                with self._state_lock:
                    while self.request_queue and len(batch) < self.max_batch_size:
                        batch.append(self.request_queue.popleft())

                if batch:
                    # never let one bad batch kill the worker permanently here.
                    try:
                        self._process_batch(batch)
                    except Exception as e:
                        logger.error(f'[!] Batch processing failed: {e}')
                        # deliver the failure to every waiter in this batch
                        # instead of leaving them hanging forever
                        for req in batch:
                            self._deliver_result(req, None, error=e)
        finally:
            # always reset processing, even if something above
            # raised unexpectedly
            with self._state_lock:
                self.processing = False

            # need catch the case where requests arrived after the
            # while-loop's last empty check but before processing=False landed
            with self._state_lock:
                still_pending = bool(self.request_queue)
                if still_pending and not self.processing:
                    self.processing = True
                    restart = True
                else:
                    restart = False
            if restart:
                self._start_processing()

    def _process_batch(self, batch):
        texts   = [req['text'] for req in batch]
        results = self.pipeline.prediction_batch(texts)

        for i, req in enumerate(batch):
            result = results[i] if i < len(results) else None
            self._deliver_result(req, result)

    def _deliver_result(self, req, result, error=None):
        """Single delivery path — callback or stored result, always signals."""
        if req['callback']:
            try:
                req['callback'](result if error is None else error)
            except Exception as cb_err:
                logger.error(f'[!] Callback failed for request {req["id"]}: {cb_err}')
        else:
            with self._state_lock:
                self.results[req['id']] = result if error is None else None

        # signal the waiting event 
        event = self.result_events.get(req['id'])
        if event:
            event.set()

    def get_result(self, request_id, timeout=5):
        """Block up to `timeout` seconds for the result of `request_id` and clean up its bookkeeping; None if unknown/timed out."""
        event = self.result_events.get(request_id)
        if event is None:
            return None

        # blocks efficiently 
        signaled = event.wait(timeout=timeout)

        with self._state_lock:
            # always need clean up
            result = self.results.pop(request_id, None)
            self.result_events.pop(request_id, None)

        return result if signaled else None

    def cleanup_stale(self, max_age_seconds=300):
        """
        periodic sweep for requests that were never collected
        via get_result (e.g. caller crashed or forgot to call it).
        Call this periodically from a health check loop.
        """
        with self._state_lock:
            now = time.time()
            stale_ids = [
                rid for rid, event in self.result_events.items()
                if event.is_set()   # already delivered but never collected
            ]
            for rid in stale_ids:
                self.results.pop(rid, None)
                self.result_events.pop(rid, None)
            if stale_ids:
                logger.info(f'[=] Cleaned up {len(stale_ids)} stale results')
