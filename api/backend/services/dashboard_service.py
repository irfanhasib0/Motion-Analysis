import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional

import psutil


logger = logging.getLogger(__name__)


class DashboardService:
    def __init__(self, camera_service, window_seconds: int = 300, sample_interval_seconds: int = 5):
        self.camera_service = camera_service
        self.window_seconds = window_seconds
        self.sample_interval_seconds = sample_interval_seconds

        self._samples: Deque[Dict] = deque()
        self._lock = threading.Lock()

        self._last_disk_counters = None
        self._last_process_io = None
        self._last_sample_time = None

        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread.start()

    def _find_start_server_process(self) -> psutil.Process:
        try:
            candidates = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                cmdline = proc.info.get('cmdline') or []
                if any('start_server.py' in str(part) for part in cmdline):
                    candidates.append(proc)

            if candidates:
                return sorted(candidates, key=lambda proc: proc.info.get('create_time') or 0)[0]
        except Exception as error:
            logger.warning(f"Unable to resolve start_server process: {error}")

        return psutil.Process(os.getpid())

    def _sample_loop(self):
        while True:
            try:
                self._collect_sample()
            except Exception as error:
                logger.warning(f"Dashboard sampler failed: {error}")
            time.sleep(self.sample_interval_seconds)

    def _collect_sample(self):
        now = time.time()
        memory = psutil.virtual_memory()
        disk_counters = psutil.disk_io_counters()

        process = self._find_start_server_process()
        process_cpu = process.cpu_percent(interval=0.05)
        process_memory_percent = process.memory_percent()
        process_memory_mb = process.memory_info().rss / (1024 ** 2)

        process_io = None
        process_read_mb_total = 0.0
        process_write_mb_total = 0.0
        try:
            process_io = process.io_counters()
            process_read_mb_total = process_io.read_bytes / (1024 ** 2)
            process_write_mb_total = process_io.write_bytes / (1024 ** 2)
        except Exception:
            process_io = None

        disk_read_mb_total = (disk_counters.read_bytes / (1024 ** 2)) if disk_counters else 0.0
        disk_write_mb_total = (disk_counters.write_bytes / (1024 ** 2)) if disk_counters else 0.0

        disk_read_mb_s = 0.0
        disk_write_mb_s = 0.0
        process_read_mb_s = 0.0
        process_write_mb_s = 0.0

        if self._last_sample_time is not None:
            elapsed = max(1e-6, now - self._last_sample_time)

            if self._last_disk_counters is not None and disk_counters is not None:
                delta_disk_read = max(0.0, disk_counters.read_bytes - self._last_disk_counters.read_bytes)
                delta_disk_write = max(0.0, disk_counters.write_bytes - self._last_disk_counters.write_bytes)
                disk_read_mb_s = (delta_disk_read / (1024 ** 2)) / elapsed
                disk_write_mb_s = (delta_disk_write / (1024 ** 2)) / elapsed

            if self._last_process_io is not None and process_io is not None:
                delta_proc_read = max(0.0, process_io.read_bytes - self._last_process_io.read_bytes)
                delta_proc_write = max(0.0, process_io.write_bytes - self._last_process_io.write_bytes)
                process_read_mb_s = (delta_proc_read / (1024 ** 2)) / elapsed
                process_write_mb_s = (delta_proc_write / (1024 ** 2)) / elapsed

        sample = {
            'timestamp': now,
            'cpu_usage': psutil.cpu_percent(interval=0.05),
            'memory_usage': memory.percent,
            'disk_io_read_mb_s': disk_read_mb_s,
            'disk_io_write_mb_s': disk_write_mb_s,
            'disk_io_read_mb_total': disk_read_mb_total,
            'disk_io_write_mb_total': disk_write_mb_total,
            'process_name': process.name() or 'start_server',
            'process_pid': process.pid,
            'process_cpu_percent': process_cpu,
            'process_memory_percent': process_memory_percent,
            'process_memory_mb': process_memory_mb,
            'process_disk_io_read_mb_s': process_read_mb_s,
            'process_disk_io_write_mb_s': process_write_mb_s,
            'process_disk_io_read_mb_total': process_read_mb_total,
            'process_disk_io_write_mb_total': process_write_mb_total,
        }

        with self._lock:
            self._samples.append(sample)
            cutoff = now - self.window_seconds
            while self._samples and self._samples[0]['timestamp'] < cutoff:
                self._samples.popleft()

        self._last_sample_time = now
        self._last_disk_counters = disk_counters
        self._last_process_io = process_io

    @staticmethod
    def _average(samples, key: str) -> float:
        if not samples:
            return 0.0
        values = [float(sample.get(key, 0.0) or 0.0) for sample in samples]
        return sum(values) / len(values)

    @staticmethod
    def _get_directory_size_bytes(directory_path: str) -> int:
        total_size = 0
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                except OSError:
                    continue
        return total_size

    def get_system_info(self) -> Dict:
        with self._lock:
            if not self._samples:
                self._collect_sample()
            samples = list(self._samples)

        latest = samples[-1] if samples else {}

        recording_disk = psutil.disk_usage(self.camera_service.recordings_dir)
        overall_disk = psutil.disk_usage(os.path.abspath(os.sep))
        recording_dir_size_bytes = self._get_directory_size_bytes(self.camera_service.recordings_dir)
        uptime_delta = datetime.now() - self.camera_service.start_time
        total_uptime_seconds = max(0, int(uptime_delta.total_seconds()))

        processing_active = {
            camera.id: camera.processing_type
            for camera in self.camera_service.get_cameras()
            if camera.processing_active
        }

        return {
            'uptime': {
                'text': str(uptime_delta).split('.')[0],
                'days': total_uptime_seconds // 86400,
                'hours': (total_uptime_seconds % 86400) // 3600,
                'minutes': (total_uptime_seconds % 3600) // 60,
                'seconds': total_uptime_seconds % 60,
            },
            'cpu_usage': latest.get('cpu_usage', 0.0),
            'memory_usage': latest.get('memory_usage', 0.0),
            'disk_usage': {
                'percent_used': recording_disk.percent,
                'used_gb': recording_disk.used / (1024 ** 3),
                'total_gb': recording_disk.total / (1024 ** 3),
                'free_gb': recording_disk.free / (1024 ** 3),
                'io_read_mb': latest.get('disk_io_read_mb_total', 0.0),
                'io_write_mb': latest.get('disk_io_write_mb_total', 0.0),
                'io_read_mb_s': latest.get('disk_io_read_mb_s', 0.0),
                'io_write_mb_s': latest.get('disk_io_write_mb_s', 0.0),
            },
            'disk_size': {
                'overall_total_gb': overall_disk.total / (1024 ** 3),
                'overall_used_gb': overall_disk.used / (1024 ** 3),
                'overall_free_gb': overall_disk.free / (1024 ** 3),
                'recording_dir_size_gb': recording_dir_size_bytes / (1024 ** 3),
            },
            'process_usage': {
                'name': latest.get('process_name', 'start_server'),
                'pid': latest.get('process_pid'),
                'cpu_percent': latest.get('process_cpu_percent', 0.0),
                'memory_percent': latest.get('process_memory_percent', 0.0),
                'memory_mb': latest.get('process_memory_mb', 0.0),
                'disk_io_read_mb': latest.get('process_disk_io_read_mb_total', 0.0),
                'disk_io_write_mb': latest.get('process_disk_io_write_mb_total', 0.0),
                'disk_io_read_mb_s': latest.get('process_disk_io_read_mb_s', 0.0),
                'disk_io_write_mb_s': latest.get('process_disk_io_write_mb_s', 0.0),
                'recording_dir_size_gb': recording_dir_size_bytes / (1024 ** 3),
            },
            'averages_5m': {
                'window_seconds': self.window_seconds,
                'sample_count': len(samples),
                'cpu_usage': self._average(samples, 'cpu_usage'),
                'memory_usage': self._average(samples, 'memory_usage'),
                'disk_io_read_mb_s': self._average(samples, 'disk_io_read_mb_s'),
                'disk_io_write_mb_s': self._average(samples, 'disk_io_write_mb_s'),
                'process_cpu_percent': self._average(samples, 'process_cpu_percent'),
                'process_memory_percent': self._average(samples, 'process_memory_percent'),
                'process_disk_io_read_mb_s': self._average(samples, 'process_disk_io_read_mb_s'),
                'process_disk_io_write_mb_s': self._average(samples, 'process_disk_io_write_mb_s'),
            },
            'processing_active': processing_active,
            'active_recordings': len(self.camera_service.active_recordings),
            'total_cameras': len(self.camera_service.get_cameras()),
            'total_recordings': len(self.camera_service.get_recordings()),
        }
