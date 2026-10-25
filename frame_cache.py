"""进程内帧图像内存缓存。"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 默认上限 1.5 GiB（约 250 帧 1080p BGR / 1000 帧 720p BGR）
DEFAULT_MAX_BYTES = 1_500 * 1024 * 1024


class FrameCache:
    """线程安全的 LRU 帧图像缓存（值为只读 BGR ndarray）。"""

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES):
        self.max_bytes = max(0, int(max_bytes))
        self._lock = threading.RLock()
        self._store: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ---- 内部 ----
    @staticmethod
    def _key(path: str) -> str:
        try:
            return os.path.abspath(path)
        except Exception:
            return path

    @staticmethod
    def _freeze(arr: np.ndarray) -> np.ndarray:
        if arr.flags.writeable:
            arr.flags.writeable = False
        return arr

    def _evict_locked(self) -> None:
        while self._bytes > self.max_bytes and self._store:
            _, old = self._store.popitem(last=False)
            self._bytes -= int(old.nbytes)
            self._evictions += 1

    # ---- 公共 ----
    def get(self, path: str) -> Optional[np.ndarray]:
        """命中返回只读 ndarray；读盘失败返回 None（同 cv2.imread）。"""
        if not path:
            return None
        if self.max_bytes <= 0:
            return cv2.imread(path, cv2.IMREAD_COLOR)

        key = self._key(path)
        with self._lock:
            hit = self._store.get(key)
            if hit is not None:
                self._store.move_to_end(key)
                self._hits += 1
                return hit

        # 读盘放在锁外，避免长 IO 阻塞其他线程
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            with self._lock:
                self._misses += 1
            return None
        img = self._freeze(img)

        with self._lock:
            # 二次检查：其他线程可能刚放进来
            exist = self._store.get(key)
            if exist is not None:
                self._store.move_to_end(key)
                self._hits += 1
                return exist
            self._store[key] = img
            self._bytes += int(img.nbytes)
            self._misses += 1
            self._evict_locked()
        return img

    def preload(self, paths: Iterable[str],
                max_items: Optional[int] = None) -> int:
        """占满 90% 预算即停止，避免把缓存挤爆。"""
        if self.max_bytes <= 0:
            return 0
        items = list(paths)
        if max_items is not None:
            items = items[:max_items]
        budget = int(self.max_bytes * 0.9)
        loaded = 0
        t0 = time.time()
        for p in items:
            if self._bytes >= budget:
                logger.info(
                    "[frame_cache] 预加载提前停止：已达预算 %.1f/%.1f MiB",
                    self._bytes / 1048576, self.max_bytes / 1048576,
                )
                break
            if self.get(p) is not None:
                loaded += 1
        logger.info(
            "[frame_cache] 预加载 %d/%d 帧，占用 %.1f MiB / %.1f MiB，耗时 %.1fs",
            loaded, len(items),
            self._bytes / 1048576, self.max_bytes / 1048576,
            time.time() - t0,
        )
        return loaded

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._bytes = 0

    def stats(self) -> Dict[str, float]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "items": len(self._store),
                "bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": (self._hits / total) if total else 0.0,
            }


# ---- 进程级默认缓存 ----
_DEFAULT_CACHE: Optional[FrameCache] = None
_DEFAULT_LOCK = threading.Lock()


def configure(max_bytes: Optional[int] = None,
              preload: Optional[Iterable[str]] = None,
              preload_limit: Optional[int] = None) -> FrameCache:
    """max_bytes=None 保留现有实例；<=0 表示禁用缓存。"""
    global _DEFAULT_CACHE
    with _DEFAULT_LOCK:
        if _DEFAULT_CACHE is None or max_bytes is not None:
            _DEFAULT_CACHE = FrameCache(
                max_bytes if max_bytes is not None else DEFAULT_MAX_BYTES
            )
    cache = _DEFAULT_CACHE
    if preload:
        cache.preload(preload, max_items=preload_limit)
    return cache


def get_default_cache() -> FrameCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_CACHE is None:
                _DEFAULT_CACHE = FrameCache()
    return _DEFAULT_CACHE


def get_image(path: str) -> Optional[np.ndarray]:
    return get_default_cache().get(path)


def preload_images(paths: Iterable[str],
                   max_items: Optional[int] = None) -> int:
    return get_default_cache().preload(paths, max_items=max_items)


def clear_cache() -> None:
    get_default_cache().clear()


def cache_stats() -> Dict[str, float]:
    return get_default_cache().stats()