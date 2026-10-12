"""Read-only access to the official Video-3D-LLM data assets.

Nothing here writes into the reference repository.  The frame index is built
from the official ``embodiedscan_infos_{train,val,test}.pkl`` files (the exact
source used by ``llava.video_utils.VideoProcessor``) and cached as JSON so that
the WebUI frame picker and the actual inference always agree on frame indices.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from . import config

MEDIA_ROOT = os.path.join(config.REPO_ROOT, "data")

_FRAME_INDEX_CACHE = os.path.join(config.WEBUI_ROOT, "results", "frame_index.json")


def default_frame_indices(n_frames: int, k: int) -> List[int]:
    """Exactly ``VideoProcessor.sample_frame_files``' uniform sampling."""
    if n_frames <= 0:
        return []
    k = max(1, min(int(k), n_frames))
    return np.linspace(0, n_frames - 1, k, dtype=int).tolist()


def to_abs(rel_path: str) -> str:
    """``scannet/posed_images/...`` -> absolute path inside the repo."""
    return os.path.join(MEDIA_ROOT, rel_path)


class DataStore:
    """Lazy, thread-safe cache over the official question files / frame index."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._questions: Dict[str, List[dict]] = {}
        self._frame_index: Optional[Dict[str, List[str]]] = None
        self._id2idx: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # questions
    # ------------------------------------------------------------------
    def questions(self, key: str) -> List[dict]:
        with self._lock:
            if key not in self._questions:
                path = config.abs_question_file(key)
                with open(path, "r", encoding="utf-8") as fh:
                    self._questions[key] = json.load(fh)
            return self._questions[key]

    def get(self, key: str, index: int) -> dict:
        data = self.questions(key)
        if index < 0 or index >= len(data):
            raise IndexError(f"{key} question index out of range: {index} (n={len(data)})")
        return data[index]

    def count(self, key: str) -> int:
        return len(self.questions(key))

    def summary(self, key: str, index: int) -> dict:
        rec = self.get(key, index)
        conv = rec.get("conversations") or []
        question = conv[0]["value"] if conv else ""
        return {
            "index": index,
            "id": rec.get("id"),
            "scene": rec.get("video"),
            "question": question,
            "question_type": (rec.get("metadata") or {}).get("question_type"),
            "gt": self.gt_display(key, rec),
        }

    def gt_display(self, key: str, rec: dict) -> Any:
        """Ground-truth in the shape the official evaluation script consumes."""
        meta = rec.get("metadata") or {}
        if key in ("scanqa", "sqa3d"):
            return meta.get("answers") or [(rec.get("conversations") or [{}])[-1].get("value")]
        if key == "scan2cap":
            return rec.get("annotations") or [(rec.get("conversations") or [{}])[-1].get("value")]
        if key == "scanrefer":
            return rec.get("box")
        if key == "multi3drefer":
            return rec.get("box")
        return meta.get("answers")

    def search(self, key: str, needle: str, limit: int = 200) -> List[dict]:
        needle = (needle or "").strip().lower()
        if not needle:
            return []
        out: List[dict] = []
        for i, rec in enumerate(self.questions(key)):
            conv = rec.get("conversations") or []
            hay = " ".join([str(rec.get("id")), str(rec.get("video")), conv[0]["value"] if conv else ""]).lower()
            if needle in hay:
                out.append(self.summary(key, i))
                if len(out) >= limit:
                    break
        return out

    # ------------------------------------------------------------------
    # frames
    # ------------------------------------------------------------------
    def _build_frame_index(self) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        for split in ("train", "val", "test"):
            path = os.path.join(MEDIA_ROOT, "embodiedscan", f"embodiedscan_infos_{split}.pkl")
            if not os.path.exists(path):
                continue
            with open(path, "rb") as fh:
                for item in pickle.load(fh)["data_list"]:
                    sample_idx = item.get("sample_idx", "")
                    if not sample_idx.startswith("scannet"):
                        continue
                    # later splits win, mirrors VideoProcessor.__init__
                    index[sample_idx] = [img["img_path"] for img in item["images"]]
        return index

    def frame_index(self) -> Dict[str, List[str]]:
        with self._lock:
            if self._frame_index is None:
                if os.path.exists(_FRAME_INDEX_CACHE):
                    try:
                        with open(_FRAME_INDEX_CACHE, "r", encoding="utf-8") as fh:
                            self._frame_index = json.load(fh)
                    except Exception:
                        self._frame_index = None
                if self._frame_index is None:
                    self._frame_index = self._build_frame_index()
                    tmp = _FRAME_INDEX_CACHE + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(self._frame_index, fh)
                    os.replace(tmp, _FRAME_INDEX_CACHE)
            return self._frame_index

    def scene_frames(self, scene: str) -> List[str]:
        """Relative jpg paths, in the official (metadata) order."""
        frames = self.frame_index().get(scene)
        if frames is None:
            # fall back to the filesystem for scenes missing from embodiedscan
            folder = os.path.join(MEDIA_ROOT, "scannet", "posed_images", scene.split("/")[-1])
            frames = [
                os.path.join("scannet", "posed_images", scene.split("/")[-1], os.path.basename(p))
                for p in sorted(glob.glob(os.path.join(folder, "*.jpg")))
            ]
        return frames

    def frame_descriptors(self, scene: str, k: int = config.DEFAULT_MAX_FRAME_NUM) -> List[dict]:
        frames = self.scene_frames(scene)
        selected = set(default_frame_indices(len(frames), k))
        out = []
        for i, rel in enumerate(frames):
            name = os.path.basename(rel)
            out.append(
                {
                    "i": i,
                    "name": name,
                    "rgb": _url(scene, name, "jpg"),
                    "depth": _url(scene, name, "png"),
                    "default_selected": i in selected,
                    # frame timestamps follow the embodiedscan 3 fps convention
                    "time": round(i / 3.0, 2),
                }
            )
        return out

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def scene_ids(self) -> List[str]:
        return sorted(self.frame_index().keys())


def _url(scene: str, name: str, ext: str) -> str:
    scene_name = scene.split("/")[-1]
    if ext == "png":
        name = os.path.splitext(name)[0] + ".png"
    return f"/api/media/frame?scene={scene_name}&name={name}"


STORE = DataStore()
