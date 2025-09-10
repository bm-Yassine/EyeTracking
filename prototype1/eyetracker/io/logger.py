from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .schemas import FrameRow, EventRow

class ParquetLogger:
    def __init__(self, path: str, schema: pa.schema, flush_every: int = 120):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = schema
        self.flush_every = flush_every
        self._rows: List[dict] = []
        self._writer: Optional[pq.ParquetWriter] = None

    def _ensure_writer(self):
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")

    def write(self, row_dict: dict):
        self._rows.append(row_dict)
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._rows:
            return
        self._ensure_writer()
        tbl = pa.Table.from_pylist(self._rows, schema=self.schema)
        self._writer.write_table(tbl)
        self._rows.clear()

    def close(self):
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

def frames_logger(path: str) -> ParquetLogger:
    schema = pa.schema([
        ("t_mono", pa.float64()),
        ("frame_id", pa.int64()),
        ("screen_w", pa.int32()),
        ("screen_h", pa.int32()),
        ("cam_w", pa.int32()),
        ("cam_h", pa.int32()),
        ("head_yaw_deg", pa.float32()),
        ("head_pitch_deg", pa.float32()),
        ("head_roll_deg", pa.float32()),
        ("head_dist_mm", pa.float32()),
        ("left_yaw", pa.float32()),
        ("left_pitch", pa.float32()),
        ("right_yaw", pa.float32()),
        ("right_pitch", pa.float32()),
        ("face_present", pa.bool_()),
        ("blink", pa.bool_()),
        ("landmark_score", pa.float32()),
        ("target_x", pa.float32()),
        ("target_y", pa.float32()),
    ])
    return ParquetLogger(path, schema)

def events_logger(path: str) -> ParquetLogger:
    schema = pa.schema([
        ("t_mono", pa.float64()),
        ("idx", pa.int32()),
        ("target_x", pa.int32()),
        ("target_y", pa.int32()),
        ("click_x", pa.int32()),
        ("click_y", pa.int32()),
    ])
    return ParquetLogger(path, schema)
