import os, time, uuid, yaml, pandas as pd
from pathlib import Path

class SessionLogger:
    def __init__(self, cfg):
        self.session_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.out_dir = Path(cfg["session"]["out_dir"]) / self.session_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        # dump snapshot of config + screen meta
        snap = dict(cfg)
        snap["session"]["session_id"] = self.session_id
        (self.out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snap))

    def add(self, row: dict):
        row["session_id"] = self.session_id
        self.rows.append(row)

    def flush(self):
        if not self.rows: return
        df = pd.DataFrame(self.rows)
        df.to_parquet(self.out_dir / "frames.parquet", index=False)
        # lightweight session info (for quick scan)
        df_info = df[["session_id","t_mono","frame_id"]].tail(1)
        df_info.to_parquet(self.out_dir / "session_info.parquet", index=False)
        self.rows.clear()

    def video_path(self):
        return str(self.out_dir / "calib.mp4")
