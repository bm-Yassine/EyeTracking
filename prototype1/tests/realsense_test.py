#!/usr/bin/env python3
"""
realsense_d435i_probe.py
Small script to enumerate and sample D435i: device/sensors/options, intrinsics/extrinsics,
and a few frames (depth/color/IMU). Saves a JSON report next to the script.
"""

import json
import time
from collections import defaultdict

import pyrealsense2 as rs
import numpy as np


def as_list(m):  # helper for serializing rs structs
    return [float(x) for x in m]

def option_name(o):
    try:
        return str(o).split('.')[-1]
    except Exception:
        return str(o)

def profile_dict(p):
    d = {"stream": str(p.stream_type()).split('.')[-1],
         "format": str(p.format()).split('.')[-1],
         "fps": int(p.fps())}
    # Video profile fields
    try:
        vp = p.as_video_stream_profile()
        d.update({"width": vp.width(), "height": vp.height()})
    except Exception:
        pass
    return d

def get_video_intrinsics(vp):
    try:
        intr = vp.get_intrinsics()
        return {
            "width": intr.width,
            "height": intr.height,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "fx": intr.fx,
            "fy": intr.fy,
            "model": str(intr.model).split('.')[-1],
            "coeffs": list(intr.coeffs),
        }
    except Exception:
        return None

def get_motion_intrinsics(mp):
    try:
        mi = mp.get_motion_intrinsics()
        return {
            "data": [as_list(row) for row in mi.data],  # 3x4
            "noise_variances": as_list(mi.noise_variances),   # 3
            "bias_variances": as_list(mi.bias_variances),     # 3
        }
    except Exception:
        return None

def main():
    report = {"timestamp": time.time(), "device": {}, "sensors": []}

    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No RealSense device found. Plug in the D435i and try again.")
        return

    dev = ctx.devices[0]
    # Device info
    dev_info_keys = [
        rs.camera_info.name, rs.camera_info.serial_number,
        rs.camera_info.firmware_version, rs.camera_info.product_line,
        rs.camera_info.camera_locked, rs.camera_info.usb_type_descriptor
    ]
    device_info = {}
    for k in dev_info_keys:
        try:
            device_info[str(k).split('.')[-1]] = dev.get_info(k)
        except Exception:
            pass
    report["device"] = device_info
    print("Device:", device_info.get("name"), "| S/N:", device_info.get("serial_number"),
          "| FW:", device_info.get("firmware_version"))

    # Sensors, profiles, options
    for s in dev.query_sensors():
        s_info = {
            "name": s.get_info(rs.camera_info.name) if s.supports(rs.camera_info.name) else "Unknown",
            "profiles": [],
            "options": {},
            "depth_scale": None,
        }

        # Supported profiles
        try:
            for p in s.get_stream_profiles():
                s_info["profiles"].append(profile_dict(p))
        except Exception:
            pass

        # Options (current + range)
        try:
            for o in s.get_supported_options():
                try:
                    rng = s.get_option_range(o)
                    cur = s.get_option(o)
                    s_info["options"][option_name(o)] = {
                        "current": float(cur),
                        "min": float(rng.min), "max": float(rng.max),
                        "step": float(rng.step), "default": float(rng.default)
                    }
                except Exception:
                    pass
        except Exception:
            pass

        # Depth scale if this sensor is a depth sensor
        try:
            ds = s.as_depth_sensor()
            if ds:
                s_info["depth_scale"] = float(ds.get_depth_scale())
        except Exception:
            pass

        report["sensors"].append(s_info)

    # Start a pipeline with common streams; include IMU if available
    cfg = rs.config()
    # Depth + Color (fallback to defaults if not supported)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Try IMU streams (D435i has them)
    try:
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
    except Exception:
        pass

    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    try:
        active = {"streams": {}, "extrinsics": {}, "imu_intrinsics": {}}

        # Collect intrinsics for video streams
        for st in [rs.stream.depth, rs.stream.color, rs.stream.infrared]:
            try:
                sp = profile.get_stream(st)
                vp = sp.as_video_stream_profile()
                if vp:
                    active["streams"][str(st).split('.')[-1]] = {
                        "profile": profile_dict(vp),
                        "intrinsics": get_video_intrinsics(vp)
                    }
            except Exception:
                pass

        # Motion intrinsics
        for st in [rs.stream.gyro, rs.stream.accel]:
            try:
                sp = profile.get_stream(st)
                mp = sp.as_motion_stream_profile()
                if mp:
                    active["imu_intrinsics"][str(st).split('.')[-1]] = get_motion_intrinsics(mp)
            except Exception:
                pass

        # Extrinsics depth->color if both exist
        try:
            d_vp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            c_vp = profile.get_stream(rs.stream.color).as_video_stream_profile()
            ex = d_vp.get_extrinsics_to(c_vp)
            active["extrinsics"]["depth_to_color"] = {
                "rotation_3x3_rowmajor": as_list(ex.rotation),
                "translation_xyz_m": as_list(ex.translation),
            }
        except Exception:
            pass

        report["active_profile"] = active

        # Sample a few frames to show live data
        print("\nSampling frames for ~2 seconds...")
        t_end = time.time() + 2.0
        last_samples = defaultdict(dict)
        depth_scale = None
        try:
            dsens = dev.first_depth_sensor()
            depth_scale = dsens.get_depth_scale()
        except Exception:
            # or fetch from earlier
            for s in report["sensors"]:
                if s.get("depth_scale"):
                    depth_scale = s["depth_scale"]; break

        while time.time() < t_end:
            frames = pipeline.wait_for_frames(2000)

            # Depth
            try:
                df = frames.get_depth_frame()
                if df:
                    d = np.asanyarray(df.get_data())
                    if depth_scale:
                        d_m = d * float(depth_scale)
                    else:
                        d_m = d.astype(np.float32)
                    nonzero = d_m[d > 0]
                    if nonzero.size:
                        last_samples["depth"] = {
                            "width": int(df.get_width()),
                            "height": int(df.get_height()),
                            "timestamp_ms": float(df.get_timestamp()),
                            "min_m": float(np.percentile(nonzero, 5)),
                            "median_m": float(np.median(nonzero)),
                            "max_m": float(np.percentile(nonzero, 95)),
                        }
            except Exception:
                pass

            # Color
            try:
                cf = frames.get_color_frame()
                if cf:
                    last_samples["color"] = {
                        "width": int(cf.get_width()),
                        "height": int(cf.get_height()),
                        "timestamp_ms": float(cf.get_timestamp()),
                    }
            except Exception:
                pass

            # IMU (gyro/accel may arrive in separate frames)
            try:
                # Newer SDKs allow first_or_default; fall back to frames.get_frame
                gf = frames.first_or_default(rs.stream.gyro) if hasattr(frames, "first_or_default") else None
                af = frames.first_or_default(rs.stream.accel) if hasattr(frames, "first_or_default") else None
            except Exception:
                gf = af = None

            try:
                if gf:
                    md = gf.as_motion_frame().get_motion_data()
                    last_samples["gyro"] = {
                        "timestamp_ms": float(gf.get_timestamp()),
                        "x": float(md.x), "y": float(md.y), "z": float(md.z),
                        "units": "rad/s"
                    }
            except Exception:
                pass

            try:
                if af:
                    md = af.as_motion_frame().get_motion_data()
                    last_samples["accel"] = {
                        "timestamp_ms": float(af.get_timestamp()),
                        "x": float(md.x), "y": float(md.y), "z": float(md.z),
                        "units": "m/s^2"
                    }
            except Exception:
                pass

        report["live_samples"] = last_samples

    finally:
        pipeline.stop()

    # Print a concise summary
    print("\n=== SUMMARY ===")
    print(json.dumps({
        "device": report["device"],
        "depth_scale": next((s["depth_scale"] for s in report["sensors"] if s.get("depth_scale")), None),
        "active_streams": list(report.get("active_profile", {}).get("streams", {}).keys()),
        "has_gyro": "gyro" in report.get("active_profile", {}).get("imu_intrinsics", {}),
        "has_accel": "accel" in report.get("active_profile", {}).get("imu_intrinsics", {}),
        "sample_keys": list(report.get("live_samples", {}).keys()),
    }, indent=2))

    # Save full JSON
    out_path = "realsense_d435i_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
