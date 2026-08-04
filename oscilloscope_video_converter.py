#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Black/white video -> stereo XY oscilloscope WAV converter.

Left channel  = X axis
Right channel = Y axis

The program contains both a Tkinter desktop GUI and a CLI.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class ConvertOptions:
    sample_rate: int = 192_000
    process_width: int = 640
    process_height: int = 480
    threshold: int = 127
    invert_input: bool = False
    min_perimeter: float = 8.0
    simplify_percent: float = 0.12
    amplitude: float = 0.85
    traces_per_frame: int = 2
    swap_xy: bool = False
    invert_x: bool = False
    invert_y: bool = False
    smoothing: float = 0.0
    border_margin: float = 0.04

    def validate(self) -> None:
        if self.sample_rate < 8_000:
            raise ValueError("采样率必须至少为 8000 Hz。")
        if self.process_width < 32 or self.process_height < 32:
            raise ValueError("处理分辨率至少应为 32×32。")
        if not 0 <= self.threshold <= 255:
            raise ValueError("阈值必须在 0 到 255 之间。")
        if self.min_perimeter < 0:
            raise ValueError("最小轮廓周长不能为负数。")
        if not 0 <= self.simplify_percent <= 10:
            raise ValueError("轮廓简化百分比必须在 0 到 10 之间。")
        if not 0 < self.amplitude <= 1:
            raise ValueError("输出幅度必须在 0 到 1 之间。")
        if self.traces_per_frame < 1:
            raise ValueError("每帧重复扫描次数至少为 1。")
        if not 0 <= self.smoothing < 1:
            raise ValueError("平滑系数必须在 0（关闭）到 1 之间。")
        if not 0 <= self.border_margin < 0.45:
            raise ValueError("边缘留白必须在 0 到 0.45 之间。")


def _resample_polyline(points: np.ndarray, count: int, closed: bool = True) -> np.ndarray:
    """Resample a polyline to `count` equally-spaced points."""
    count = int(count)
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((count, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, count, axis=0)

    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    segments = np.diff(pts, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-6
    if not np.any(valid):
        return np.repeat(pts[:1], count, axis=0)

    # Keep degenerate points out of interpolation while retaining sequence order.
    kept = np.concatenate([[True], valid])
    pts = pts[kept]
    segments = np.diff(pts, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])

    if total <= 1e-6:
        return np.repeat(pts[:1], count, axis=0)

    targets = np.linspace(0.0, total, count, endpoint=not closed, dtype=np.float32)
    idx = np.searchsorted(cumulative, targets, side="right") - 1
    idx = np.clip(idx, 0, len(lengths) - 1)
    local = (targets - cumulative[idx]) / np.maximum(lengths[idx], 1e-12)
    result = pts[idx] + segments[idx] * local[:, None]
    return result.astype(np.float32, copy=False)


def _extract_contours(frame_bgr: np.ndarray, options: ConvertOptions) -> list[np.ndarray]:
    resized = cv2.resize(
        frame_bgr,
        (options.process_width, options.process_height),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    threshold_type = cv2.THRESH_BINARY_INV if options.invert_input else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, options.threshold, 255, threshold_type)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    result: list[np.ndarray] = []

    for contour in contours:
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < options.min_perimeter:
            continue

        epsilon = perimeter * (options.simplify_percent / 100.0)
        if epsilon > 0:
            contour = cv2.approxPolyDP(contour, epsilon, True)

        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) >= 2:
            result.append(points)

    return result


def _order_contours_nearest(contours: list[np.ndarray], start: np.ndarray) -> list[np.ndarray]:
    """Nearest-neighbour contour ordering; rotate each closed contour to its nearest point."""
    remaining = [c.copy() for c in contours]
    ordered: list[np.ndarray] = []
    current = np.asarray(start, dtype=np.float32)

    while remaining:
        best_contour_index = 0
        best_point_index = 0
        best_distance = float("inf")

        for contour_index, contour in enumerate(remaining):
            distances = np.sum((contour - current) ** 2, axis=1)
            point_index = int(np.argmin(distances))
            distance = float(distances[point_index])
            if distance < best_distance:
                best_distance = distance
                best_contour_index = contour_index
                best_point_index = point_index

        contour = remaining.pop(best_contour_index)
        contour = np.roll(contour, -best_point_index, axis=0)
        ordered.append(contour)
        current = contour[0]

    return ordered


def _allocate_samples(weights: np.ndarray, total: int, minimum: int = 2) -> np.ndarray:
    """Allocate exactly `total` integer samples proportionally to weights."""
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if total <= 0:
        return np.zeros(n, dtype=np.int64)

    minimum = max(0, int(minimum))
    base = np.full(n, minimum, dtype=np.int64)
    if int(base.sum()) > total:
        base[:] = 0

    remaining = total - int(base.sum())
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    exact = weights / weights.sum() * remaining
    extra = np.floor(exact).astype(np.int64)
    allocation = base + extra

    leftover = total - int(allocation.sum())
    if leftover > 0:
        order = np.argsort(-(exact - extra))
        allocation[order[:leftover]] += 1
    elif leftover < 0:
        order = np.argsort(exact - extra)
        for index in order:
            removable = min(allocation[index], -leftover)
            allocation[index] -= removable
            leftover += removable
            if leftover == 0:
                break

    return allocation


def _contour_perimeter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    closed = np.vstack([points, points[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def _frame_to_xy(
    frame_bgr: np.ndarray,
    samples_per_frame: int,
    options: ConvertOptions,
    previous_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    contours = _extract_contours(frame_bgr, options)
    if samples_per_frame <= 0:
        return np.empty((0, 2), np.float32), previous_point

    if not contours:
        held = np.repeat(previous_point.reshape(1, 2), samples_per_frame, axis=0)
        return held.astype(np.float32), previous_point

    contours = _order_contours_nearest(contours, previous_point)
    perimeters = np.array([max(_contour_perimeter(c), 1.0) for c in contours], dtype=np.float64)

    repeats = max(1, min(options.traces_per_frame, samples_per_frame))
    repeat_sizes = _allocate_samples(np.ones(repeats, dtype=np.float64), samples_per_frame, minimum=0)
    chunks: list[np.ndarray] = []
    current = previous_point.copy()

    for repeat_size in repeat_sizes:
        repeat_size = int(repeat_size)
        if repeat_size <= 0:
            continue

        # Re-order each repeat from the current beam position to reduce visible jump lines.
        ordered = _order_contours_nearest(contours, current)
        weights = np.array([max(_contour_perimeter(c), 1.0) for c in ordered], dtype=np.float64)
        counts = _allocate_samples(weights, repeat_size, minimum=2)

        repeat_chunks: list[np.ndarray] = []
        for contour, count in zip(ordered, counts):
            if count <= 0:
                continue
            sampled = _resample_polyline(contour, int(count), closed=True)
            repeat_chunks.append(sampled)
            if len(sampled):
                current = sampled[-1]

        if repeat_chunks:
            chunk = np.vstack(repeat_chunks)
        else:
            chunk = np.repeat(current.reshape(1, 2), repeat_size, axis=0)

        # Numerical guard: force the exact assigned length.
        if len(chunk) < repeat_size:
            chunk = np.vstack([chunk, np.repeat(chunk[-1:].copy(), repeat_size - len(chunk), axis=0)])
        elif len(chunk) > repeat_size:
            chunk = chunk[:repeat_size]
        chunks.append(chunk)

    if not chunks:
        points = np.repeat(previous_point.reshape(1, 2), samples_per_frame, axis=0)
    else:
        points = np.vstack(chunks)

    if len(points) < samples_per_frame:
        points = np.vstack([points, np.repeat(points[-1:].copy(), samples_per_frame - len(points), axis=0)])
    elif len(points) > samples_per_frame:
        points = points[:samples_per_frame]

    return points.astype(np.float32, copy=False), points[-1].copy()


def _pixel_points_to_audio(points: np.ndarray, options: ConvertOptions) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int16)

    width = max(2, options.process_width)
    height = max(2, options.process_height)
    margin = options.border_margin
    usable = 1.0 - 2.0 * margin

    x = points[:, 0] / (width - 1)
    y = points[:, 1] / (height - 1)

    # Pixel coordinates have Y downward; oscilloscopes conventionally display Y upward.
    x = (x * usable + margin) * 2.0 - 1.0
    y = 1.0 - (y * usable + margin) * 2.0

    if options.invert_x:
        x = -x
    if options.invert_y:
        y = -y
    if options.swap_xy:
        x, y = y, x

    stereo = np.column_stack([x, y]).astype(np.float32)

    if options.smoothing > 0 and len(stereo) > 1:
        alpha = float(options.smoothing)
        smoothed = np.empty_like(stereo)
        smoothed[0] = stereo[0]
        for i in range(1, len(stereo)):
            smoothed[i] = alpha * smoothed[i - 1] + (1.0 - alpha) * stereo[i]
        stereo = smoothed

    stereo *= float(options.amplitude)
    stereo = np.clip(stereo, -1.0, 1.0)
    return np.round(stereo * 32767.0).astype(np.int16)


def convert_video_to_wav(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    options: ConvertOptions,
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
) -> dict[str, float | int | str]:
    options.validate()
    input_path = str(Path(input_path).expanduser().resolve())
    output_path = str(Path(output_path).expanduser().resolve())

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"找不到输入视频：{input_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError("无法打开视频。请确认 OpenCV 支持该视频编码。")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("无法读取有效的视频帧率。")
    if total_frames <= 0:
        total_frames = 0

    temp_output = output_path + ".partial.wav"
    previous_point = np.array(
        [(options.process_width - 1) / 2.0, (options.process_height - 1) / 2.0],
        dtype=np.float32,
    )
    frame_index = 0
    written_samples = 0

    try:
        with wave.open(temp_output, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(options.sample_rate)

            while True:
                if cancelled and cancelled():
                    raise InterruptedError("用户取消了转换。")

                ok, frame = capture.read()
                if not ok:
                    break

                start_sample = round(frame_index * options.sample_rate / fps)
                end_sample = round((frame_index + 1) * options.sample_rate / fps)
                samples_for_frame = max(1, end_sample - start_sample)

                pixel_points, previous_point = _frame_to_xy(
                    frame,
                    samples_for_frame,
                    options,
                    previous_point,
                )
                audio = _pixel_points_to_audio(pixel_points, options)
                wav_file.writeframesraw(audio.astype("<i2", copy=False).tobytes())
                written_samples += len(audio)
                frame_index += 1

                if progress and (frame_index == 1 or frame_index % 5 == 0):
                    progress(frame_index, total_frames, "正在提取轮廓并写入 WAV…")

        capture.release()
        os.replace(temp_output, output_path)

    except Exception:
        capture.release()
        try:
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except OSError:
            pass
        raise

    if frame_index == 0:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise RuntimeError("视频中没有可读取的帧。")

    duration = written_samples / options.sample_rate
    if progress:
        progress(frame_index, total_frames or frame_index, "转换完成。")

    return {
        "input": input_path,
        "output": output_path,
        "frames": frame_index,
        "fps": fps,
        "samples": written_samples,
        "duration_seconds": duration,
        "sample_rate": options.sample_rate,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将黑白视频转换为模拟示波器 XY 模式可播放的立体声 WAV。"
    )
    parser.add_argument("input", nargs="?", help="输入视频路径；不填写则启动图形界面")
    parser.add_argument("output", nargs="?", help="输出 WAV 路径")
    parser.add_argument("--sample-rate", type=int, default=192000, help="WAV 采样率，默认 192000")
    parser.add_argument("--width", type=int, default=640, help="内部处理宽度，默认 640")
    parser.add_argument("--height", type=int, default=480, help="内部处理高度，默认 480")
    parser.add_argument("--threshold", type=int, default=127, help="二值化阈值，默认 127")
    parser.add_argument("--invert-input", action="store_true", help="将黑色视为主体、白色视为背景")
    parser.add_argument("--min-perimeter", type=float, default=8.0, help="忽略小于该周长的轮廓")
    parser.add_argument("--simplify", type=float, default=0.12, help="轮廓简化百分比，默认 0.12")
    parser.add_argument("--amplitude", type=float, default=0.85, help="输出幅度 0~1，默认 0.85")
    parser.add_argument("--traces", type=int, default=2, help="每个视频帧重复扫描次数，默认 2")
    parser.add_argument("--swap-xy", action="store_true", help="交换 X/Y 声道")
    parser.add_argument("--invert-x", action="store_true", help="反转 X 轴")
    parser.add_argument("--invert-y", action="store_true", help="反转 Y 轴")
    parser.add_argument("--smoothing", type=float, default=0.0, help="一阶平滑系数 0~<1")
    parser.add_argument("--margin", type=float, default=0.04, help="画面边缘留白比例，默认 0.04")
    return parser


def _run_cli(args: argparse.Namespace) -> int:
    if not args.input or not args.output:
        print("CLI 模式需要同时提供输入视频和输出 WAV。", file=sys.stderr)
        return 2

    options = ConvertOptions(
        sample_rate=args.sample_rate,
        process_width=args.width,
        process_height=args.height,
        threshold=args.threshold,
        invert_input=args.invert_input,
        min_perimeter=args.min_perimeter,
        simplify_percent=args.simplify,
        amplitude=args.amplitude,
        traces_per_frame=args.traces,
        swap_xy=args.swap_xy,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        smoothing=args.smoothing,
        border_margin=args.margin,
    )

    def progress(done: int, total: int, message: str) -> None:
        if total > 0:
            print(f"\r{message} {done}/{total} ({done / total * 100:.1f}%)", end="", flush=True)
        else:
            print(f"\r{message} {done} 帧", end="", flush=True)

    try:
        result = convert_video_to_wav(args.input, args.output, options, progress=progress)
    except Exception as exc:
        print(f"\n转换失败：{exc}", file=sys.stderr)
        return 1

    print(
        f"\n完成：{result['frames']} 帧，{result['duration_seconds']:.3f} 秒，"
        f"{result['sample_rate']} Hz\n输出：{result['output']}"
    )
    return 0


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("黑白视频 → 示波器 XY 音频")
    root.geometry("760x690")
    root.minsize(720, 620)

    ui_queue: queue.Queue[tuple[str, object]] = queue.Queue()
    cancel_event = threading.Event()
    worker: list[Optional[threading.Thread]] = [None]

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    sample_rate_var = tk.StringVar(value="192000")
    width_var = tk.StringVar(value="640")
    height_var = tk.StringVar(value="480")
    threshold_var = tk.StringVar(value="127")
    min_perimeter_var = tk.StringVar(value="8")
    simplify_var = tk.StringVar(value="0.12")
    amplitude_var = tk.StringVar(value="0.85")
    traces_var = tk.StringVar(value="2")
    smoothing_var = tk.StringVar(value="0")
    margin_var = tk.StringVar(value="0.04")

    invert_input_var = tk.BooleanVar(value=False)
    swap_xy_var = tk.BooleanVar(value=False)
    invert_x_var = tk.BooleanVar(value=False)
    invert_y_var = tk.BooleanVar(value=False)

    outer = ttk.Frame(root, padding=16)
    outer.pack(fill="both", expand=True)
    outer.columnconfigure(1, weight=1)

    def add_entry(row: int, label: str, variable: tk.StringVar, width: int = 16) -> ttk.Entry:
        ttk.Label(outer, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        entry = ttk.Entry(outer, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        return entry

    ttk.Label(
        outer,
        text="黑白视频 → 模拟示波器 XY 立体声音频",
        font=("TkDefaultFont", 15, "bold"),
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

    ttk.Label(outer, text="输入视频").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
    ttk.Entry(outer, textvariable=input_var).grid(row=1, column=1, sticky="ew", pady=5)

    def choose_input() -> None:
        path = filedialog.askopenfilename(
            title="选择黑白视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            input_var.set(path)
            if not output_var.get().strip():
                output_var.set(str(Path(path).with_suffix(".xy.wav")))

    ttk.Button(outer, text="浏览…", command=choose_input).grid(row=1, column=2, padx=(10, 0), pady=5)

    ttk.Label(outer, text="输出 WAV").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
    ttk.Entry(outer, textvariable=output_var).grid(row=2, column=1, sticky="ew", pady=5)

    def choose_output() -> None:
        path = filedialog.asksaveasfilename(
            title="保存 XY 音频",
            defaultextension=".wav",
            filetypes=[("WAV 音频", "*.wav")],
        )
        if path:
            output_var.set(path)

    ttk.Button(outer, text="浏览…", command=choose_output).grid(row=2, column=2, padx=(10, 0), pady=5)

    separator = ttk.Separator(outer)
    separator.grid(row=3, column=0, columnspan=3, sticky="ew", pady=12)

    ttk.Label(outer, text="采样率 Hz").grid(row=4, column=0, sticky="w", padx=(0, 10), pady=5)
    ttk.Combobox(
        outer,
        textvariable=sample_rate_var,
        values=("48000", "96000", "192000", "384000"),
        state="normal",
        width=14,
    ).grid(row=4, column=1, sticky="w", pady=5)

    resolution_frame = ttk.Frame(outer)
    ttk.Label(outer, text="处理分辨率").grid(row=5, column=0, sticky="w", padx=(0, 10), pady=5)
    resolution_frame.grid(row=5, column=1, sticky="w", pady=5)
    ttk.Entry(resolution_frame, textvariable=width_var, width=8).pack(side="left")
    ttk.Label(resolution_frame, text=" × ").pack(side="left")
    ttk.Entry(resolution_frame, textvariable=height_var, width=8).pack(side="left")

    add_entry(6, "二值化阈值", threshold_var)
    add_entry(7, "最小轮廓周长", min_perimeter_var)
    add_entry(8, "轮廓简化 %", simplify_var)
    add_entry(9, "输出幅度 0~1", amplitude_var)
    add_entry(10, "每帧重复扫描", traces_var)
    add_entry(11, "平滑系数 0~<1", smoothing_var)
    add_entry(12, "边缘留白 0~0.45", margin_var)

    options_frame = ttk.LabelFrame(outer, text="方向与输入", padding=10)
    options_frame.grid(row=13, column=0, columnspan=3, sticky="ew", pady=(12, 8))
    ttk.Checkbutton(options_frame, text="输入反相（黑色是主体）", variable=invert_input_var).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(options_frame, text="交换 X/Y", variable=swap_xy_var).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(options_frame, text="反转 X", variable=invert_x_var).pack(side="left", padx=(0, 14))
    ttk.Checkbutton(options_frame, text="反转 Y", variable=invert_y_var).pack(side="left")

    note = (
        "建议从 192 kHz、幅度 0.85、每帧重复 2 次开始。左右声道分别接示波器 X/Y。\n"
        "只有 X/Y 两路时无法真正关闭电子束，因此多个不相连轮廓之间可能出现较暗的连接线；程序会尽量缩短这些跳线。"
    )
    ttk.Label(outer, text=note, foreground="#555555", wraplength=700, justify="left").grid(
        row=14, column=0, columnspan=3, sticky="w", pady=(4, 10)
    )

    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(outer, variable=progress_var, maximum=100)
    progress_bar.grid(row=15, column=0, columnspan=3, sticky="ew", pady=(4, 4))
    status_var = tk.StringVar(value="准备就绪")
    ttk.Label(outer, textvariable=status_var).grid(row=16, column=0, columnspan=3, sticky="w")

    button_frame = ttk.Frame(outer)
    button_frame.grid(row=17, column=0, columnspan=3, sticky="e", pady=(14, 0))

    def collect_options() -> ConvertOptions:
        return ConvertOptions(
            sample_rate=int(sample_rate_var.get()),
            process_width=int(width_var.get()),
            process_height=int(height_var.get()),
            threshold=int(threshold_var.get()),
            invert_input=invert_input_var.get(),
            min_perimeter=float(min_perimeter_var.get()),
            simplify_percent=float(simplify_var.get()),
            amplitude=float(amplitude_var.get()),
            traces_per_frame=int(traces_var.get()),
            swap_xy=swap_xy_var.get(),
            invert_x=invert_x_var.get(),
            invert_y=invert_y_var.get(),
            smoothing=float(smoothing_var.get()),
            border_margin=float(margin_var.get()),
        )

    def set_running(running: bool) -> None:
        convert_button.configure(state="disabled" if running else "normal")
        cancel_button.configure(state="normal" if running else "disabled")

    def start_conversion() -> None:
        input_path = input_var.get().strip()
        output_path = output_var.get().strip()
        if not input_path or not output_path:
            messagebox.showerror("缺少路径", "请选择输入视频和输出 WAV。")
            return

        try:
            options = collect_options()
            options.validate()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        if os.path.exists(output_path):
            if not messagebox.askyesno("覆盖文件", "输出文件已经存在，是否覆盖？"):
                return

        cancel_event.clear()
        progress_var.set(0)
        status_var.set("正在打开视频…")
        set_running(True)

        def report(done: int, total: int, text: str) -> None:
            ui_queue.put(("progress", (done, total, text)))

        def work() -> None:
            try:
                result = convert_video_to_wav(
                    input_path,
                    output_path,
                    options,
                    progress=report,
                    cancelled=cancel_event.is_set,
                )
                ui_queue.put(("done", result))
            except InterruptedError as exc:
                ui_queue.put(("cancelled", str(exc)))
            except Exception:
                ui_queue.put(("error", traceback.format_exc()))

        worker[0] = threading.Thread(target=work, daemon=True)
        worker[0].start()

    def cancel_conversion() -> None:
        cancel_event.set()
        status_var.set("正在取消…")
        cancel_button.configure(state="disabled")

    convert_button = ttk.Button(button_frame, text="开始转换", command=start_conversion)
    convert_button.pack(side="left", padx=(0, 8))
    cancel_button = ttk.Button(button_frame, text="取消", command=cancel_conversion, state="disabled")
    cancel_button.pack(side="left")

    def poll_queue() -> None:
        try:
            while True:
                kind, payload = ui_queue.get_nowait()
                if kind == "progress":
                    done, total, text = payload  # type: ignore[misc]
                    if total:
                        progress_var.set(done / total * 100)
                        status_var.set(f"{text}  {done}/{total} 帧")
                    else:
                        status_var.set(f"{text}  {done} 帧")
                elif kind == "done":
                    result = payload  # type: ignore[assignment]
                    progress_var.set(100)
                    status_var.set("转换完成")
                    set_running(False)
                    messagebox.showinfo(
                        "完成",
                        f"已生成：\n{result['output']}\n\n"
                        f"帧数：{result['frames']}\n"
                        f"时长：{result['duration_seconds']:.3f} 秒\n"
                        f"采样率：{result['sample_rate']} Hz",
                    )
                elif kind == "cancelled":
                    progress_var.set(0)
                    status_var.set("已取消")
                    set_running(False)
                elif kind == "error":
                    progress_var.set(0)
                    status_var.set("转换失败")
                    set_running(False)
                    error_text = str(payload)
                    last_line = error_text.strip().splitlines()[-1] if error_text.strip() else "未知错误"
                    messagebox.showerror("转换失败", last_line)
        except queue.Empty:
            pass
        root.after(100, poll_queue)

    def on_close() -> None:
        if worker[0] and worker[0].is_alive():
            if not messagebox.askyesno("退出", "转换仍在进行，是否取消并退出？"):
                return
            cancel_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll_queue)
    root.mainloop()


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.input is None:
        launch_gui()
        return 0
    return _run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
