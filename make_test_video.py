#!/usr/bin/env python3
"""Generate a small black/white test video for the converter."""
from __future__ import annotations

import math
import cv2
import numpy as np

WIDTH, HEIGHT, FPS, SECONDS = 640, 480, 30, 4
writer = cv2.VideoWriter(
    "test_black_white.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (WIDTH, HEIGHT),
)
if not writer.isOpened():
    raise RuntimeError("无法创建测试视频。")

for i in range(FPS * SECONDS):
    t = i / FPS
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cx = int(WIDTH / 2 + 130 * math.sin(t * 1.7))
    cy = int(HEIGHT / 2 + 80 * math.cos(t * 1.1))
    radius = 70 + int(15 * math.sin(t * 2.4))
    cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 5)
    cv2.line(frame, (cx - 45, cy), (cx + 45, cy), (255, 255, 255), 4)
    cv2.line(frame, (cx, cy - 45), (cx, cy + 45), (255, 255, 255), 4)
    writer.write(frame)

writer.release()
print("已生成 test_black_white.mp4")
