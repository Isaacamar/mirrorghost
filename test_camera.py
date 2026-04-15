"""
Quick sanity check: tests camera capture and MediaPipe without any ML models.
Run: .venv/bin/python test_camera.py
Press Q to quit.
"""

import time
import sys
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import numpy as np

from src.camera import CameraProcessor
from src.features import FeatureAccumulator
from src.landmark_renderer import render_pose_image, render_blank_pose


def main():
    camera = CameraProcessor(camera_index=0)
    accumulator = FeatureAccumulator()

    pygame.init()
    screen = pygame.display.set_mode((512, 560))
    pygame.display.set_caption("camera test — press Q to quit")
    font = pygame.font.SysFont("monospace", 13)
    clock = pygame.time.Clock()

    camera.start()
    print("Camera started. Waiting for face...")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        frame = camera.get_latest()
        if frame:
            accumulator.update(frame)

        features = accumulator.get()

        if frame and frame.landmarks is not None:
            pose_img = render_pose_image(frame, features)
        else:
            pose_img = render_blank_pose()

        # Convert PIL to pygame surface
        surf = pygame.image.fromstring(pose_img.tobytes(), pose_img.size, "RGB")
        screen.fill((8, 8, 8))
        screen.blit(surf, (0, 0))

        # Status
        stab = features.stability
        conf = frame.confidence if frame else 0.0
        texts = [
            f"stability:  {stab:.2f}",
            f"confidence: {conf:.2f}",
            f"frames:     {features.frame_count}",
            f"staleness:  {features.staleness:.1f}s",
        ]
        for i, t in enumerate(texts):
            s = font.render(t, True, (80, 80, 80))
            screen.blit(s, (8, 516 + i * 14))

        pygame.display.flip()
        clock.tick(30)

    camera.stop()
    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
