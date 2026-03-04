import cv2
import time


class Button:
    def __init__(self, label: str, x: int, y: int, w: int, h: int):
        self.label = label
        self.x, self.y, self.w, self.h = x, y, w, h
        self.toggled = False

    def contains(self, px: int, py: int) -> bool:
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)

    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


class HoverSelectUI:
    """
    Kinect-style hover-to-select UI.
    - Draws buttons
    - Draws cursor
    - If cursor hovers over a button for dwell_seconds, triggers select
    - Shows circular progress meter over hovered button
    """

    def __init__(
        self,
        dwell_seconds: float = 1.5,
        smoothing_alpha: float = 0.25,
        cursor_radius: int = 10,
    ):
        self.dwell_seconds = dwell_seconds
        self.smoothing_alpha = smoothing_alpha
        self.cursor_radius = cursor_radius

        self.buttons = []
        self.hovered_idx = None
        self.hover_start_t = None

        self.cursor_x = None
        self.cursor_y = None

        self.prev_t = time.time()
        self.fps = 0.0

        self._initialized_layout = False

    def init_layout(self, frame_w: int, frame_h: int):
        """Create buttons based on frame dimensions (call once when you know size)."""
        if self._initialized_layout:
            return

        bw, bh = 300, 80
        x0 = 60
        y0 = 80
        gap = 30

        self.buttons = [
            Button("Toggle Overlay", x0, y0 + 0 * (bh + gap), bw, bh),
            Button("Start Demo Mode", x0, y0 + 1 * (bh + gap), bw, bh),
            Button("Reset / Clear", x0, y0 + 2 * (bh + gap), bw, bh),
        ]

        self._initialized_layout = True

    def update_cursor_from_norm(self, tip_norm, frame_w: int, frame_h: int):
        """
        tip_norm: (x_norm, y_norm) or None
        Updates internal smoothed cursor position.
        """
        if tip_norm is None:
            # Hand lost -> stop selection
            self.hovered_idx = None
            self.hover_start_t = None
            return

        tx = int(tip_norm[0] * frame_w)
        ty = int(tip_norm[1] * frame_h)

        if self.cursor_x is None:
            self.cursor_x, self.cursor_y = tx, ty
        else:
            a = self.smoothing_alpha
            self.cursor_x = int((1 - a) * self.cursor_x + a * tx)
            self.cursor_y = int((1 - a) * self.cursor_y + a * ty)

        # Clamp
        self.cursor_x = max(0, min(frame_w - 1, self.cursor_x))
        self.cursor_y = max(0, min(frame_h - 1, self.cursor_y))

    def _draw_button(self, frame, btn: Button, hovered: bool):
        base = (60, 60, 60)
        hover = (90, 90, 90)
        on = (60, 120, 60)

        color = on if btn.toggled else (hover if hovered else base)

        cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), color, thickness=-1)
        cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), (220, 220, 220), thickness=2)

        text = btn.label + ("  [ON]" if btn.toggled else "")
        cv2.putText(
            frame,
            text,
            (btn.x + 12, btn.y + btn.h // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    def _draw_cursor(self, frame):
        if self.cursor_x is None or self.cursor_y is None:
            return
        x, y = self.cursor_x, self.cursor_y
        r = self.cursor_radius
        cv2.circle(frame, (x, y), r, (0, 255, 0), thickness=-1)
        cv2.circle(frame, (x, y), r + 6, (0, 255, 0), thickness=2)

    def _draw_progress_ring(self, frame, center, progress: float, radius: int = 38, thickness: int = 6):
        cx, cy = center
        cv2.circle(frame, (cx, cy), radius, (200, 200, 200), thickness)
        start_angle = -90
        end_angle = int(start_angle + 360 * max(0.0, min(1.0, progress)))
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle, end_angle, (0, 255, 255), thickness)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    def _compute_hover_target(self):
        if self.cursor_x is None or self.cursor_y is None:
            return None
        for i, btn in enumerate(self.buttons):
            if btn.contains(self.cursor_x, self.cursor_y):
                return i
        return None

    def _handle_selection(self, idx: int):
        btn = self.buttons[idx]
        if btn.label == "Reset / Clear":
            for b in self.buttons:
                b.toggled = False
        else:
            btn.toggled = not btn.toggled

    def update_and_draw(self, frame):
        """
        Main UI call per frame:
        - updates hover timing
        - triggers selection
        - draws everything
        Returns: list of events (strings) if you want to hook actions later
        """
        h, w, _ = frame.shape
        self.init_layout(w, h)

        events = []

        now = time.time()

        # Determine hover target
        new_hovered = self._compute_hover_target()

        if new_hovered is None:
            self.hovered_idx = None
            self.hover_start_t = None
        else:
            if self.hovered_idx != new_hovered:
                self.hovered_idx = new_hovered
                self.hover_start_t = now

        # Draw header
        cv2.putText(
            frame,
            "Hover cursor over a button for 1.5s to select",
            (60, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw buttons
        for i, btn in enumerate(self.buttons):
            self._draw_button(frame, btn, hovered=(i == self.hovered_idx))

        # Cursor
        self._draw_cursor(frame)

        # Dwell progress + trigger
        if self.hovered_idx is not None and self.hover_start_t is not None:
            elapsed = now - self.hover_start_t
            progress = min(1.0, elapsed / self.dwell_seconds)

            cx, cy = self.buttons[self.hovered_idx].center()
            self._draw_progress_ring(frame, (cx, cy), progress)

            if elapsed >= self.dwell_seconds:
                label = self.buttons[self.hovered_idx].label
                self._handle_selection(self.hovered_idx)
                events.append(f"selected:{label}")

                # reset so it doesn't instantly retrigger
                self.hovered_idx = None
                self.hover_start_t = None

        # FPS
        dt = now - self.prev_t
        self.prev_t = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (w - 180, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return events