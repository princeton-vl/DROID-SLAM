import time
from typing import Any, Optional, Union

import glm
from glm import cos, radians, sin

import moderngl_window
from moderngl_window.scene.camera import Camera


class OrbitCamera(Camera):
    def __init__(
        self,
        target: Union[glm.vec3, tuple[float, float, float]] = (0.0, 0.0, 0.0),
        radius: float = 2.0,
        angles: tuple[float, float] = (60.0, -100.0),
        **kwargs: Any,
    ):
        self.radius = radius  # radius in base units
        self.angle_x, self.angle_y = angles  # angles in degrees
        self.target = glm.vec3(target)  # camera target in base units
        self.up = glm.vec3(0.0, 1.0, 0.0)  # camera up vector

        self._mouse_sensitivity = 1.0
        self._zoom_sensitivity = 1.0

        self.world_up = glm.vec3(0.0, -1.0, 0.0)
        self._mouse_sensitivity = 1.0
        self._zoom_sensitivity = 1.0
        self._pan_sensitivity = 0.001
        super().__init__(**kwargs)

    @property
    def pan_sensitivity(self) -> float:
        return self._pan_sensitivity

    @pan_sensitivity.setter
    def pan_sensitivity(self, value: float):
        self._pan_sensitivity = value

    def rot_state(self, dx: float, dy: float) -> None:
        """Unclamped, continuous orbit around the target."""
        self.angle_x = (self.angle_x - dx * self.mouse_sensitivity / 10.0)
        self.angle_y = (self.angle_y - dy * self.mouse_sensitivity / 10.0)
        self.angle_y = max(min(self.angle_y, -5.0), -175.0)
        # self.angle_y = self.angle_y.clamp()

    def zoom_state(self, y_offset: float) -> None:
        self.radius = max(1.0, self.radius - y_offset * self._zoom_sensitivity)

    @property
    def matrix(self) -> glm.mat4:
        # compute camera position as before
        px = cos(radians(self.angle_x)) * sin(radians(self.angle_y)) * self.radius + self.target.x
        py = cos(radians(self.angle_y)) * self.radius + self.target.y
        pz = sin(radians(self.angle_x)) * sin(radians(self.angle_y)) * self.radius + self.target.z
        pos = glm.vec3(px, py, pz)
        self.set_position(*pos)
        return glm.lookAt(pos, self.target, self.world_up)

    def pan_state(self, dx: float, dy: float) -> None:
        """Pan the orbit‐center using camera‐relative axes."""
        # Recompute camera position & forward vector
        px = cos(radians(self.angle_x)) * sin(radians(self.angle_y)) * self.radius + self.target.x
        py = cos(radians(self.angle_y)) * self.radius + self.target.y
        pz = sin(radians(self.angle_x)) * sin(radians(self.angle_y)) * self.radius + self.target.z
        pos = glm.vec3(px, py, pz)
        forward = glm.normalize(self.target - pos)

        # Build a stable right & up in camera‐space:
        right = glm.normalize(glm.cross(forward, self.world_up))
        up    = glm.normalize(glm.cross(right, forward))

        # Screen‐space offset: right = +dx, up = +dy
        offset = (-right * dx + up * dy) * self._pan_sensitivity * self.radius
        self.target += offset



class OrbitDragCameraWindow(moderngl_window.WindowConfig):
    """Base class with drag-based 3D orbit support

    Click and drag with the left mouse button to orbit the camera around the view point.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = OrbitCamera(aspect_ratio=self.wnd.aspect_ratio)

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if action == keys.ACTION_PRESS:
            if key == keys.SPACE:
                self.timer.toggle_pause()

    def on_mouse_drag_event(self, x: int, y: int, dx: float, dy: float):
        mb = self.wnd.mouse_states
        if mb.right:                         # ← right‑button drag → pan
            self.camera.pan_state(dx, dy)
        else:                                # ← left‑button drag → orbit
            self.camera.rot_state(dx, dy)
    
    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.camera.zoom_state(y_offset)

    def on_resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

