import numpy as np
from PyQt6.QtWidgets import QLabel


class Renderer3D:
    def __init__(self, parent=None) -> None:
        self.available = False
        self.widget = QLabel("3D view requires PyVistaQt.")
        self._plotter = None
        self._mesh = None
        self._domain_actor = None

        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except Exception as exc:
            self.widget.setText(f"3D view unavailable: {exc}")
            return

        self.available = True
        self._pv = pv
        self._plotter = QtInteractor(parent)
        self.widget = self._plotter
        self._plotter.set_background("black")
        self._plotter.add_axes()

        self._mesh = pv.PolyData(np.zeros((1, 3)))
        self._first_render = True
        self._plotter.add_points(
            self._mesh,
            render_points_as_spheres=True,
            point_size=10,
            color="yellow",
        )
        self._plotter.view_isometric()

        self._set_default_domain()

    def update_particles(self, positions) -> None:
        if not self.available:
            return
        if positions is None or len(positions) == 0:
            points = np.zeros((1, 3))
        else:
            points = np.asarray(positions, dtype=float)
        self._mesh.points = points
        self._mesh.Modified()
        if self._first_render:
            self._plotter.reset_camera()
            self._first_render = False
        self._plotter.render()

    def set_domain(self, xy_extent: float, gap_distance: float) -> None:
        if not self.available:
            return
        self._add_domain_box(xy_extent, gap_distance)
        self._set_camera_for_domain(xy_extent, gap_distance)

    def _set_default_domain(self) -> None:
        self._add_domain_box(0.005, 0.01)
        self._set_camera_for_domain(0.005, 0.01)

    def _add_domain_box(self, xy_extent: float, gap_distance: float) -> None:
        if self._domain_actor is not None:
            self._plotter.remove_actor(self._domain_actor)
        bounds = (
            -xy_extent,
            xy_extent,
            -xy_extent,
            xy_extent,
            0.0,
            gap_distance,
        )
        box = self._pv.Box(bounds=bounds)
        self._domain_actor = self._plotter.add_mesh(
            box,
            style="wireframe",
            color="gray",
            opacity=0.35,
            line_width=1.0,
        )

    def _set_camera_for_domain(self, xy_extent: float, gap_distance: float) -> None:
        center = (0.0, 0.0, gap_distance * 0.5)
        distance = max(xy_extent, gap_distance) * 6.0
        self._plotter.camera_position = [
            (distance, distance, distance),
            center,
            (0.0, 0.0, 1.0),
        ]
        self._plotter.reset_camera()

    def clear(self) -> None:
        if not self.available:
            return
        self._mesh.points = np.zeros((1, 3))
        self._mesh.Modified()
        self._first_render = True
        self._plotter.render()
