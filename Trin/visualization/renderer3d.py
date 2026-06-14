import numpy as np
from PyQt6.QtWidgets import QLabel


class Renderer3D:
    def __init__(self, parent=None) -> None:
        self.available = False
        self.widget = QLabel("3D view requires PyVistaQt.")
        self._plotter = None
        self._electron_mesh = None
        self._electron_actor = None
        self._neutral_mesh = None
        self._neutral_actor = None
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

        self._first_render = True
        self._electron_mesh = pv.PolyData(np.zeros((1, 3)))
        self._neutral_mesh = None
        self._electron_actor = self._plotter.add_mesh(
            self._electron_mesh,
            render_points_as_spheres=True,
            point_size=12,
            color="yellow",
        )
        self._neutral_actor = None
        self._plotter.view_isometric()

        self._set_default_domain()

    # MODIFICADO: Ahora acepta explícitamente los 4 argumentos posicionales
    def update_particles(self, electron_positions, neutral_positions=None, ion_positions=None, recombined_positions=None) -> None:
        if not self.available:
            return

        # 1. Electrones
        if electron_positions is None or len(electron_positions) == 0:
            electron_points = np.empty((0, 3))
        else:
            electron_points = np.asarray(electron_positions, dtype=float)

        # 2. Neutras
        if neutral_positions is None or len(neutral_positions) == 0:
            neutral_points = np.empty((0, 3))
        else:
            neutral_points = np.asarray(neutral_positions, dtype=float)

        # 3. Iones Positivos
        if ion_positions is None or len(ion_positions) == 0:
            ion_points = np.empty((0, 3))
        else:
            ion_points = np.asarray(ion_positions, dtype=float)

        # 4. Recombinadas (Moradas)
        if recombined_positions is None or len(recombined_positions) == 0:
            recombined_points = np.empty((0, 3))
        else:
            recombined_points = np.asarray(recombined_positions, dtype=float)

        # Reemplazar los 4 actores correspondientes
        self._replace_actor("electron", electron_points, "yellow")
        self._replace_actor("neutral", neutral_points, "dodgerblue")
        self._replace_actor("ion", ion_points, "orangered")
        self._replace_actor("recombined", recombined_points, "purple")

        if self._first_render:
            self._plotter.reset_camera()
            self._first_render = False
        self._plotter.render()

    def _replace_actor(self, kind: str, points: np.ndarray, color: str) -> None:
        actor_attr = f"_{kind}_actor"
        mesh_attr = f"_{kind}_mesh"
        actor = getattr(self, actor_attr)
        if actor is not None:
            self._plotter.remove_actor(actor)
        if points.size == 0:
            setattr(self, mesh_attr, None)
            setattr(self, actor_attr, None)
            return
        mesh = self._pv.PolyData(points)
        actor = self._plotter.add_mesh(
            mesh,
            render_points_as_spheres=True,
            point_size=12,
            color=color,
        )
        setattr(self, mesh_attr, mesh)
        setattr(self, actor_attr, actor)

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
        if self._electron_actor is not None:
            self._plotter.remove_actor(self._electron_actor)
        if self._neutral_actor is not None:
            self._plotter.remove_actor(self._neutral_actor)
        if self._ion_actor is not None:          # <-- AGREGAR
            self._plotter.remove_actor(self._ion_actor)
        if self._recombined_actor is not None:   # <-- AGREGAR
            self._plotter.remove_actor(self._recombined_actor)
            
        self._electron_mesh = self._pv.PolyData(np.zeros((1, 3)))
        self._neutral_mesh = None
        self._ion_mesh = None                     # <-- AGREGAR
        self._recombined_mesh = None              # <-- AGREGAR
        
        self._electron_actor = self._plotter.add_mesh(
            self._electron_mesh,
            render_points_as_spheres=True,
            point_size=12,
            color="yellow",
        )
        self._neutral_actor = None
        self._ion_actor = None                    # <-- AGREGAR
        self._recombined_actor = None             # <-- AGREGAR
        self._first_render = True
        self._plotter.render()
