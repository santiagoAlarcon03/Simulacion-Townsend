}import numpy as np
from PyQt6.QtWidgets import QLabel


class Renderer3D:
    def __init__(self, parent=None) -> None:
        """Inicializa la ventana gráfica 3D y define de forma explícita todos los canales de actores."""
        self.available = False
        self.widget = QLabel("3D view requires PyVistaQt.")
        self._plotter = None

        # --- DECLARACIÓN EXPLICÍTICA DE MALLAS (MESHES) ---
        self._electron_mesh = None
        self._neutral_mesh = None
        self._ion_mesh = None
        self._recombined_mesh = None

        # --- DECLARACIÓN EXPLICÍTICA DE ACTORES (ACTORS) ---
        self._electron_actor = None
        self._neutral_actor = None
        self._ion_actor = None
        self._recombined_actor = None
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

        # =====================================================
        # ASIGNACIÓN DE ESTADOS INICIALES (Evita desajustes de punteros)
        # =====================================================
        self._electron_mesh = pv.PolyData(np.zeros((1, 3)))
        self._electron_actor = self._plotter.add_mesh(
            self._electron_mesh,
            render_points_as_spheres=True,
            point_size=12,
            color="yellow",
        )
        
        # Inicializamos los demás canales en None de forma segura
        self._neutral_mesh = None
        self._neutral_actor = None
        
        self._ion_mesh = None
        self._ion_actor = None
        
        self._recombined_mesh = None
        self._recombined_actor = None

        self._plotter.view_isometric()
        self._set_default_domain()

    def update_particles(self, electron_positions, neutral_positions=None, ion_positions=None, recombined_positions=None) -> None:
        """Versión de depuración profunda para rastrear la desaparición de iones."""
        if not self.available:
            print("🚨 DEPURACIÓN: Renderer3D no está disponible.")
            return

        print("\n=== 🔍 DIAGNÓSTICO DE FRAME GRÁFICO ===")
        
        # 1. Monitoreo de Electrones
        e_len = len(electron_positions) if electron_positions is not None else 0
        print(f"   [1/4] Electrones recibidos: {e_len} | Tipo: {type(electron_positions)}")
        electron_points = np.asarray(electron_positions, dtype=float) if e_len > 0 else np.empty((0, 3))

        # 2. Monitoreo de Neutras
        n_len = len(neutral_positions) if neutral_positions is not None else 0
        print(f"   [2/4] Neutras recibidas:    {n_len} | Tipo: {type(neutral_positions)}")
        neutral_points = np.asarray(neutral_positions, dtype=float) if n_len > 0 else np.empty((0, 3))

        # 3. Monitoreo Crítico de Iones
        i_len = len(ion_positions) if ion_positions is not None else 0
        print(f"🔴 [3/4] IONES RECIBIDOS:      {i_len} | Tipo: {type(ion_positions)}")
        if ion_positions is not None and i_len > 0:
            print(f"      👉 Muestra de datos de Iones (primeros 2): \n{ion_positions[:2]}")
        ion_points = np.asarray(ion_positions, dtype=float) if i_len > 0 else np.empty((0, 3))

        # 4. Monitoreo de Recombinadas
        r_len = len(recombined_positions) if recombined_positions is not None else 0
        print(f"🔮 [4/4] Recombinados rec.:     {r_len} | Tipo: {type(recombined_positions)}")
        recombined_points = np.asarray(recombined_positions, dtype=float) if r_len > 0 else np.empty((0, 3))

        # Envoltura de seguridad para el reemplazo de actores
        try:
            self._replace_actor("electron", electron_points, "yellow")
            self._replace_actor("neutral", neutral_points, "dodgerblue")
            
            # Rastreamos si el método de reemplazo falla por dentro para los iones
            print("   -> Intentando pintar actor 'ion'...")
            self._replace_actor("ion", ion_points, "orangered")
            print(f"   -> Actor 'ion' procesado. ¿Existe en self._actors?: {'ion' in getattr(self, '_actors', {})}")
            
            self._replace_actor("recombined", recombined_points, "purple")
        except Exception as e:
            print(f"🚨 ERROR CRÍTICO durante el renderizado de actores: {e}")
            import traceback
            traceback.print_exc()

        if self._first_render:
            self._plotter.reset_camera()
            self._first_render = False
        
        try:
            self._plotter.render()
            print("=== ✅ FIN DEL DIAGNÓSTICO (Render ejecutado) ===\n")
        except Exception as e:
            print(f"🚨 ERROR al ejecutar self._plotter.render(): {e}")

    def _replace_actor(self, kind: str, points: np.ndarray, color: str) -> None:
        """Remueve de forma eficiente el actor gráfico anterior y dibuja uno nuevo."""
        actor_attr = f"_{kind}_actor"
        mesh_attr = f"_{kind}_mesh"
        
        # Extraemos el actor actual de la clase con un valor por defecto seguro
        actor = getattr(self, actor_attr, None)
        
        # 1. REMOCIÓN: Si el objeto ya existía en la escena, se destruye su buffer
        if actor is not None:
            try:
                self._plotter.remove_actor(actor)
            except Exception:
                pass

        # 2. LIMPIEZA: Si la matriz viene vacía, reseteamos las propiedades y salimos
        if points is None or points.size == 0:
            setattr(self, mesh_attr, None)
            setattr(self, actor_attr, None)
            return
            
        # 3. CONSTRUCCIÓN: Reconstruimos la PolyData con los nuevos límites de memoria filtrados
        mesh = self._pv.PolyData(points)
        
        # Definimos el tamaño de punto adecuado por especie
        p_size = 7 if kind == "neutral" else (14 if kind == "recombined" else 12)
        
        # Agregamos la nueva geometría a la escena
        new_actor = self._plotter.add_mesh(
            mesh,
            render_points_as_spheres=True,
            point_size=p_size,
            color=color,
            opacity=0.35 if kind == "neutral" else 1.0,
            name=kind  # El parámetro name ayuda a PyVista a optimizar el reemplazo
        )
        
        # Guardamos los nuevos punteros en la instancia de la clase
        setattr(self, mesh_attr, mesh)
        setattr(self, actor_attr, new_actor)

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
            try:
                self._plotter.remove_actor(self._domain_actor)
            except Exception:
                pass
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
        """Limpia por completo la escena eliminando todos los actores móviles."""
        if not self.available:
            return
            
        # Removemos de forma segura todos los actores registrados
        for kind in ["electron", "neutral", "ion", "recombined"]:
            attr = f"_{kind}_actor"
            actor = getattr(self, attr, None)
            if actor is not None:
                try:
                    self._plotter.remove_actor(actor)
                except Exception:
                    pass
                setattr(self, attr, None)
            setattr(self, f"_{kind}_mesh", None)
            
        # Reinyectamos el electrón semilla inicial para el próximo arranque
        self._electron_mesh = self._pv.PolyData(np.zeros((1, 3)))
        self._electron_actor = self._plotter.add_mesh(
            self._electron_mesh,
            render_points_as_spheres=True,
            point_size=12,
            color="yellow",
        )
        self._first_render = True
        self._plotter.render()