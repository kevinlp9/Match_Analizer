# ⚽ Geo-Goal Local — Análisis Táctico de Fútbol mediante Homografía

Sistema **100% offline** de análisis táctico de fútbol que transforma video de partidos en datos métricos mediante **homografía proyectiva (DLT + SVD)**, genera visualizaciones 2D interactivas y renders 3D en Blender.

---

## 🎯 ¿Qué hace?

1. **Detecta** jugadores y balón con YOLOv8
2. **Rastrea** con ByteTrack (IDs persistentes)
3. **Proyecta** coordenadas de píxeles a metros reales (homografía 3×3)
4. **Interpola** posiciones faltantes (lineal + dead-reckoning)
5. **Calcula** estadísticas avanzadas (velocidad, sprints, Voronoi, posesión, formación)
6. **Genera** un HTML interactivo con vista táctica 2D + algoritmos gráficos a mano
7. **Renderiza** una escena 3D cinemática en Blender

---

## 🚀 Inicio rápido

### Requisitos

- Python 3.10+
- Blender 4.x (opcional, solo para el render 3D)

### Instalación

```bash
pip install -r requirements_local.txt
```

### Uso (GUI — recomendado)

```bash
python3 -m geogoal_local.gui
```

Abre `http://localhost:8888` en tu navegador. Flujo:

1. **Subir video** (mp4/avi/mov)
2. **Calibrar** — marcar ≥4 puntos de la cancha (usa "Pre-detectar campo" para ayuda automática)
3. **Procesar** — detección + tracking + homografía + interpolación
4. **Ver resultados** — HTML táctico, JSON de datos, seguimiento en vivo

### Uso (CLI)

```bash
# Calibración interactiva
python3 -m geogoal_local.cli select mi_video.mp4

# Procesar video
python3 -m geogoal_local.cli process mi_video.mp4 --calib output/calib.json

# Generar reporte HTML
python3 -m geogoal_local.cli report

# Render 3D en Blender
python3 -m geogoal_local.cli blender
```

---

## 📐 Calibración — ¿Qué puntos marcar?

El sistema necesita **≥4 puntos** visibles en el frame, asociados a coordenadas reales del campo (105×68 m):

```
        0m                    52.5m                   105m
   0m   ┌────────────────────┬─────────────────────────┐  ← línea lejana
        │                    │                          │
        │   ┌──┐             │              ┌──┐       │
13.84m  │   │  │─────────────│──────────────│  │       │
   34m  │   │  │   ● penal   │    ● penal   │  │       │
54.16m  │   │  │─────────────│──────────────│  │       │
        │   └──┘             │              └──┘       │
        │                    │                          │
  68m   └────────────────────┴─────────────────────────┘  ← línea cercana
      portería izq                              portería der
```

| Landmark | Coordenadas | Descripción |
|---|---|---|
| Top-left corner | (0, 0) | Esquina campo, lejana-izquierda |
| Top-right corner | (105, 0) | Esquina campo, lejana-derecha |
| Bottom-left corner | (0, 68) | Esquina campo, cercana-izquierda |
| Bottom-right corner | (105, 68) | Esquina campo, cercana-derecha |
| Halfway top | (52.5, 0) | Medio campo × línea lejana |
| Halfway bottom | (52.5, 68) | Medio campo × línea cercana |
| Center spot | (52.5, 34) | Punto central |
| Left PA top | (0, 13.84) | Área penal izq, esquina lejana |
| Left PA bottom | (0, 54.16) | Área penal izq, esquina cercana |
| Right PA top | (105, 13.84) | Área penal der, esquina lejana |
| Right PA bottom | (105, 54.16) | Área penal der, esquina cercana |

**Tip:** Las esquinas del área penal y las intersecciones del medio campo son las más fáciles de identificar.

---

## 📁 Estructura del proyecto

```
geogoal_local/
├── __init__.py
├── cli.py                 # CLI: select | process | report | blender | gui
├── gui.py                 # GUI web (FastAPI, localhost:8888)
├── video_processor.py     # Pipeline principal (YOLO, ByteTrack, DLT, etc.)
├── analytics.py           # 15 estadísticas avanzadas
├── report.py              # Generador de HTML autocontenido
├── selector/
│   ├── __init__.py
│   └── server.py          # Selector de calibración (localhost:8989)
└── blender/
    ├── __init__.py
    ├── build_scene.py     # Script bpy para escena 3D
    └── assets/            # Texturas (césped, balón)

output/                    # Resultados generados
├── calib.json             # Calibración (puntos src/dst)
├── match_data.json        # Datos de frames procesados
├── stats.json             # Estadísticas calculadas
├── report.html            # Vista táctica interactiva
├── scene.blend            # Escena Blender (abrible en vivo)
└── render.mp4             # Render 3D del partido

docs/
├── REPORTE.md             # Reporte académico completo
└── architecture.md        # Diagramas de arquitectura
```

---

## 🧮 Matemática del núcleo

### Homografía DLT (Direct Linear Transform)

Dado un conjunto de ≥4 correspondencias punto-a-punto:
- **src** = coordenadas en píxeles del frame `[u, v]`
- **dst** = coordenadas reales en metros `[x, y]`

Se calcula una matriz **H (3×3)** tal que:

```
       ┌    ┐       ┌         ┐   ┌   ┐
       │ x' │       │ h1 h2 h3│   │ u │
   s · │ y' │   =   │ h4 h5 h6│ · │ v │
       │ 1  │       │ h7 h8 h9│   │ 1 │
       └    ┘       └         ┘   └   ┘
```

Resolución por **SVD** (descomposición en valores singulares) del sistema de ecuaciones homogéneas `Ah = 0`.

### Pipeline de procesamiento

```
Video → Detección (YOLO) → Tracking (ByteTrack) → Filtro de cancha (HSV mask)
  → Clasificación equipos (K-Means HSV) → Homografía (DLT/RANSAC)
  → Transformación de perspectiva → Interpolación → Estadísticas → Salidas
```

---

## 📊 Estadísticas calculadas

| Métrica | Descripción |
|---|---|
| Distancia total | Metros recorridos por jugador |
| Velocidad media/máxima | km/h |
| Sprints | Tramos sostenidos >25 km/h |
| Mapa de calor (KDE) | Densidad de posiciones por jugador |
| Voronoi | Control de espacio por equipo (%) |
| Envolvente convexa | Compactación del equipo |
| Posesión | Jugador más cercano al balón → equipo |
| Centroide | Trayectoria del centro del equipo |
| Amplitud/Profundidad | Extensión lateral y longitudinal |
| Línea defensiva | Altura media de los 4 defensas |
| Presión | Rivales dentro de radio 5m |
| Formación | Clustering → etiqueta tipo 4-4-2 |

---

## 🎨 Graficación por computadora

| Criterio | Implementación |
|---|---|
| **Transformaciones** | Homografía proyectiva; pan/zoom en Canvas 2D; animación por keyframes en Blender |
| **Iluminación** | Sun lamp + Area lights en Blender; heatmap como campo emisivo en 2D |
| **Texturas** | Materiales UV en Blender (césped, camisetas, balón); césped procedimental en Canvas |
| **Algoritmos gráficos** | Voronoi, envolvente convexa (Andrew), KDE, splines Catmull-Rom — todos a mano en JS |

---

## ⚙️ Configuración avanzada

### Variables de entorno (opcionales)

```bash
GEOGOAL_PORT=8888          # Puerto del GUI
GEOGOAL_DEVICE=cpu         # Dispositivo YOLO (cpu/cuda/mps)
GEOGOAL_FRAME_SKIP=4       # Procesar 1 de cada N frames
```

### Parámetros del procesador

En `video_processor.py`:
- `ObjectDetector`: conf=0.20, retry adaptativo a 0.10+1280px
- `ByteTrack`: lost_track_buffer=120, activation=0.15
- `TrackInterpolator`: max_gap=30 frames, predict=15 frames
- `PitchMaskDetector`: actualiza máscara cada 50 frames

---

## 📝 Licencia

Proyecto académico — Universidad.

---

## 👤 Autor

Kevin Atilano Gutiérrez
