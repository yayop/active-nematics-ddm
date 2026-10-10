# Active Nematics DDM

Pipeline reproducible de **Differential Dynamic Microscopy (DDM)** para vídeos
de microscopía de nemáticos activos. El objetivo inicial es medir decorrelación
espacio-temporal de la textura —no inferir automáticamente una velocidad, un
MSD o una relajación mecánica— y contrastarla con PIV/optical flow.

El repositorio contiene un MVP funcional que:

- lee TIFF, NumPy y formatos de vídeo soportados por ImageIO/FFmpeg;
- calcula una FFT por frame y reutiliza el resultado para lags logarítmicos;
- evalúa
  \(D(q,\Delta t)=\langle|\tilde I(q,t+\Delta t)-\tilde I(q,t)|^2\rangle_t\);
- promedia radialmente con calibración explícita en rad µm⁻¹ y segundos;
- ajusta \(B+A[1-\exp(-(\Delta t/\tau)^\gamma)]\) y reporta también
  \(\tau_{63}\) empírico;
- guarda arrays, tabla de ajustes, procedencia, hash del input y figuras QC;
- paraleliza lags dentro de un vídeo o vídeos completos mediante procesos.

La extensión angular, DDM local, correlación compleja y DDM del tensor nemático
\(Q\) están diseñadas en la documentación, pero todavía no se presentan como
implementadas. Véase [estado y roadmap](docs/measurement-plan.md).

## Inicio rápido en Windows

```powershell
cd <ruta-al-proyecto>\active-nematics-ddm
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
an-ddm demo data\demo --run
pytest
```

Para recrear exactamente el entorno validado en esta estación, usar
`requirements-dev.lock.txt` y después instalar el proyecto editable con
`python -m pip install -e . --no-deps`. El `pyproject.toml` mantiene rangos más
amplios para probar compatibilidad en CI.

El demo crea un TIFF sintético, su configuración y un análisis completo. Para
un vídeo real:

```powershell
Copy-Item configs\bulk.example.toml configs\local.bulk.toml
# Editar ruta y calibraciones en configs\local.bulk.toml
an-ddm inspect "D:\datos\bulk.tif"
an-ddm run configs\local.bulk.toml
```

Las calibraciones `pixel_size_um` y `frame_interval_s` son obligatorias y deben
provenir del experimento, no asumirse a partir del contenedor de vídeo. Los
vídeos y resultados están excluidos de Git.

## Paralelismo recomendado en esta estación

La estación detectada tiene 24 núcleos físicos/48 hilos, 64 GB de RAM y una
Quadro P400 de 2 GB. El camino útil es CPU-first:

- un solo vídeo: empezar con `workers = 8`, después medir 4/8/12 con
  `benchmarks/benchmark_local.py`;
- varios vídeos: crear un TOML por vídeo/ROI y ejecutar
  `an-ddm batch configs/run1.toml configs/run2.toml --jobs 4`;
- `batch` fija un worker de lag por proceso para evitar pools anidados;
- reservar aproximadamente 20 GB para el sistema y limitar crops/ventanas para
  que frames + FFTs + buffers no superen unos 40 GB.

No se recomienda invertir trabajo en CUDA para la P400. El extra opcional
`.[fast]` instala [fastDDM](https://github.com/somexlab/fastddm) como backend de
referencia/benchmark futuro; el MVP actual es NumPy/SciPy independiente.

## Productos de una ejecución

```text
results/<run>/
├─ source_config.toml
├─ resolved_config.json
├─ provenance.json
├─ ddm_radial.npz
├─ fits.csv
└─ overview.png
```

`origin_sem` es sólo un diagnóstico entre orígenes temporales muestreados. Los
pares se solapan y no son réplicas independientes; la inferencia final requiere
bootstrap temporal por bloques y, sobre todo, réplicas de vídeo independientes.

## Propuesta científica resumida

1. Validar calibración, estacionariedad, exposición, drift, clipping y rango
   óptico útil de cada vídeo.
2. Medir \(D(q_x,q_y,\Delta t)\), su promedio radial, \(\tau_{63}(q)\),
   \(\tau(q)\), \(\gamma(q)\) y \(\Gamma(q)=1/\tau(q)\).
3. Comparar colapsos en \(q\Delta t\) y \(q^2\Delta t\), estimar el crossover
   \(q^*\) y reportar \(\ell_{DDM}=2\pi/q^*\) como *longitud dinámica de
   crossover*, no como active length sin validación.
4. Añadir DDM angular y local alineado al director; cuantificar el segundo
   armónico, \(\tau_\parallel/\tau_\perp\), heterogeneidad e intermitencia.
5. Validar cualquier \(v_\mathrm{eff}\) o \(D_\mathrm{eff}\) contra PIV/optical
   flow y comparar \(\ell_{DDM}\) con correlaciones de velocidad/vorticidad y
   spacing de defectos.

La justificación, referencias y criterios de aceptación están en
[revisión de literatura](docs/literature.md),
[plan de medición](docs/measurement-plan.md) y
[ejecución paralela](docs/parallel-execution.md). El
[inventario inicial](docs/data-inventory.md) documenta dónde están los datos y
qué calibración ya se pudo recuperar.

## Licencia

MIT. Las publicaciones que utilicen el software deben citar además los métodos
DDM originales y las referencias específicas indicadas en `CITATION.cff` y en
la revisión de literatura.
