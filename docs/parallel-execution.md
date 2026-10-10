# Ejecución paralela y memoria

## Hardware observado

- 2 × Intel Xeon Silver 4214R, 12 cores cada uno.
- 24 cores físicos / 48 hilos lógicos.
- 64 GB RAM.
- NVIDIA Quadro P400, 2 GB VRAM.
- Cuatro NVMe de ~477 GB (`E:` y `G:` casi vacíos) y un HDD `D:` de 3.73 TB;
  `C:` tenía aproximadamente 237 GB libres al arrancar el proyecto.

El MVP carga los frames seleccionados como `float32` y conserva una FFT
`complex64` por frame. Una estimación conservadora antes de buffers es:

\[
M\approx N_t N_y N_x(4+4)\ \text{bytes},
\]

porque la RFFT ocupa aproximadamente otros 4 bytes por pixel original. Los
buffers de pares y Python elevan el pico; se recomienda que esta estimación no
supere 25–30 GB en esta estación.

## Dos niveles, nunca anidados

### Un vídeo grande

`an-ddm run config.toml` paraleliza lags con memoria FFT compartida entre
threads. Probar 4, 8 y 12 workers; más threads pueden empeorar por ancho de
banda de memoria y NUMA.

### Varios vídeos/ROIs

```powershell
an-ddm batch configs\local.a.toml configs\local.b.toml `
  configs\local.c.toml configs\local.d.toml --jobs 4
```

Cada proceso usa un único worker interno y limita BLAS/FFT a un thread. Esto
evita oversubscription y mantiene un único escritor por directorio de salida.

## Benchmark

```powershell
python benchmarks\benchmark_local.py | Tee-Object benchmark.csv
```

Elegir el mínimo tiempo estable, no el máximo número de workers. Monitorizar
RAM y disco durante un vídeo representativo. Para películas que no caben:

1. aplicar ROI físicamente justificada o binning espacial;
2. analizar ventanas temporales estacionarias;
3. implementar el milestone de caché FFT disk-backed/Zarr antes de procesar el
   dataset completo, preferentemente con scratch en `E:` o `G:`.

fastDDM se usará como referencia de exactitud/rendimiento. La P400 puede servir
para una prueba, pero 2 GB de VRAM no justifican hacer de CUDA el backend
principal; decodificación, DDM radial, fits y PIV deben repartirse por CPU.
