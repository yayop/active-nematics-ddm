# Inventario inicial de datos (2026-10-10)

## Qué existía en los árboles de análisis

No se encontraron vídeos/imágenes crudas en `AN-bulk` ni `ParamsExp` con
extensiones AVI, MP4, MOV, MKV, WMV, TIFF, ND2, CZI, LIF, STK, MRC, OME o SER.
`AN-bulk/LocalData` contiene tres MAT derivados de PIVlab (aprox. 1.26 GiB):

| Serie | Archivo | Campos | Malla |
|---|---|---:|---:|
| `20241108_BULK` | `PIVlab2/PIVlab.mat` | 2733 | 91×65 |
| `BULK` | `PIVlab/PIVlab.mat` | 2999 | 91×65 |
| `BULK` | `PIVlab2/PIVlab.mat` | 2999 | 91×65 |

Son campos de velocidad, no intensidad, por lo que no son una entrada válida
para DDM. Sí servirán después para validación independiente.

## Corpus indicado por los resultados existentes

Los resultados locales registran una raíz de NAS interna. Por seguridad, la
ruta/host exactos no se versionan; se representa como:

```text
<NAS_ROOT>/Bulk Active Nematics Videos
```

Se identificaron seis series fuente y variantes PIVlab/PIVlab2:

- `20241106_BULK001` — 2167 campos por variante, ~1083 s;
- `20241108_BULK` — 2733, ~1366 s;
- `20250227` — 2733;
- `BULK` — 2999, ~1499 s;
- `Bulk_1_12_11` — 7199 pares de flujo, ~3599 s;
- `bulk2` — 7199.

La raíz se volvió accesible durante la preparación de la primera corrida. El
host y la ruta exactos siguen sin versionarse.

## Serie con calibración recuperada

El ND2 original de `Bulk_1_12_11` se inspeccionó directamente:

- archivo ND2 de 23.731 GiB, un canal, `T=7200`, `Y=1578`, `X=1120`;
- `uint16` con 12 bits significativos;
- calibración embebida `0.65 µm/px` en x e y;
- timestamps 0–3600.2414 s, paso mediano `0.500102976 s`;
- exposición 500 ms, objetivo 10× NA 0.3 y cámara Andor Zyla;
- la media cae aproximadamente 34% durante la hora.

Los análisis previos usaron `0.7 µm/px` y `0.5 s/frame`. La discrepancia de
escala espacial (~7.7%) debe resolverse antes de combinar DDM con PIV/RAFT; la
corrida DDM usa por ahora el valor embebido de `0.65 µm/px` y registra ambas
decisiones de manera explícita.

La carpeta `ImageSequence` contiene TIFF reescalados que no son equivalentes
pixel a pixel al ND2 y no tiene manifiesto de transformación. La carpeta
`ImageSequence_divide_background_median` sí tiene manifiesto, pero se declara
`uint8_preview`: división por mediana temporal, cuantización a 8 bits y clipping
global 1–99.8%. Sirve para QC visual, no como fuente cuantitativa final.

El flat-field `Backgrounds/background_median.tif` es la mediana de los 7200
frames. El pipeline lo aplica al ND2 en `float32`, sin clipping. La caída global
se trata como análisis de sensibilidad con una curva exponencial suave y se
compara después contra ventanas sin corrección; no se normaliza cada frame.

La configuración piloto reproducible está en
`configs/bulk_1_12_11.example.toml`. El loader ND2 rechaza posiciones, canales o
planos Z ambiguos en vez de concatenarlos silenciosamente.

## Almacenamiento y staging

Además de `C:`, la estación dispone de NVMe `E:` y `G:` casi vacíos y un HDD
`D:` de 3.73 TiB. Para la futura caché FFT disk-backed se recomienda `E:` o
`G:`; conservar los crudos/archivo en `D:` o NAS. No copiar ni versionar datos
crudos dentro de Git.
