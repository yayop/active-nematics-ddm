# Inventario inicial de datos (2026-10-10)

## Qué existe localmente

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

El NAS no respondió durante una prueba de 30 s. Por tanto el repositorio y el
demo están completos, pero el primer análisis físico queda a la espera de que
esa ruta sea visible/montada.

## Serie con calibración recuperada

`ParamsExp/results/.../summary.json` documenta para `Bulk_1_12_11`:

- secuencia: `ImageSequence_divide_background_median`;
- 7200 frames de orientación / 7199 pares de flujo;
- frame 1120×1578 px;
- 0.7 µm/px;
- 0.5 s/frame (2 fps);
- FOV 784×1104.6 µm;
- duración aproximada 60 min;
- paso RAFT 12 px (8.4 µm) y orientación 10 px (7 µm).

La extensión de los archivos individuales no consta en los artefactos locales.
El loader acepta un directorio con TIFF/PNG/JPEG/BMP/NPY y orden natural. La
configuración piloto reproducible está en
`configs/bulk_1_12_11.example.toml`.

## Almacenamiento y staging

Además de `C:`, la estación dispone de NVMe `E:` y `G:` casi vacíos y un HDD
`D:` de 3.73 TiB. Para la futura caché FFT disk-backed se recomienda `E:` o
`G:`; conservar los crudos/archivo en `D:` o NAS. No copiar ni versionar datos
crudos dentro de Git.
