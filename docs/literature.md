# Estado del arte y decisiones metodológicas

## Fundamento

DDM transforma diferencias de imágenes a espacio de Fourier:

\[
D(\mathbf q,\Delta t)=
\left\langle\left|\tilde I(\mathbf q,t+\Delta t)-
\tilde I(\mathbf q,t)\right|^2\right\rangle_t.
\]

Para señal estacionaria y formación de imagen aproximadamente lineal,

\[
D(\mathbf q,\Delta t)=A(\mathbf q)
[1-\operatorname{Re}f(\mathbf q,\Delta t)]+B(\mathbf q).
\]

La normalización de `A` varía por un factor dos entre convenciones. Aquí `A`
es simplemente el incremento entre baseline y plateau; las interpretaciones se
basan en tiempos y escalado, no en su valor absoluto.

El método nació con Cerbino y Trappe,
[Phys. Rev. Lett. 100, 188102 (2008)](https://doi.org/10.1103/PhysRevLett.100.188102),
y el papel de la óptica fue desarrollado por Giavazzi et al.,
[Phys. Rev. E 80, 031403 (2009)](https://doi.org/10.1103/PhysRevE.80.031403).
La guía moderna más completa es Lattuada et al.,
[J. Chem. Phys. 163, 161501 (2025)](https://doi.org/10.1063/5.0289471),
acompañada por [fastDDM](https://github.com/somexlab/fastddm).

## Precedentes directamente relevantes

### Active nematic

Molaei et al.,
[Measuring response functions of active materials from data (PNAS, 2023)](https://doi.org/10.1073/pnas.2305283120),
compararon DDM con funciones de respuesta en un nemático activo extensil de
actina–miosina. Ajustaron un exponential stretched/compressed y observaron un
crossover aproximado de `γ ≈ 2` a gran escala a `γ ≈ 1` a escalas menores. La
advertencia central para este proyecto es igualmente importante: DDM responde
a cualquier mecanismo que cambie la intensidad y puede dar tiempos y escalados
distintos de una relajación mecánica.

McGorty et al.,
[Kinesin and myosin motors compete to drive rich multiphase dynamics…
(PNAS Nexus, 2023)](https://doi.org/10.1093/pnasnexus/pgad245),
ofrecen el patrón más cercano para la segunda fase: conservar
`D(qx,qy,Δt)`, medir anisotropía mediante integrales azimutales, validar
velocidades DDM con PIV y usar DDM instantáneo/distribuciones para detectar
eventos intermitentes.

Otros trabajos de materia activa sustentan separar difusión y advección:

- Wilson et al., bacterias,
  [Phys. Rev. Lett. 106, 018101 (2011)](https://doi.org/10.1103/PhysRevLett.106.018101)
  y [Biophys. J. 103, 1520 (2012)](https://doi.org/10.1016/j.bpj.2012.08.045).
- Germain et al., active Brownian/Janus,
  [Phys. Rev. Lett. 121, 078001 (2018)](https://doi.org/10.1103/PhysRevLett.121.078001).
- Drechsler et al., difusión activa y advección intracelular,
  [Nature Communications 8, 15264 (2017)](https://doi.org/10.1038/s41467-017-01414-6).

## Anisotropía y orientación

Son mediciones distintas y no deben confundirse:

1. **DDM direccional:** `D(q,φ,Δt)` y tiempos/velocidades paralelos y
   perpendiculares. Reufer et al.,
   [Langmuir 28, 4618 (2012)](https://doi.org/10.1021/la204904a).
2. **DDM polarizado de cristales líquidos:** bajo geometría y alineamiento
   conocidos, las fluctuaciones del director informan razones viscoelásticas.
   Giavazzi et al.,
   [Soft Matter 10, 3938 (2014)](https://doi.org/10.1039/C4SM00145A).
3. **Squared-gradient DDM:** hace visible relajación rotacional de partículas
   anisotrópicas sub-resolución. Giavazzi et al.,
   [Eur. Phys. J. E 44, 61 (2021)](https://doi.org/10.1140/epje/s10189-021-00063-4).

En una película fluorescente de active nematics, intensity-DDM mezcla densidad,
orientación, advección, defectos y movimiento fuera de plano. La extensión
propuesta —tiles alineados con el director local estimado por tensor de
estructura/Q— debe presentarse como análisis nuevo y validarse por separado.

## Riesgos documentados

- Cruces por los bordes producen una relajación espuria casi independiente de
  `q`; una ventana espacial amplía el rango fiable. Giavazzi et al.,
  [Image windowing mitigates edge effects… (2017)](https://doi.org/10.1140/epje/i2017-11587-3).
- FPS/exposición insuficientes pierden relajaciones rápidas; duración/FOV
  insuficientes impiden observar el plateau a bajo `q`.
- Drift, photobleaching, autoexposure, clipping y codecs con pérdida pueden
  simular dinámica o romper la respuesta lineal.
- En un “bulk” 3D, movimiento fuera de plano y profundidad de campo añaden otra
  decorrelación.
- El promedio azimutal puede borrar precisamente la anisotropía nemática.
- `A(q)` contiene OTF y form factor; no es automáticamente un factor de
  estructura termodinámico.
- `γ≈2` o `Γ∝q` no demuestran por sí solos balística. Se exige escalado,
  colapso, residuales y validación PIV coherentes.
- Los errores de lags están correlacionados. Gu et al. desarrollaron DDM-UQ,
  [Phys. Rev. E 104, 034610 (2021)](https://doi.org/10.1103/PhysRevE.104.034610);
  aquí se comenzará con bootstrap temporal por bloques y entre vídeos.
