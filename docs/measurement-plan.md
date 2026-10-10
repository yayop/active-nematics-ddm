# Plan de medición para los vídeos bulk

## Pregunta y unidad experimental

Pregunta inicial: **¿cómo cambia la escala temporal de decorrelación de la
textura activa con longitud de onda, dirección, tiempo y condición
experimental?** La unidad experimental es un vídeo/realización independiente;
los frames y modos Fourier no son réplicas biológicas.

## Tier 1: resultados model-light

Para cada vídeo, ROI y ventana temporal:

1. `D(qx,qy,Δt)` y `D(q,Δt)`, con número de pares y modos por bin.
2. Baseline `B(q)`, plateau/amplitud `A(q)` y máscara explícita de calidad.
3. `τ63(q)` empírico; después `τ(q)` y `γ(q)` del modelo fenomenológico.
4. `Γ(q)=1/τ(q)`, pendiente local y pruebas de colapso en `qΔt` y `q²Δt`.
5. Crossover `q*` y `ℓDDM=2π/q*`, denominado longitud dinámica de crossover.
6. Sensibilidad a crop, ventana, duración, lags, corrección de drift y rango q.

Puerta de aceptación: no ajustar un bin q si no tiene SNR suficiente y una
fracción sustancial de baseline–plateau. El rango final se fija desde QC, no
desde el aspecto deseado de la ley de potencias.

## Tier 2: anisotropía, heterogeneidad y director

- Mantener el mapa Fourier completo y calcular `D(q,φ,Δt)`.
- Segundo armónico complejo
  \(C_2=\int D e^{2i\phi}d\phi/\int Dd\phi\), amplitud `|C2|` y dirección
  `arg(C2)/2`.
- Dividir en tiles, estimar director y coherencia local mediante tensor de
  estructura, y acumular en el marco `(q∥,q⊥)` sin rotar toda la película.
- Construir `Qxx=S cos(2θ)`, `Qxy=S sin(2θ)` y medir una función DDM
  orientacional separada de intensity-DDM.
- DDM en ventanas deslizantes e instantaneous-DDM para distribuciones,
  skewness y susceptibilidad de heterogeneidad.

Puerta de aceptación: imágenes sintéticas rotadas/trasladadas recuperan módulo,
eje y periodicidad π; resultados globales y locales se distinguen claramente.

## Tier 3: interpretación física validada

Sólo en un intervalo donde scaling, data collapse y residuales coincidan:

- si `Γ≈v_eff q`, estimar `v_eff` y compararlo con PIV/optical flow;
- si `Γ≈D_eff q²`, estimar `D_eff` como parámetro efectivo;
- comparar uno/dos modos o advección–difusión con AIC/CV, sin imponerlos;
- comparar `ℓDDM` con longitud de correlación de velocidad/vorticidad y spacing
  de defectos.

DDM por sí solo no identifica `K`, actividad `α`, `sqrt(K/|α|)`, tasa de
defectos ni relajación pura del director.

## Metadatos que faltan antes del primer resultado físico

Por vídeo: tamaño de pixel calibrado, timestamps/FPS real, exposure, canal,
bit-depth, objetivo/NA, espesor/profundidad de campo, condición, réplica y
tratamiento. También conviene conservar dark/flat frames si existen.

## Controles y validación

- Estático, ruido blanco y frames barajados.
- Modos OU con `f(q,t)` conocido.
- Difusión browniana (`Γ=Dq²`).
- Traslación/advección, advección+difusión y dos velocidades.
- Textura/director rotante con periodicidad π.
- Resultado raw frente a drift-corrected; el registro puede borrar flujo real.
- Resultado `photobleach_method="none"` frente a `"global-exponential"`, con
  background medido independientemente y revisión de `photobleach_trace.csv`.
- Comparación con PIV/optical flow ya disponible en `AN-bulk`.

Objetivo de validación sintética: error <5% en el rango declarado válido. Para
datos reales: intervalos de confianza por bootstrap de bloques y resumen entre
réplicas, más un análisis de sensibilidad predefinido.

## Roadmap verificable

| Fase | Entrega | Criterio para avanzar |
|---|---|---|
| 0 | inventario y calibración | manifest completo, vídeos accesibles |
| 1 | MVP radial (actual) | tests, demo y un vídeo piloto |
| 2 | QC + bootstrap + benchmark | rango q/t defendible, reproducibilidad |
| 3 | batch de condiciones | artefacto independiente por vídeo |
| 4 | DDM angular/complejo | validación con traslaciones sintéticas |
| 5 | DDM de Q/local-director | validación π y comparación de director |
| 6 | PIV/defectos y estadística | concordancia y réplica experimental |
