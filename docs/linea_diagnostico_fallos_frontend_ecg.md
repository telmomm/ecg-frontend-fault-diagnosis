# Línea de investigación: verificación de especificaciones y diagnóstico de fallos en servicio de front-ends de ECG mediante simulación y aprendizaje automático

> **Versión 2.** Reorientada tras el estado del arte (ver `sota_diagnostico_fallos_frontend_ecg.md`). Toda la línea se basa en datos de simulación; la validación en banco queda como fase futura.
>
> Repositorio previsto: `ecg-frontend-fault-diagnosis`

---

## 0. Qué cambia respecto a la versión 1

| Aspecto | Versión 1 | Versión 2 | Motivo |
|---|---|---|---|
| **Aportación central** | Diagnóstico de fallos con ML en un dominio nuevo | **Puente entre el test orientado a especificaciones y el diagnóstico con ML**, aplicado al autodiagnóstico en servicio de un equipo biomédico | La idea de juzgar el fallo por su efecto en las especificaciones ya existe en el test de circuitos integrados (*specification-based test*, *alternate test*); lo nuevo es combinarla con la localización de componentes y aplicarla en servicio a un front-end biomédico |
| **Circuito** | Amplificador de instrumentación discreto de tres operacionales | **Front-end integrado + red de componentes discretos** que lo rodea; el circuito discreto queda como caso de referencia | Los equipos reales usan front-ends integrados con autodiagnóstico propio; los componentes que fallan y no están cubiertos son los discretos externos |
| **Etiquetado** | Componente y magnitud del fallo | **Tres niveles**: cumplimiento de especificaciones, localización y origen (electrodo o circuito) | Responde a lo que necesita saber un equipo médico: si sigue siendo apto y por qué no |
| **Generalización y conjunto mínimo de pruebas** | Presentados como aportación | Presentados como **evaluación rigurosa**, posicionada frente a la literatura existente | Ya existen trabajos específicos (transferencia entre severidades, selección de puntos de prueba con aprendizaje por refuerzo) |
| **Precedentes biomédicos** | No identificados | Dos precedentes débiles (electroencefalógrafo, equipos médicos a nivel de placa) | Hay que citarlos y diferenciarse |

---

## 1. Pregunta de investigación y posicionamiento

**Pregunta principal:** con las medidas que el propio equipo puede adquirir (su ADC y una señal de prueba interna), ¿se puede determinar en servicio si el front-end analógico de un electrocardiógrafo **sigue cumpliendo sus especificaciones clínicas** y, cuando no las cumple, **localizar la causa** y **distinguir si está en el circuito o en los electrodos**?

### El hueco: dos comunidades que no se hablan

| | Test orientado a especificaciones / *alternate test* | Diagnóstico de fallos con ML |
|---|---|---|
| **Contexto** | Test de producción de circuitos integrados | Investigación académica, mantenimiento |
| **Pregunta** | ¿Cumple el circuito sus especificaciones? | ¿Qué componente ha fallado? |
| **Gravedad del fallo** | Por su efecto en las especificaciones | Por el porcentaje de desviación del componente |
| **Localización** | No | Sí |
| **Circuitos habituales** | Amplificadores, convertidores, RF integrados | Filtros de referencia (Sallen-Key, biquad de cuatro operacionales, leapfrog) |
| **Dominio biomédico** | No | Prácticamente no |

**Este trabajo une ambos enfoques** en un caso de uso donde los dos importan: un equipo médico que debe saber si sigue siendo apto (especificaciones) y qué hay que reparar (localización), y que además debe separar los problemas del circuito de los del electrodo.

---

## 2. Hipótesis

- **H1.** Las especificaciones clínicas del front-end (ganancia, ancho de banda, rechazo en modo común) pueden **predecirse** a partir de medidas autoadquiribles con un error suficiente para decidir apto/no apto con tasas bajas de escape y de falso rechazo.
- **H2.** Muchas desviaciones de componentes dentro o algo fuera de tolerancia **no comprometen** las especificaciones; clasificarlas como fallo produce falsas alarmas. La gravedad funcional reduce las falsas alarmas frente a la gravedad por porcentaje.
- **H3.** Cuando hay incumplimiento, la localización está limitada por **grupos de ambigüedad** identificables a priori.
- **H4.** La degradación del electrodo y los fallos del circuito producen firmas distinguibles con las mismas medidas.
- **H5.** Un circuito realista (front-end integrado con red discreta) presenta una estructura de ambigüedad distinta a la de los circuitos de referencia de la literatura, lo que justifica evaluar el diagnóstico en el dominio de aplicación.

---

## 3. Aportaciones

**Lo que el trabajo aporta:**

1. **Integración de verificación de especificaciones y localización** en un esquema de dos niveles para autodiagnóstico en servicio.
2. **Primer estudio sistemático basado en simulación** (Monte Carlo, fallos duros y paramétricos) sobre un front-end de biopotencial con arquitectura realista.
3. **Separación entre degradación del electrodo y fallo del circuito**, con modelos de interfaz electrodo-piel para electrodos de gel y secos.
4. **Comparación de la gravedad funcional frente a la gravedad por porcentaje**, cuantificando su efecto sobre las falsas alarmas.
5. **Dataset y código abiertos** de fallos en un front-end biomédico.

**Lo que no se presenta como aportación** (se usa y se cita):

- Los algoritmos de clasificación.
- La idea de evaluar el circuito por sus especificaciones (*alternate test*).
- La generalización entre magnitudes de fallo y la selección de puntos de prueba como métodos.

---

## 4. Alcance

**Dentro:**
- Un front-end de ECG de una derivación con arquitectura realista y un circuito discreto de referencia.
- Fallos duros, paramétricos, degradación de condensadores y fallos de electrodo.
- Variación por tolerancias mediante Monte Carlo.
- Medidas restringidas a lo autoadquirible.
- Fallo simple (un componente por simulación).

**Fuera (trabajos posteriores):**
- Validación en banco (sim-to-real).
- Fallos múltiples y compuestos.
- EMG y EEG.
- Envejecimiento continuo y mantenimiento predictivo.
- Implementación en microcontrolador.

---

## 5. Circuito bajo estudio

### 5.1 Circuito principal: front-end integrado + red discreta

| Bloque | Implementación en simulación | Componentes con fallos inyectados |
|---|---|---|
| **Amplificador de instrumentación integrado** | Macromodelo SPICE del fabricante si está disponible (por ejemplo, de la familia INA33x o similar); si no, modelo de comportamiento con ganancia, ancho de banda, rechazo en modo común y offset finitos | Resistencia de ganancia externa; opcionalmente degradación global del bloque |
| **Protección de entrada** | Resistencias serie en cada entrada | Resistencias de protección |
| **Filtro RFI de entrada** | Condensadores diferencial y de modo común | Condensadores (incluido el desequilibrio entre los de modo común, que degrada el rechazo en modo común) |
| **Acoplamiento AC / paso alto** | RC o lazo servo de continua | R y C del paso alto |
| **Pierna derecha activa** | Operacional discreto con red RC | Resistencias y condensador de la red |
| **Filtro antialiasing** | Sallen-Key de 2.º orden | R y C del filtro |
| **Referencia / polarización** | Divisor o referencia de tensión | Resistencias del divisor |

Tamaño objetivo: unos 15–25 componentes discretos con fallos inyectables.

### 5.2 Circuito de referencia

El amplificador discreto de tres operacionales con pierna derecha activa y filtros de la versión 1. Sirve para:
- comparar con la literatura clásica,
- comprobar H5 (estructura de ambigüedad distinta según la arquitectura).

### 5.3 Interfaz electrodo-piel

- Modelo por electrodo: resistencia serie + paralelo RC + potencial de media celda.
- Dos familias de parámetros: **electrodos de gel** y **electrodos secos** (rangos de la literatura reciente; para secos, resistencia serie de decenas de Ω a ~100 kΩ, resistencia paralelo de ~80 kΩ a varios MΩ y capacidad de unidades a decenas de nF).
- Desequilibrio entre electrodos como variable, porque convierte interferencia de modo común en diferencial.

---

## 6. Especificaciones clínicas

Las especificaciones definen la **gravedad funcional**. Fijar los valores a partir de las normas de electrocardiógrafos (IEC 60601-2-25 para diagnóstico e IEC 60601-2-27 para monitorización; **verificar ediciones vigentes y valores exactos**) y documentarlos en el artículo.

| Especificación | Ejemplo de criterio (a confirmar con la norma) |
|---|---|
| Ganancia | Error relativo máximo respecto al nominal |
| Ancho de banda | Límites inferior y superior: modo diagnóstico (del orden de 0,05–150 Hz) y modo monitorización (más estrecho) |
| Rechazo en modo común | Mínimo en presencia de desequilibrio de electrodos |
| Tolerancia a offset de electrodo | Funcionamiento correcto con offsets de continua del orden de cientos de mV |
| Ruido referido a la entrada | Máximo en la banda |
| Respuesta al pulso de calibración | Amplitud y forma dentro de tolerancia |

Cada caso simulado se etiqueta como **apto** o **no apto**, indicando **qué especificación** incumple.

---

## 7. Modelo de fallos y etiquetado

### 7.1 Fallos

| Tipo | Modelado | Niveles |
|---|---|---|
| Duro: abierto | Resistencia muy alta en serie | 1 |
| Duro: corto | Resistencia muy baja en paralelo | 1 |
| Paramétrico | Desviación fuera de tolerancia | ±5 %, ±10 %, ±20 %, ±50 % (incluir niveles pequeños es clave para H2) |
| Degradación de condensador | Pérdida de capacidad + aumento de ESR | 2–3 |
| Bloque integrado (opcional) | Offset, ganancia, rechazo en modo común degradados | 2 |
| Electrodo | Desconexión, impedancia alta (secado del gel), desequilibrio | 3–4 |

Variación normal: Monte Carlo con resistencias al 1 % y condensadores al 5–10 % sobre todos los componentes.

### 7.2 Etiquetado en tres niveles

1. **Funcional:** apto / no apto (+ especificación incumplida).
2. **Localización:** componente o grupo de ambigüedad.
3. **Origen:** ninguno / circuito / electrodo.

Se guardan además las **especificaciones calculadas** de cada caso (valores continuos), para poder entrenar modelos de regresión al estilo del *alternate test*.

---

## 8. Medidas autoadquiribles

| Conjunto | Características |
|---|---|
| **C1. Continua** | Tensión de salida y de los nodos que el equipo puede leer |
| **C2. Frecuencia** | Ganancia (y fase) a 4–6 frecuencias; estimación del rechazo en modo común inyectando señal común por la pierna derecha |
| **C3. Tiempo** | Respuesta completa al pulso de calibración de 1 mV |
| **C4. Electrodo** | Medida de impedancia de contacto tipo *lead-off* en continua o en alterna, como haría el equipo |

Antes del entrenamiento se añaden ruido y cuantificación de un ADC típico, con el nivel de ruido como variable del estudio.

---

## 9. Ensayos

| Ensayo | Qué se hace | Hipótesis |
|---|---|---|
| **E1. Validación nominal** | Nominal + Monte Carlo sin fallos; comprobar que el circuito cumple todas las especificaciones | — |
| **E2. Testabilidad y ambigüedad** | Sensibilidad de cada medida a cada componente; grupos de ambigüedad en ambos circuitos | H3, H5 |
| **E3. Predicción de especificaciones** | Regresión de cada especificación a partir de C1–C4; decisión apto/no apto | H1 |
| **E4. Gravedad funcional frente a porcentual** | Comparar falsas alarmas y escapes al etiquetar por especificaciones o por desviación | H2 |
| **E5. Localización** | Clasificadores sobre los casos no aptos | H3 |
| **E6. Electrodo frente a circuito** | Clasificación del origen | H4 |
| **E7. Generalización y robustez** | Magnitudes no vistas, otras tolerancias, ruido y cuantificación | — |
| **E8. Conjunto mínimo de medidas** | Selección de características; coste frente a rendimiento | — |
| **E9. Efecto de la arquitectura** | Repetir E2 y E5 en el circuito de referencia; separabilidad entre clases | H5 |

### Métricas

- **Tasa de escape** (no apto clasificado como apto): la métrica crítica en un equipo médico.
- **Tasa de falso rechazo** (apto clasificado como no apto): equivale a las falsas alarmas.
- Error de predicción de cada especificación (RMSE relativo).
- F1 macro y matrices de confusión por grupo de ambigüedad para la localización.
- Separabilidad entre clases (por ejemplo, distancia entre centroides) para E9.
- Intervalos de confianza por validación cruzada repetida o bootstrap.

### Protocolo de evaluación

- Separar entrenamiento y prueba por condición y magnitud de fallo en E7.
- Hiperparámetros solo con validación.
- Semillas fijas y publicadas.

---

## 10. Datos, cómputo y herramientas

- Volumen estimado: ~25 componentes × ~10 condiciones + fallos de electrodo, con 200–500 réplicas Monte Carlo por condición, para dos circuitos: **del orden de 100.000–200.000 simulaciones**. Viable en horas con ngspice y paralelización.
- Formato: Parquet o HDF5 con metadatos y especificaciones calculadas.

| Herramienta | Uso |
|---|---|
| ngspice + PySpice | Simulación y automatización |
| Qucs-S | Captura del esquema |
| NumPy, pandas | Procesado |
| scikit-learn | Modelos clásicos, regresión, selección de características |
| PyTorch | CNN 1D |
| Git + Zenodo | Código y dataset con DOI |

---

## 11. Fases de desarrollo

Marcar cada casilla al completarla. Cada fase tiene un entregable y un criterio de cierre.

### Fase 0 — Estado del arte ✅
- [x] Revisión inicial con búsqueda web
- [x] Revisión con Consensus
- [x] Búsquedas B1–B6 en OpenAlex y seguimiento de citas
- [x] Análisis del corpus (1.197 trabajos, 976 con resumen)
- [x] Resumen del SOTA (`sota_diagnostico_fallos_frontend_ecg.md`)

**Entregable:** resumen del SOTA. **Cierre:** hueco reformulado.

### Fase 1 — Lecturas clave y posicionamiento
- [ ] Leer Chen et al. (2025), IEEE TIM
- [ ] Leer Dieste-Velasco (2025), Integration
- [ ] Leer el trabajo del electroencefalógrafo (2025) y el de LSTM en equipos médicos (2021)
- [ ] Leer la revisión de test analógico de 2010 y los dos trabajos fundacionales de *alternate test* (1998)
- [ ] Leer el trabajo de ITC 2022 sobre test estadístico y ML
- [ ] Leer DiffDA-Net (2026)
- [ ] Redactar borrador de introducción y estado del arte (2–3 páginas)

**Entregable:** borrador de introducción. **Cierre:** las cinco aportaciones están justificadas con referencias.

### Fase 2 — Diseño del circuito y especificaciones
- [ ] Elegir el front-end integrado y conseguir su macromodelo (o definir el modelo de comportamiento)
- [ ] Diseñar la red discreta con valores justificados
- [ ] Diseñar el circuito discreto de referencia
- [ ] Definir el modelo de electrodo (gel y seco) con rangos de la literatura
- [ ] Consultar las normas y fijar la tabla de especificaciones
- [ ] Capturar ambos esquemas en Qucs-S

**Entregable:** esquemas y tabla de especificaciones. **Cierre:** E1 superado (circuitos nominales aptos).

### Fase 3 — Pipeline de simulación
- [ ] Automatizar con PySpice un filtro sencillo (prueba de concepto)
- [ ] Generación paramétrica de netlists con inyección de fallos
- [ ] Monte Carlo de tolerancias
- [ ] Cálculo automático de especificaciones de cada caso
- [ ] Extracción de C1–C4
- [ ] Paralelización y guardado en Parquet/HDF5
- [ ] Tests del pipeline (casos conocidos)

**Entregable:** pipeline en el repositorio. **Cierre:** reproduce E1 de forma automática.

### Fase 4 — Generación del dataset
- [ ] Ejecutar todas las condiciones en ambos circuitos
- [ ] Añadir ruido y cuantificación
- [ ] Etiquetado en tres niveles
- [ ] Verificar el balance de clases y documentar el dataset (*datasheet*)

**Entregable:** dataset versionado. **Cierre:** dataset completo y documentado.

### Fase 5 — Testabilidad (E2, E9)
- [ ] Análisis de sensibilidad
- [ ] Grupos de ambigüedad en ambos circuitos
- [ ] Separabilidad entre clases

**Entregable:** figuras y tablas de testabilidad. **Cierre:** H3 y H5 evaluadas.

### Fase 6 — Modelos (E3–E6)
- [ ] Regresión de especificaciones y decisión apto/no apto
- [ ] Comparación gravedad funcional frente a porcentual
- [ ] Localización
- [ ] Electrodo frente a circuito

**Entregable:** resultados con intervalos de confianza. **Cierre:** H1, H2 y H4 evaluadas.

### Fase 7 — Robustez y conjunto mínimo (E7, E8)
- [ ] Magnitudes no vistas y cambio de tolerancias
- [ ] Barrido de ruido y cuantificación
- [ ] Selección de medidas y curva coste-rendimiento

**Entregable:** recomendación de medidas mínimas. **Cierre:** resultados estables entre semillas.

### Fase 8 — Redacción y publicación
- [ ] Redactar el artículo (estructura en la sección 12)
- [ ] Publicar el dataset en Zenodo con DOI
- [ ] Limpiar y documentar el repositorio
- [ ] Preprint
- [ ] Envío a revista

**Entregable:** manuscrito enviado. **Cierre:** envío realizado.

### Fase 9 — Futuro: validación en banco
- [ ] Montar el circuito y provocar un subconjunto de fallos
- [ ] Medir con el myDAQ (pulso de calibración por su salida analógica)
- [ ] Evaluar los modelos entrenados con simulación sobre datos reales

---

## 12. Estructura del artículo

**Título provisional:** *Specification-aware fault diagnosis of ECG analog front-ends: bridging alternate test and machine-learning diagnosis for in-service self-test*

1. **Introducción.** Autodiagnóstico en servicio de equipos de ECG; limitaciones del autodiagnóstico integrado de los front-ends y de la detección de electrodo desconectado; dos comunidades (test orientado a especificaciones y diagnóstico con ML) que no se hablan; problema de los circuitos de referencia; aportaciones.
2. **Estado del arte.** Diagnóstico con ML; *alternate test*; fallos a nivel de señal en ECG; precedentes en equipos biomédicos; interfaz electrodo-piel.
3. **Métodos.** Circuitos; especificaciones clínicas; modelo de fallos y electrodos; Monte Carlo; medidas autoadquiribles; etiquetado en tres niveles; modelos; protocolo de evaluación.
4. **Resultados.** Testabilidad; predicción de especificaciones; gravedad funcional frente a porcentual; localización; electrodo frente a circuito; robustez; conjunto mínimo; efecto de la arquitectura.
5. **Discusión.** Implicaciones para el diseño de autotest en equipos médicos; limitaciones (solo simulación, fallo simple, modelos de electrodo).
6. **Conclusiones y trabajo futuro.** Validación en banco, fallos múltiples, EMG y EEG.

**Disponibilidad de datos y código:** repositorio y DOI de Zenodo.

---

## 13. Revistas candidatas

| Revista | Encaje |
|---|---|
| **Medical Engineering & Physics** | El mejor encaje si se refuerza el ángulo de equipo médico y especificaciones clínicas |
| **IEEE Transactions on Instrumentation and Measurement** | Donde publica Chen et al. (2025); exige rigor metodológico |
| **Integration** / **Microelectronics Reliability** | Comunidad de diagnóstico de fallos analógicos |
| **Electronics** / **Sensors** (MDPI) / **IEEE Access** | Rápidas; valorar APC |
| **Journal of Electronic Testing (JETTA)** | Comunidad de test; buena para el ángulo de *alternate test* |

---

## 14. Riesgos

| Riesgo | Mitigación |
|---|---|
| No conseguir el macromodelo del front-end integrado | Modelo de comportamiento con parámetros de la hoja de datos |
| Pocos casos no aptos con fallos pequeños | Incluir niveles grandes y analizar el desbalance; usarlo como resultado (H2) |
| Valores de norma ambiguos | Documentar la elección y hacer análisis de sensibilidad del umbral |
| Revisor de la comunidad de test: "esto es *alternate test*" | Citarlo desde la introducción y presentar la aportación como integración y aplicación en servicio |
| Revisor biomédico: "solo simulación" | Declararlo como limitación y anunciar la fase 9 |

---

## 15. Referencias clave

Ver la lista completa y comentada en `sota_diagnostico_fallos_frontend_ecg.md`. Imprescindibles:

- Chen et al. (2025). *Evaluating and optimizing conventional training circuits for analog fault diagnosis via transfer learning.* IEEE TIM. doi:10.1109/tim.2025.3586376
- Dieste-Velasco, M. I. (2025). *Soft fault diagnosis in analog electronic circuits using supervised machine learning.* Integration. doi:10.1016/j.vlsi.2025.102482
- *Specification-driven test design for analog circuits* (1998). doi:10.1109/dftvs.1998.732183
- *Enhancing test effectiveness for analog circuits using synthesized measurements* (1998). doi:10.1109/vtest.1998.670860
- *Recent advances in analog, mixed-signal, and RF testing* (2010). doi:10.2197/ipsjtsldm.3.19
- Detección de fallos de circuito en electroencefalógrafo médico (2025). doi:10.3233/faia250392
- Agarwal, Sinha, Das (2022). *FauDigPro.* doi:10.1109/icmiam56779.2022.10146898
