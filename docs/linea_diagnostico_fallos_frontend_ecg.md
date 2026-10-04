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
| ngspice | Simulación, lanzado desde Python con un lanzador propio |
| schemdraw | Esquemas generados desde la tabla de componentes del código (`scripts/draw_schematics.py`) |
| NumPy, pandas | Procesado |
| scikit-learn | Modelos clásicos, regresión, selección de características |
| PyTorch | CNN 1D |
| Git + Zenodo | Código y dataset con DOI |

---

## 11. Fases de desarrollo

Marcar cada casilla al completarla. Cada fase tiene un entregable y un criterio de cierre.

> Los resultados de las fases 4 a 7 corresponden a la versión 2 del dataset (`data/v2`): electrodos secos según los datos por sujeto del artículo de 2024, salida del amplificador de instrumentación dentro del conjunto principal de medidas, y localización por grupos de ambigüedad.

### Fase 0 — Estado del arte ✅
- [x] Revisión inicial con búsqueda web
- [x] Revisión con Consensus
- [x] Búsquedas B1–B6 en OpenAlex y seguimiento de citas
- [x] Análisis del corpus (1.197 trabajos, 976 con resumen)
- [x] Resumen del SOTA (`sota_diagnostico_fallos_frontend_ecg.md`)

**Entregable:** resumen del SOTA. **Cierre:** hueco reformulado.

### Fase 1 — Lecturas clave y posicionamiento ✅
- [x] Leer Chen et al. (2025), IEEE TIM
- [x] Leer Dieste-Velasco (2025), Integration (y los de 2021 y 2024)
- [x] Leer el trabajo del electroencefalógrafo (2025) y el de LSTM en equipos médicos (2021) (el segundo, en traducción automática)
- [x] Leer la revisión de test analógico de 2010 y los dos trabajos fundacionales de *alternate test* (1998)
- [x] Leer el trabajo de ITC 2022 sobre test estadístico y ML
- [x] Leer DiffDA-Net (2026)
- [x] Redactar borrador de introducción y estado del arte (`docs/paper/introduction_draft.md`)

**Entregable:** borrador de introducción. **Cierre:** las cinco aportaciones están justificadas con referencias.

**Estado:** el borrador y las notas de lectura (`docs/lecturas_fase1.md`) están contrastados con los textos de los once trabajos. La revisión de 2010 se leyó entera; del resto, el método, el modelo de fallo, los datos y las conclusiones. Las cinco aportaciones tienen su respaldo en una tabla. Quedan para la redacción final cinco referencias de apoyo citadas por su resumen y dos afirmaciones sin referencia (qué detectan por sí solos los front-ends integrados y cuándo se verifica la conformidad), listadas al final del borrador.

**Resultado de contrastar:** el trabajo de ITC 2022 no une test por especificaciones y diagnóstico (define el fallo por tolerancia, sobre los circuitos de referencia); Dieste-Velasco (2021) ya agrupa por simulación los fallos indistinguibles antes de entrenar, y pasa a citarse como precedente del tratamiento por grupos.

**Cambio de posicionamiento.** La revisión de 2010 recoge diagnóstico apoyado en *alternate test* para transceptores de RF integrados. La aportación 1 no puede formularse como "unir verificación de especificaciones y localización" sin más; se acota a una red de componentes discretos, en servicio, con las medidas del propio equipo, contra una norma clínica y con el electrodo como confusor.

### Fase 2 — Diseño del circuito y especificaciones ✅
- [x] Elegir el front-end integrado y conseguir su macromodelo (o definir el modelo de comportamiento): **INA333** de Texas Instruments, con modelo de comportamiento construido con su hoja de datos y contrastado a nivel de bloque con el macromodelo del fabricante
- [x] Diseñar la red discreta con valores justificados
- [x] Diseñar el circuito discreto de referencia
- [x] Definir el modelo de electrodo (gel y seco): secos con las medianas medidas de seis materiales (*Scientific Reports* 2024); gel con la red de 51 kΩ ‖ 47 nF de la norma; la dispersión en torno a cada mediana es un supuesto
- [x] Fijar la tabla de especificaciones según IEC 60601-2-25: límites y montajes de ensayo contrastados con el texto de la norma
- [x] Esquemas de ambos circuitos en `docs/figures/` (la captura en Qucs-S sale del alcance)

**Entregable:** esquemas y tabla de especificaciones. **Cierre:** E1 superado (circuitos nominales aptos).

**Resultado:** ambos circuitos nominales y 500 circuitos sanos de Monte Carlo de cada uno cumplen las once especificaciones. El diseño, la procedencia de cada límite y lo que queda por contrastar están en `docs/circuit.md`.

### Fase 3 — Pipeline de simulación ✅
- [x] Automatización de ngspice desde Python (lanzador propio en lugar de PySpice; la prueba de concepto con un filtro sencillo quedó cubierta por el circuito completo)
- [x] Generación paramétrica de netlists con inyección de fallos
- [x] Monte Carlo de tolerancias
- [x] Cálculo automático de especificaciones de cada caso
- [x] Extracción de C1–C4 (C4 en su variante de alterna)
- [x] Paralelización y guardado en Parquet, con reanudación tras una interrupción
- [x] Reetiquetado sin volver a simular cuando cambian los límites (`ecgfd relabel`)
- [x] Tests del pipeline (casos conocidos)

**Entregable:** pipeline en el repositorio. **Cierre:** reproduce E1 de forma automática (`make e1`).

Los puntos abiertos de todas las fases están reunidos en `docs/pendientes.md`.

### Fase 4 — Generación del dataset ✅
- [x] Ejecutar todas las condiciones en ambos circuitos (`make dataset`)
- [x] Añadir ruido y cuantificación (se aplican al cargar los datos, para poder variarlos sin volver a simular)
- [x] Etiquetado en tres niveles
- [x] Verificar el balance de clases sobre el dataset generado (`make report`)
- [x] Documentar el dataset (`docs/dataset.md`; las cifras de cada versión van en el `report.md` de su carpeta)

**Entregable:** dataset versionado. **Cierre:** dataset completo y documentado.

**Resultado (versión `data/v2`):** 63.600 casos del circuito integrado y 66.400 del de referencia, sin ninguna simulación fallida (81 y 75 minutos). Son aptos el 79 % y el 70 % de los casos, y el 100 % de los sanos. Las cifras completas están en el `report.md` de cada carpeta. Observaciones:

- Los fallos de ±5 % solo sacan de especificación al circuito cuando tocan la ganancia (R11/R12 del integrado, la mitad de los casos; R7 y la etapa de ganancia del de referencia). Apoya H2.
- En el circuito integrado ningún fallo de la pierna derecha activa (R7, R8, C4, U2, U3) incumple la norma: el camino pasivo de R9 hacia la referencia basta para el ensayo de 89 dB. Lo mismo ocurre con los seguidores U5 y U6.
- Algunos fallos duros dejan el circuito apto aunque afectan a la seguridad, como R9 o R1/R2 en corto (desaparece la limitación de corriente hacia el paciente). La norma de prestaciones no los ve; se declara como limitación.
- La ganancia medida en servicio por un circuito sano depende mucho del electrodo: entre 0,90 y 1,03 de la nominal con gel y metales sólidos, entre 0,59 y 1,01 con polímero, y entre 0,05 y 0,97 con tela (mediana 0,6–0,7).
- El balance de origen está muy descompensado: 5.000 sanos, 2.200 de electrodo y unos 57.000 de circuito.

### Fase 5 — Testabilidad (E2, E9) ✅
- [x] Análisis de sensibilidad (E2: sensibilidad normalizada por la dispersión de los sanos, rango de testabilidad y componentes colineales)
- [x] Grupos de ambigüedad en ambos circuitos (E2: condiciones indetectables, escapes inevitables y componentes confundibles entre los casos no aptos)
- [x] Separabilidad entre clases (E9, que compara los dos circuitos)

**Entregable:** figuras y tablas de testabilidad. **Cierre:** H3 y H5 evaluadas.

**Resultado** (`make testability` sobre `data/v2`; tablas y figuras en `results/e2/` y `results/e9/`). Cifras con el conjunto completo C1+C2+C3+C4. Son mapas con un criterio univariante, conservador.

| | Integrado | Referencia |
|---|---|---|
| Condiciones no aptas indistinguibles de los sanos, todos los electrodos | 9 | 16 |
| Ídem, sin electrodos secos porosos | 1 | 6 |
| Ídem, detector de límites (1 % de falsas alarmas), sin porosos | 3 | 8 |
| Rango de testabilidad (desviación del 10 %), todos / sin porosos | 2 / 4 | 2 / 3 |
| Componentes localizables sin ambigüedad, todos / sin porosos | 4 de 23 / 5 de 23 | 2 de 29 / 5 de 29 |
| Componentes con los que se confunde cada uno, en media, todos / sin porosos | 3,9 / 2,5 | 8,6 / 5,2 |

- **Electrodos porosos.** En el integrado, 8 de las 9 condiciones invisibles son fallos de ganancia de ±5 a ±20 %, que se confunden con la atenuación de un electrodo de alta impedancia; sin los porosos solo queda R9 abierta.
- **Escapes estructurales.** En el de referencia quedan seis sin porosos, todos de la pierna derecha (R12, R13, R14, C3): incumplen el rechazo en modo común pero las medidas no los ven. Los offsets de U1/U2 dejaron de escapar al incluir la salida del amplificador de instrumentación en las medidas.
- **H3, apoyada.** Pocos componentes se localizan sin ambigüedad, y los grupos se pueden anticipar con la sensibilidad sola. Integrado: R5+R6+R11+R12 (ganancia), C5+R10 (paso alto), R13+R14 (paso bajo), R15+R16 (referencia). Referencia: nueve resistencias de ganancia en un grupo, C4+R16, R19+R20 y R12+R13.
- **H5, apoyada.** La estructura de ambigüedad cambia con la arquitectura: el integrado tiene menos escapes, menos confusión media y, según E5, mejor localización (F1 por componente 0,74 frente a 0,61). La separabilidad entre centroides es la única medida que no lo favorece: es parecida con todos los electrodos (1,5 en ambos) y algo mayor en el de referencia sin porosos (2,1 frente a 1,8). Las resistencias simétricas del amplificador discreto son las más ambiguas, en línea con Chen et al. (2025).
- **Medida de modo común.** El tono de modo común no es sensible a ningún componente; E7 confirma que subir su amplitud o su duración no lo arregla.

### Fase 6 — Modelos (E3–E6) ✅
- [x] Regresión de especificaciones y decisión apto/no apto (E3: detector de límites, regresión con banda de guarda y clasificador con umbral, para un objetivo de escapes del 1 %)
- [x] Comparación gravedad funcional frente a porcentual (E4)
- [x] Localización (E5: cinco clasificadores y CNN 1D; contraste con la ambigüedad prevista en E2)
- [x] Electrodo frente a circuito (E6: tres clases con ponderación por desequilibrio)

**Entregable:** resultados con intervalos de confianza. **Cierre:** H1, H2 y H4 evaluadas.

Protocolo común: tres repeticiones con partición entrenamiento/validación/prueba (52,5 / 17,5 / 30 %), hiperparámetros, bandas de guarda y umbrales elegidos solo en validación, e intervalos de confianza del 95 % sobre las repeticiones. Se lanza con `make models` (y `make models-sensitivity` para las variantes de electrodos).

**Resultado** (sobre `data/v2`, con los seis tipos de electrodo y el conjunto completo C1+C2+C3+C4; tablas en `results/e3` a `results/e6`; entre paréntesis, intervalos de confianza del 95 %):

- **H1, apoyada en el circuito integrado y no en el de referencia.**

  | | Escapes | Falsos rechazos |
  |---|---|---|
  | Integrado, detector de límites | 22,5 % | 20,0 % |
  | Integrado, regresión de especificaciones | 8,4 % | 1,5 % |
  | Integrado, clasificador | 7,9 % | 0,7 % |
  | Integrado, clasificador ajustado a 1 % de escapes | 0,9 % | 7,8 % (4,8–10,8) |
  | Referencia, clasificador | 11,2 % | 1,2 % |
  | Referencia, clasificador ajustado a 1 % de escapes | 1,2 % | 54 % (48–60) |

  En el de referencia los escapes que quedan son fallos que las medidas no ven (pierna derecha), como anticipó E2. La regresión de especificaciones iguala al clasificador en su punto por defecto, pero su banda de guarda no da un punto de operación útil cerca del 1 % de escapes. Sin electrodos porosos, el integrado alcanza el 1 % de escapes con solo un 1,9 % de falsos rechazos; conocer el tipo de electrodo, en cambio, no ayuda.
- **H2, apoyada.** Entrenar con "componente fuera de tolerancia" da un 37 % de falsos rechazos en ambos circuitos; entrenar con "circuito fuera de especificación", un 1,6 % y un 2,2 %, con menos escapes (4,1 % frente a 7,9 % en el integrado; 7,6 % frente a 8,3 % en el de referencia). Siguen siendo aptos más del 93 % de los fallos de ±5 % y entre el 54 % y el 66 % de los de ±50 %.
- **H3, apoyada.** Localización sobre los casos no aptos, mejor modelo (bosque aleatorio):

  | | Integrado | Referencia |
  |---|---|---|
  | Por grupo de ambigüedad: F1 macro / exactitud / entre los 3 primeros | 0,87 / 95 % / 99,9 % | 0,76 / 94 % / 99,4 % |
  | Por componente: F1 macro / entre los 3 primeros | 0,74 / 96 % | 0,61 / 84 % |

  Los componentes peor localizados son los de los grupos previstos por la sensibilidad (R5/R6, R13/R14, R11/R12). El acierto por componente se correlaciona con el número de componentes confundibles que predijo E2 (ρ = −0,42 y −0,51, p < 0,05). La CNN sobre la forma de onda queda por debajo de los modelos tabulares (F1 por grupo 0,56 y 0,32).
- **H4, apoyada en parte.** Con las clases "nada que hacer / circuito no apto / electrodo o cable": F1 macro 0,89 y 0,87. Se reconoce el 99 % de los casos sin nada que hacer y el 94 % y 92 % de los circuitos no aptos, pero solo el 64 % y 61 % de los fallos de electrodo. Un electrodo desconectado se reconoce en el 75 % y 89 % de los casos, y la resistencia de protección abierta siempre. Lo que falla es la degradación del contacto: la mitad pasa por normal, porque una impedancia cinco o veinte veces mayor cae dentro de lo que se mide en otros sujetos sanos. Sin electrodos porosos el reconocimiento de electrodo sube al 76 % y 74 %.

### Fase 7 — Robustez y conjunto mínimo (E7, E8) ✅
- [x] Magnitudes no vistas y cambio de tolerancias (E7: magnitudes paramétricas fuera del entrenamiento; datasets de prueba con tolerancias gaussianas truncadas y con componentes al 2 % y 10 %)
- [x] Barrido de ruido y cuantificación (E7: ruido del ADC de 0,1 a 10 mV y de 8 a 16 bits, entrenando en la misma condición o en la de referencia)
- [x] Selección de medidas y curva coste-rendimiento (E8: selección voraz de acciones de autotest, con su tiempo, para la decisión de aptitud y para la localización)

**Entregable:** recomendación de medidas mínimas. **Cierre:** resultados estables entre semillas.

Se lanza con `make shift-datasets` y `make robustness`. **Resultado** (sobre `data/v2`; tablas y figuras en `results/e7/` y `results/e8/`; tres repeticiones, con intervalos estrechos en todas las cifras):

- **Magnitudes no vistas.** La decisión de aptitud generaliza: la exactitud equilibrada baja entre 1 y 6 puntos al puntuar magnitudes que no estaban en el entrenamiento. La localización por componente no: cae del 45–70 % al 3–33 %, porque dentro de un grupo colineal el modelo distingue los componentes memorizando magnitudes. Por grupo de ambigüedad se mantiene: 99 % y 95 % de acierto en el integrado con magnitudes pequeñas e intermedias no vistas, y 84 % con las grandes (96 %, 81 % y 79 % en el de referencia). La localización debe darse por grupo.
- **Ruido y cuantificación.** Entrenando en la misma condición, el rendimiento apenas cambia entre 0,1 y 10 mV de ruido y entre 8 y 16 bits. Entrenando a 1 mV y midiendo a 10 mV sí se degrada: los escapes del integrado pasan del 4 % al 10 % y los falsos rechazos del de referencia del 2 % al 19 %.
- **Otras tolerancias.** Con distribución gaussiana truncada no cambia nada. Con componentes más holgados (2 % y 10 %) los escapes suben del 4,1 % al 7,1 % en el integrado y del 7,6 % al 11,4 % en el de referencia; la localización por grupo se mantiene en el integrado (F1 0,88 a 0,87) y cae en el de referencia (0,76 a 0,45), que es más frágil.
- **Tono de modo común.** Subir su amplitud de 0,1 a 1 V o alargarlo diez veces no hace detectables los fallos de la pierna derecha: se escapan alrededor de la mitad en cualquier caso. Hace falta otra medida, como la salida de la pierna derecha.
- **Conjunto mínimo de medidas.** El pulso de calibración es siempre la primera medida elegida. Para decidir la aptitud bastan 3 medidas y 3 segundos en el integrado (pulso, un tono de impedancia de contacto y el tono diferencial de 150 Hz) para quedar a menos de un punto del autotest completo, que dura 97 segundos; en el de referencia son 4 medidas, con la salida de la pierna derecha. Para localizar, en el integrado bastan el pulso, el tono de 150 Hz y la salida del amplificador de instrumentación (2 segundos); el de referencia necesita 5 o 6 medidas. Los tonos de 0,05 Hz, que suponen 80 de los 97 segundos, no se eligen nunca.

### Fase 8 — Redacción y publicación
- [ ] Redactar el artículo (estructura en la sección 12): borrador completo en `manuscript/` (plantilla de IEEE TIM, 8 páginas más 4 de material suplementario, sin cargos por exceso; se genera con `make paper`; la versión larga de 12 páginas queda en `manuscript/long_version/`); pendiente de revisión del autor
- [x] Publicar el dataset en Zenodo: <https://doi.org/10.5281/zenodo.23134950>
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
