# Estado del arte: diagnóstico de fallos en front-ends analógicos de ECG

> Resumen de la revisión realizada en octubre de 2026 para la línea `ecg-frontend-fault-diagnosis`. Acompaña al documento de la línea (versión 2).

---

## 1. Método

La revisión se hizo en tres pasadas complementarias:

| Pasada | Herramienta | Alcance |
|---|---|---|
| 1 | Búsqueda web | Panorama general, trabajos recientes, front-ends comerciales, detección de electrodo desconectado |
| 2 | Consensus | Síntesis automática sobre diagnóstico con simulación, testabilidad, generalización y conjuntos mínimos de prueba |
| 3 | OpenAlex (API) | Seis búsquedas booleanas en título y resumen + seguimiento de citas hacia delante y hacia atrás |

### Búsquedas en OpenAlex

| Código | Tema | Trabajos |
|---|---|---|
| B1 | Diagnóstico a nivel de componente en front-ends biomédicos | 25 |
| B2 | Fallos a nivel de señal en sensores de ECG | 27 |
| B3 | Autodiagnóstico y mantenimiento predictivo en equipos médicos | 475 |
| B4 | Test orientado a especificaciones / *alternate test* | 155 |
| B5 | Impedancia de electrodo y detección de desconexión | 353 |
| B6 | Diagnóstico analógico con transferencia y circuitos de referencia | 36 |

Seguimiento de citas sobre: FauDigPro (2022), Chen et al. (2025), Dieste-Velasco (2021, 2024).

**Corpus final:** 1.197 trabajos únicos, 976 con resumen.

### Limitaciones

- No es una revisión sistemática formal (sin protocolo PRISMA ni doble cribado).
- Clasificación por título y resumen; los trabajos clave requieren lectura completa.
- OpenAlex no indexa todo; conviene confirmar el hueco principal con Scopus o Web of Science.
- Una primera exportación tuvo errores (búsquedas sin filtro de título y resumen, ficheros duplicados) y se repitió con la API.

---

## 2. Hallazgos por área

### 2.1 Diagnóstico de fallos analógicos con ML: campo saturado

- Volumen muy alto de publicaciones con un patrón repetido: **arquitectura nueva + circuitos de referencia** (Sallen-Key paso banda, biquad de cuatro operacionales, CTSV, leapfrog).
- Técnicas: wavelets, EEMD/CEEMDAN, SVM y ELM optimizadas con metaheurísticas (PSO, lobo gris, gorrión), DBN, CNN 1D, ResNet, transformers, GAN y modelos de difusión.
- Exactitudes publicadas del 91,5 % al 100 %; trabajos de 2025–2026 siguen reportando ~98 % con random forest sobre filtros simulados.
- Ya existen **datasets y código abiertos** (dataset en IEEE DataPort, 2026; código de DiffDA-Net en Figshare, 2026).

**Conclusión:** ni el algoritmo ni la exactitud pueden ser la aportación.

### 2.2 Subcampos activos dentro del diagnóstico analógico

| Subcampo | Trabajos representativos | Implicación |
|---|---|---|
| Transferencia simulación → realidad | Varios con aprendizaje por transferencia y adaptación de dominio; MC-DT (gemelo digital, 2026) | La validación en banco tiene literatura de apoyo |
| Transferencia entre severidades de fallo | DiffDA-Net (Scientific Reports, 2026): ~71 % en tarea de 13 clases (50 % → 25 %) | La generalización a magnitudes no vistas es evaluación, no aportación |
| Selección de puntos de prueba | Aprendizaje por refuerzo profundo (2025); estrategias de Pareto (2024); Faster R-CNN (2023) | El conjunto mínimo de medidas es evaluación, no aportación |
| Testabilidad y grupos de ambigüedad | Análisis de testabilidad automático (Tang et al., 2018) | Base metodológica para E2 |
| **Calidad de los circuitos de referencia** | **Chen et al. (IEEE TIM, 2025)**: las simetrías de los circuitos clásicos producen solapamiento de firmas de fallo y perjudican la generalización | **Argumento central para usar un circuito realista del dominio** |

### 2.3 Test orientado a especificaciones: la idea de gravedad funcional ya existe

- El test analógico industrial se basa en comprobar la conformidad con especificaciones porque no hay modelos de fallo de aplicación general (revisión de 2010).
- El **alternate test** (desde finales de los 90) predice si el circuito cumple especificaciones a partir de medidas baratas e indirectas, minimizando la probabilidad de clasificar un circuito defectuoso como bueno y viceversa.
- Hay trabajos de compactación de tests de especificaciones (2005), predicción de prestaciones con firmas (2006–2007), autotest integrado orientado a especificaciones reutilizando DAC y ADC del propio sistema (2011) y combinación con ML (ITC 2022).
- **Contexto:** test de producción de circuitos integrados; no localiza componentes; no trabaja en servicio ni en equipos biomédicos.

**Conclusión:** la gravedad funcional no es una idea nueva. Lo nuevo es **integrarla con la localización** y **aplicarla en servicio** a un front-end biomédico.

### 2.4 Fallos a nivel de señal en sensores de ECG: literatura adyacente

- **FauDigPro** (Agarwal, Sinha y Das, 2022): clasifica señales de ECG normales frente a defectuosas y predice si el sensor fallará. **Nivel de señal, no de componente.** Solo 4 citas; ninguna sigue la vía de componente.
- Detección de fallos en datos de sensores de ECG inalámbricos con AD8232 (2025): mala colocación, conexión y movimiento.
- Marco MIoT de anomalías de ECG y clasificación de fallos de sensor (Scientific Reports, 2026); validación de calidad de bioseñales (Sensors, 2026); detección federada de fallos de sensor (2025); redes bayesianas para probabilidad de fallo de sensores biomédicos (2026).

**Conclusión:** literatura activa y creciente, pero detecta *que* los datos son malos, no *qué parte del circuito* ha fallado ni si el equipo sigue cumpliendo especificaciones.

### 2.5 Fallos de circuito en equipos biomédicos: precedentes débiles

| Trabajo | Qué hace | Diferencia con esta línea |
|---|---|---|
| Electroencefalógrafo médico (FAIA, 2025) | Wavelet packet + red BP para clasificar fallos de circuito; fallos no detectados < 9 % | Equipo concreto; sin tolerancias ni especificaciones; sin electrodos; revista de bajo impacto |
| Equipos médicos con LSTM (2021) | Nivel de placa, sin esquemas; señales de puertos + síntomas; 7 categorías | Nivel de placa; datos reales de reparación; no front-end |
| Mantenimiento predictivo hospitalario (varias revisiones 2016–2026) | Gestión de activos, bombas, ventiladores, resonancia | Nivel de equipo/sistema; no circuito |

**Conclusión:** no se puede afirmar que nadie haya aplicado diagnóstico de circuito en biomedicina, pero **no hay ningún estudio sistemático con simulación, tolerancias y especificaciones sobre front-ends de biopotencial**.

### 2.6 Front-ends comerciales y detección de electrodo desconectado

- Los front-ends integrados comerciales (por ejemplo, ADS1293) incluyen indicadores de fallo del amplificador de instrumentación, del modulador y de electrodo desconectado.
- La detección de desconexión es una técnica industrial madura (métodos en continua y en alterna) con limitaciones conocidas: umbral variable según el electrodo, polarización indistinguible de la corriente de detección, dificultad con electrodos secos o capacitivos.
- B5 (353 trabajos) está dominada por **materiales de electrodos** (secos, textiles, microagujas, hidrogeles). **Ningún trabajo separa la degradación del electrodo de un fallo del circuito.**
- Referencias útiles para modelar la interfaz: impedancia electrodo-tejido y artefactos de movimiento (2012); impedancia de contacto, material e hidratación (2022); calidad de señal con electrodos secos y rangos de parámetros de la interfaz (2024).

**Conclusión:** el autodiagnóstico integrado no cubre la red discreta externa, y la distinción electrodo/circuito es un hueco real.

### 2.7 Referencias de diseño de front-ends

Útiles para el circuito bajo estudio (aparecen en B1):
- Amplificador de biopotencial de tres operacionales acoplado en alterna con supresión activa de continua (IEEE TBME, 2000).
- Amplificador de biopotencial totalmente diferencial con pocos componentes (IEEE TIM, 2022).
- Pierna derecha activa digital (EMBC, 2010).
- Revisión de front-ends para electrodos aislantes (Physiological Measurement, 2010).
- Análisis comparativo del ruido en lazos servo de continua (preprint, 2024).

---

## 3. Mapa del hueco

| Aportación | Evidencia en contra | Estado |
|---|---|---|
| Diagnóstico de componente en front-end de ECG con simulación y tolerancias | Precedentes débiles en EEG y nivel de placa | **Abierto, con matices** |
| Gravedad funcional (por especificaciones) | *Alternate test* y test orientado a especificaciones | **Reformulado**: integración con localización y uso en servicio |
| Separación electrodo / circuito | Ninguna | **Abierto** |
| Circuito realista del dominio | Chen et al. (2025) lo apoya | **Abierto y reforzado** |
| Generalización y conjunto mínimo | DiffDA-Net, aprendizaje por refuerzo, etc. | **Evaluación, no aportación** |
| Dataset abierto | Ya existen datasets genéricos | **Necesario, no suficiente** |

**Hueco final:** no existe un trabajo que combine verificación de especificaciones clínicas, localización de componentes y separación electrodo/circuito en el autodiagnóstico en servicio de un front-end de biopotencial.

---

## 4. Referencias

### Imprescindibles

- Chen, Peng, Huang, Tang (2025). *Evaluating and optimizing conventional training circuits for analog fault diagnosis via transfer learning.* IEEE Transactions on Instrumentation and Measurement, 74. doi:10.1109/tim.2025.3586376
- Dieste-Velasco, M. I. (2025). *Soft fault diagnosis in analog electronic circuits using supervised machine learning.* Integration. doi:10.1016/j.vlsi.2025.102482
- Dieste-Velasco, M. I. (2024). *Fault detection in analog electronic circuits using fuzzy inference systems and particle swarm optimization.* Alexandria Engineering Journal. doi:10.1016/j.aej.2024.01.054
- Dieste-Velasco, M. I. (2021). *Application of a pattern-recognition neural network for detecting analog electronic circuit faults.* Mathematics, 9(24), 3247. (DOI a verificar: 10.3390/math9243247)
- *Specification-driven test design for analog circuits* (1998). doi:10.1109/dftvs.1998.732183
- *Enhancing test effectiveness for analog circuits using synthesized measurements* (1998). doi:10.1109/vtest.1998.670860
- *Recent advances in analog, mixed-signal, and RF testing* (2010). IPSJ Trans. System LSI Design Methodology. doi:10.2197/ipsjtsldm.3.19
- *Testing of analog circuits using statistical and machine learning techniques* (ITC, 2022). doi:10.1109/itc50671.2022.00087
- *DiffDA-Net: diffusion-augmented domain adaptive network for analog circuit fault diagnosis under imbalanced and variable operating conditions* (2026). Scientific Reports. doi:10.1038/s41598-026-60313-3

### Precedentes biomédicos y nivel de señal

- Agarwal, Sinha, Das (2022). *FauDigPro: a machine learning based fault diagnosis and prognosis system for electrocardiogram sensors.* ICMIAM. doi:10.1109/icmiam56779.2022.10146898
- *Research on medical device fault diagnosis and maintenance optimization based on big data* (2025). Frontiers in AI and Applications. doi:10.3233/faia250392
- *Intelligent fault diagnosis of medical equipment based on long short term memory network* (2021). doi:10.7507/1001-5515.201912019
- *Deep Bayesian networks for failure probability estimation in biomedical sensors* (2026). Eksploatacja i Niezawodność. doi:10.17531/ein/218121
- *A power-efficient layered MIoT framework for real-time ECG anomaly detection and sensor fault classification* (2026). Scientific Reports. doi:10.1038/s41598-026-49593-x
- *High-reliability signal quality validation for biosignals using sensor fusion and software indices* (2026). Sensors. doi:10.3390/s26113478
- *Lightweight AI for sensor fault monitoring* (2025). Electronics. doi:10.3390/electronics14224532

### Diagnóstico analógico (posicionamiento)

- *Data-driven feature extraction for analog circuit fault diagnosis using 1-D CNN* (2020). IEEE Access. doi:10.1109/access.2020.2968744
- *Learnable wavelet scattering networks: applications to fault diagnosis of analog circuits and rotating machinery* (2022). Electronics. doi:10.3390/electronics11030451
- *Machine learning classifiers with explainable insights for parametric fault diagnosis in linear analog circuits* (2025). Microelectronics Reliability. doi:10.1016/j.microrel.2025.115982
- *Analog circuit test point selection method for fault diagnosis based on deep reinforcement learning* (2025). doi:10.2139/ssrn.5338908
- *Specification test compaction for analog circuits and MEMS* (DATE, 2005). doi:10.1109/date.2005.277
- *Analog circuit fault diagnosis dataset* (2026). IEEE DataPort. doi:10.21227/tvch-9d57

### Electrodos y front-ends

- *Correlation between electrode-tissue impedance and motion artifact in biopotential recordings* (2012). IEEE Sensors Journal. doi:10.1109/jsen.2012.2221163
- *Dependence of skin-electrode contact impedance on material and skin hydration* (2022). Sensors. doi:10.3390/s22218510
- *ECG signal quality in intermittent long-term dry electrode recordings with controlled motion artifacts* (2024). Scientific Reports. doi:10.1038/s41598-024-56595-0
- Mayosky, Spinelli (2000). *AC coupled three op-amp biopotential amplifier with active DC suppression.* IEEE TBME. doi:10.1109/10.887943
- *A fully-differential biopotential amplifier with a reduced number of parts* (2022). IEEE TIM. doi:10.1109/tim.2022.3220284
- Spinelli, Haberman (2010). *Insulating electrodes: a review on biopotential front ends for dielectric skin–electrode interfaces.* Physiological Measurement. doi:10.1088/0967-3334/31/10/s03
- Texas Instruments. Hoja de datos ADS1293 y nota de aplicación SBAA196A (detección de electrodo desconectado).

> Algunas fechas de OpenAlex no coinciden con las del DOI (por ejemplo, los trabajos de *alternate test* figuran como 2002 pero sus DOI son de congresos de 1998). Verificar los datos bibliográficos al citar.

---

## 5. Pendiente

- [ ] Confirmar el hueco principal con una búsqueda en Scopus o Web of Science.
- [ ] Lectura completa de los trabajos imprescindibles (Fase 1 del documento de la línea).
- [ ] Localizar el artículo original al que responde *Electrocardiogram failure in the operating room – manufacturer's comment* (Anaesthesia, 2018), como posible motivación clínica.
- [ ] Revisar las normas IEC de electrocardiógrafos para la tabla de especificaciones.
