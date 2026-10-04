# Fase 1: lecturas clave y posicionamiento

Notas de lectura de los trabajos de la fase 1 del [plan](linea_diagnostico_fallos_frontend_ecg.md) y lo que implican para situar el artículo. El borrador de introducción y estado del arte que sale de ellas está en [paper/introduction_draft.md](paper/introduction_draft.md).

## Alcance real de la lectura

Los once trabajos se han contrastado con su texto, que está en `docs/normative_and_papers/` (carpeta sin seguimiento en git). La revisión de 2010 se leyó entera; del resto se leyó lo necesario para comprobar cada afirmación del borrador: método, circuitos, modelo de fallo, datos y conclusiones. No es una lectura línea a línea de cada artículo.

| Trabajo | Leído |
|---|---|
| Cheng y Chang (2010), revisión de test analógico | Texto completo |
| Chen et al. (2025), IEEE TIM | Texto: método, circuitos, métricas, conclusiones |
| Dieste-Velasco (2025), Integration | Texto: circuito, clases, predictores, resultados |
| Dieste-Velasco (2024), Alexandria Engineering Journal | Texto, parcial: circuitos y modelo de fallo |
| Dieste-Velasco (2021), Mathematics | Texto: circuitos, modelo de fallo, grupos de ambigüedad |
| Variyam y Chatterjee (1998), dos trabajos de *alternate test* | Texto de ambos |
| Srimani y Rahaman (2022), ITC | Texto: circuitos y definición de fallo |
| Zhang et al. (2026), DiffDA-Net | Texto: circuitos, tolerancias, protocolos y cifras |
| Gao (2025), electroencefalógrafo | Texto completo (es breve) |
| Liu et al. (2021), LSTM en equipos médicos | Traducción automática al español de la página de la revista |
| Agarwal et al. (2022), FauDigPro | Texto: datos e inyección de fallos |

**Qué ha cambiado al contrastar.** El borrador se sostenía en lo esencial. Tres correcciones:

1. **Srimani y Rahaman (2022)** no une test por especificaciones y diagnóstico. Usa los circuitos de referencia habituales y define el fallo como un componente fuera de su tolerancia. Se ha reescrito la frase que lo citaba.
2. **Dieste-Velasco (2021)** no usa un filtro sino etapas amplificadoras con transistores, y determina los grupos de ambigüedad por simulación antes de entrenar. Es un precedente directo de nuestro tratamiento por grupos y ahora se cita como tal.
3. **Chen et al. (2025)**: número de artículo 3549015 (no páginas 1–15). **Zhang et al. (2026)** sigue en prensa, sin volumen.

## Hallazgo que cambia el posicionamiento

La revisión de 2010 dedica un apartado (5.1) al **diagnóstico apoyado en *alternate test***: las firmas baratas que predicen las prestaciones también se usan, con aprendizaje estadístico, para identificar la fuente del fallo en transceptores de RF integrados (Han et al., 2006; Srinivasan et al., 2006; Suenaga et al., 2009).

Por tanto, no se puede afirmar que el test orientado a especificaciones "no localiza". La primera aportación del plan ("integración de verificación de especificaciones y localización") hay que acotarla. Lo que sigue siendo nuevo:

- **Dónde se aplica**: una red de componentes discretos alrededor de un front-end integrado, no los bloques internos de un circuito integrado.
- **Cuándo**: en servicio, con el ADC del propio equipo, no en producción ni en oblea con equipo de test.
- **Contra qué**: las especificaciones de una norma clínica (IEC 60601-2-25), no las de una hoja de datos.
- **Con qué resolución**: componentes agrupados en grupos de ambigüedad previstos a priori.
- **Con qué confusor**: la interfaz electrodo-piel, que no existe en el test de circuitos integrados.

El borrador ya está escrito con esta formulación.

## Notas por trabajo

### Cheng y Chang (2010). *Recent advances in analog, mixed-signal, and RF testing*

IPSJ Transactions on System LSI Design Methodology, 3, 19–46. doi:10.2197/ipsjtsldm.3.19. **Texto completo.**

- El test analógico se basa en comprobar especificaciones porque no hay modelos de fallo de aplicación general, y eso es caro.
- ***Alternate test*** (apartado 4.2): en vez de medir las prestaciones, se predicen a partir de firmas baratas. Se apoya en la correlación entre el espacio de parámetros, el de firmas y el de especificaciones. Hay dos variantes, y son exactamente las dos de nuestro E3:
  - una región de aceptación en el espacio de firmas, obtenida por Monte Carlo y clasificación estadística, "para minimizar los escapes y los rechazos indebidos";
  - una regresión de firmas a especificaciones (MARS), tras una fase de calibración con 100–200 unidades medidas.
- **Diagnóstico** (apartado 5.1): tres familias, basadas en test de especificaciones, en *alternate test* y en test estructural. Las de *alternate test* usan aprendizaje estadístico para relacionar firmas y fuentes de fallo, con sensores integrados.
- Cierra pidiendo que el test sirva también para depuración, **test en campo** y ajuste, compartiendo recursos.
- **Para qué citarlo**: origen y definición del enfoque; fuente de la terminología (escapes, región de aceptación); y para reconocer el diagnóstico con *alternate test* en circuitos integrados.

### Variyam y Chatterjee (1998), dos trabajos

- *Enhancing test effectiveness for analog circuits using synthesized measurements*. Proc. 16th IEEE VLSI Test Symposium, 132–137. doi:10.1109/vtest.1998.670860.
- *Specification-driven test design for analog circuits*. Proc. IEEE Int. Symp. on Defect and Fault Tolerance in VLSI Systems, 335–340. doi:10.1109/dftvs.1998.732183.

**Contenido.** El primero trata cómo fijar el umbral de apto/no apto en los *alternate tests*, sintetizando medidas nuevas a partir de las existentes. El segundo genera estímulos para detectar circuitos que incumplen alguna especificación sin hacer todos los ensayos de especificación, buscando reducir la probabilidad de clasificar un circuito malo como bueno y viceversa.

**Texto.** Confirmado. Un circuito es bueno si cumple todas sus especificaciones y malo si incumple alguna; el objetivo es minimizar a la vez el rechazo de circuitos buenos y la aceptación de malos, lo que llaman cobertura de fallos y de rendimiento (*fault and yield coverage*). El ámbito es el test de producción.

- **Para qué citarlos**: la idea de gravedad funcional y el compromiso entre escapes y falsos rechazos existen desde finales de los 90.
- OpenAlex los fecha en 2002; el DOI y las actas son de 1998. Se citan como 1998.

### Srimani y Rahaman (2022). *Testing of analog circuits using statistical and machine learning techniques*

IEEE International Test Conference (ITC), 619–626. doi:10.1109/itc50671.2022.00087. **Texto.**

- Es el resumen de una tesis, no un artículo con un único método.
- Circuitos: Sallen-Key paso banda, bicuadrático paso alto de cuatro operacionales, filtro de variables de estado, amplificador en cascada y operacional de dos etapas.
- Un fallo es un componente cuyo valor queda fuera de su intervalo de tolerancia. Es una definición por porcentaje, no por especificación.
- **Para qué citarlo**: muestra que en los foros de test el diagnóstico con aprendizaje usa los mismos circuitos y la misma definición de fallo que la comunidad de instrumentación. No sirve para afirmar que alguien haya unido cumplimiento de especificaciones y diagnóstico.

### Chen, Peng, Huang y Tang (2025). *Evaluating and optimizing conventional training circuits for analog fault diagnosis via transfer learning*

IEEE Transactions on Instrumentation and Measurement, 74, 3549015. doi:10.1109/tim.2025.3586376. **Texto.**

- Las simetrías de componentes en los circuitos de referencia clásicos (Sallen-Key paso banda, bicuadrático de cuatro operacionales) producen solapamiento de respuestas de fallo y perjudican la clasificación y la generalización.
- Modifican el bicuadrático para romper las simetrías y lo evalúan con una ResNet 1D y transferencia a un circuito *leapfrog*, con 100 repeticiones. Miden la separabilidad con distancia entre centroides y varianza dentro de clase.
- **Para qué citarlo**: es el argumento central para usar un circuito del dominio, y coincide con lo que vemos: las resistencias simétricas del amplificador discreto son las más ambiguas, y las colineales del integrado forman los grupos de ambigüedad.
- **Diferencia**: ellos rediseñan el circuito de entrenamiento para que sea más separable; nosotros no podemos elegir el circuito, así que aceptamos la ambigüedad y la tratamos como grupos.
- **Datos del texto**: tolerancias del 5 % en resistencias y del 10 % en condensadores; fallo blando = desviación de ±50 %. La separabilidad se mide en el espacio de características de la red: distancia euclídea media entre centroides de clase y varianza dentro de clase. Nuestra medida de E9 es del mismo tipo, pero sobre las medidas y no sobre una representación aprendida; no son comparables en valor.
- Afirma que los componentes simétricos dan respuestas indistinguibles y que elegir los componentes con fallo por análisis de sensibilidad no basta para evitarlo.
- La introducción señala que los fallos blandos aparecen con el uso y que quien usa el equipo no tiene instrumental para detectarlos. Respalda la motivación del autotest en servicio y se cita también ahí.

### Dieste-Velasco (2021, 2024, 2025)

- (2021) *Application of a pattern-recognition neural network for detecting analog electronic circuit faults*. Mathematics, 9(24), 3247. doi:10.3390/math9243247.
- (2024) *Fault detection in analog electronic circuits using fuzzy inference systems and particle swarm optimization*. Alexandria Engineering Journal, 95, 376–393. doi:10.1016/j.aej.2024.01.054.
- (2025) *Soft fault diagnosis in analog electronic circuits using supervised machine learning*. Integration, 104, 102482. doi:10.1016/j.vlsi.2025.102482.

**Textos** (el de 2024, parcial).

- 2021: los circuitos son una etapa amplificadora con transistor bipolar y un amplificador de dos etapas, simulados con Monte Carlo en OrCAD. Fallos duros: abierto = 10 MΩ en serie, corto = 1 Ω en paralelo. Los **grupos de ambigüedad se determinan por simulación antes de entrenar**. Usa pocas medidas porque no todos los puntos son accesibles.
- 2024: Sallen-Key paso banda y una etapa amplificadora; abiertos y cortos con resistencias en serie y en paralelo, con Monte Carlo sobre las tolerancias; sistema de inferencia difusa ajustado con enjambre de partículas.
- 2025: Sallen-Key paso banda con componentes del 5 %. Seis predictores: la tensión en dos puntos (OUT y M1) a tres frecuencias. Quince clases: nominal más desviación baja y alta de cada uno de siete componentes (C1, C2, R1–R5), con intervalos uniformes de Monte Carlo. ANN 97,92 % y SVM 97,22 % en prueba; el bosque aleatorio sobreajusta (99,39 % en entrenamiento, 93,06 % en prueba) y confunde clases con fallo con la nominal.
- **Para qué citarlos**: son el precedente metodológico más cercano (medidas restringidas, tolerancias, clasificadores clásicos, grupos de ambigüedad). Nuestra diferencia es el criterio de gravedad, las medidas autoadquiribles, los grupos previstos por sensibilidad y los electrodos.
- **Comparación con nuestro modelo de fallo**: el corto coincide (1 Ω en paralelo); el abierto es 10 MΩ en serie frente a nuestro 1 GΩ. Nuestro circuito tiene resistencias de polarización de 10 MΩ, así que 10 MΩ en serie no representaría un abierto; se mantiene 1 GΩ. Las clases de fallo blando son por desviación del componente, como en la variante porcentual de E4.

### Zhang, Yang y Han (2026). *DiffDA-Net*

Scientific Reports, artículo en prensa (sin volumen). doi:10.1038/s41598-026-60313-3. **Texto.**

- Aborda a la vez el desequilibrio de clases y el cambio de severidad: un modelo de difusión equilibra los datos y una red con adaptación de dominio transfiere entre severidades.
- Circuitos: Sallen-Key paso banda (9 clases) y bicuadrático paso alto de cuatro operacionales (13 clases). Tolerancias del 5 % en resistencias y del 10 % en condensadores; fallos de ±50 % y ±25 %.
- Transferencia de 50 % a 25 %: 71,16 ± 2,01 % en el de 13 clases, con un protocolo que selecciona sobre el dominio objetivo y que ellos mismos presentan como cota superior; 65,61 % en el Sallen-Key; 65,07 ± 9,04 % en el escenario conjunto.
- **Para qué citarlo**: la generalización entre magnitudes es un problema reconocido y sin resolver. Nuestro E7 da una explicación sencilla de por qué: dentro de un grupo colineal, el componente solo se distingue por la magnitud.
- Añadir volumen y número de artículo cuando se publique.

### Gao (2025). *Research on medical device fault diagnosis and maintenance optimization based on big data*

Frontiers in Artificial Intelligence and Applications. doi:10.3233/faia250392. **Texto completo.**

- Paquetes de ondículas y red BP sobre un "circuito rutinario" simulado del electroencefalógrafo de un hospital; tasa de fallos no detectados inferior al 9 %.
- El texto no describe el circuito, los componentes, los fallos, las tolerancias, las especificaciones ni los electrodos.
- **Para qué citarlo**: precedente de diagnóstico de circuito en un equipo de biopotencial; hay que citarlo y diferenciarse. La diferencia está confirmada.

### Liu et al. (2021). *Intelligent fault diagnosis of medical equipment based on long short term memory network*

Journal of Biomedical Engineering (Sheng Wu Yi Xue Gong Cheng Xue Za Zhi), 38(2), 361–368. doi:10.7507/1001-5515.201912019. **Traducción automática de la página de la revista; el artículo está en chino.**

- Diagnóstico a nivel de placa sin esquemas: señales de los puertos muestreadas a 3 kHz durante 1 s y síntomas observados, siete categorías de fallo, LSTM (exactitud 0,9717, sensibilidad 0,9709, F1 0,9704).
- **Para qué citarlo**: precedente en equipo médico, a nivel de placa y con datos de reparación; no es simulación ni front-end.

### Agarwal, Sinha y Das (2022). *FauDigPro*

Int. Conf. on Maintenance and Intelligent Asset Management (ICMIAM), 1–6. doi:10.1109/icmiam56779.2022.10146898. **Texto.**

- Clasifica señales de ECG normales frente a defectuosas y pronostica el fallo del sensor; KNN al 95 %, sobre Raspberry Pi.
- La deriva y el sesgo se inyectan numéricamente con MATLAB en el ECG del conjunto WESAD. No hay circuito.
- **Para qué citarlo**: representa la literatura de fallos a nivel de señal, que detecta que el dato es malo pero no qué parte del circuito falla ni si el equipo sigue cumpliendo.

## Referencias de apoyo ya verificadas

| Para | Referencia |
|---|---|
| Grupos de ambigüedad | Starzyk, Pang, Manetti, Piccirilli y Fedi (2000). *Finding ambiguity groups in low testability analog circuits*. IEEE Trans. Circuits and Systems I, 47(8), 1125–1137. doi:10.1109/81.873868 |
| *Alternate test*, metodología | Voorakaranam, Akbay, Bhattacharya, Cherubal y Chatterjee (2007). *Signature testing of analog and RF circuits: algorithms and methodology*. IEEE Trans. Circuits and Systems I, 54(5), 1018–1031. doi:10.1109/tcsi.2007.895531 |
| Compactación de tests de especificación | Biswas, Li, Blanton y Pileggi (2005). *Specification test compaction for analog circuits and MEMS*. DATE, 164–169. doi:10.1109/date.2005.277 |
| CNN 1D en diagnóstico analógico | Yang, Meng y Wang (2020). IEEE Access, 8, 18305–18315. doi:10.1109/access.2020.2968744 |
| Fallos paramétricos con clasificadores explicables | Rajpal, Kumari y Srinivas (2026). Microelectronics Reliability, 176, 115982. doi:10.1016/j.microrel.2025.115982 |
| Electrodos secos | Joutsen et al. (2024). Scientific Reports, 14, 8882. doi:10.1038/s41598-024-56595-0 |
| Pierna derecha activa | Winter y Webster (1983). IEEE Trans. Biomedical Engineering, BME-30(1), 62–66. doi:10.1109/tbme.1983.325168 |
| Fallos de sensor de ECG, nivel de señal | Khezripour et al. (2026), Scientific Reports, 16. doi:10.1038/s41598-026-49593-x; Adams (2026), Sensors, 26(11), 3478. doi:10.3390/s26113478 |

Comprobadas después para el manuscrito, con el registro de Crossref y el resumen (Crossref u OpenAlex): Yang et al. (2020), Biswas et al. (2005, eliminación de ensayos redundantes con SVM), Khezripour et al. (2026, n.º de artículo 19210), Adams (2026) y Gao (2025, en el volumen *Design Studies and Intelligence Engineering*); de Rajpal et al. (2026) solo hay registro, sin resumen. Los títulos de JCGM 100:2008, 200:2012 y 106:2012 se contrastaron con el BIPM.

Sin verificar todavía: el dataset de IEEE DataPort (doi:10.21227/tvch-9d57, Crossref no lo devuelve), el trabajo de selección de puntos de prueba con aprendizaje por refuerzo (preprint en SSRN, doi:10.2139/ssrn.5338908) y la documentación de detección de electrodo desconectado de Texas Instruments.

## Las cinco aportaciones y su respaldo

| Aportación | Qué la respalda en la literatura | Qué la respalda en nuestros resultados |
|---|---|---|
| 1. Verificación de especificaciones y localización para autotest en servicio de una red discreta | El *alternate test* predice especificaciones y ya se ha usado para diagnosticar bloques de RF integrados (Cheng y Chang, 2010), pero en producción y a nivel de bloque; el diagnóstico con aprendizaje define el fallo por tolerancia (Srimani y Rahaman, 2022) | E3: 0,9 % de escapes con 7,8 % de falsos rechazos en el circuito integrado; E5: F1 0,87 por grupo |
| 2. Estudio sistemático con tolerancias de un front-end de biopotencial realista | Los precedentes biomédicos (Gao, 2025; Liu et al., 2021) no describen circuito, tolerancias ni especificaciones (confirmado en el texto) | 130.000 casos, dos circuitos, especificaciones de IEC 60601-2-25 |
| 3. Separación entre electrodo y circuito | La literatura de ECG trabaja a nivel de señal (Agarwal et al., 2022) y ningún trabajo localizado separa ambos orígenes | E6: 95 % de circuitos no aptos reconocidos, 64 % de fallos de electrodo; ambigüedad física entre cable abierto y electrodo desconectado |
| 4. Gravedad funcional frente a porcentual | La gravedad funcional es la del test por especificaciones (Variyam y Chatterjee, 1998); el diagnóstico con aprendizaje usa clases por porcentaje (Dieste-Velasco, 2025; Chen et al., 2025; Zhang et al., 2026) | E4: 37 % de falsos rechazos frente a 1,6 % |
| 5. Dataset y código abiertos | Existen datasets genéricos de circuitos de referencia | Primer dataset de un front-end biomédico con etiquetas de cumplimiento |

Dos resultados que no estaban entre las aportaciones del plan y merecen destacarse en el artículo: los grupos de ambigüedad se pueden anticipar con la sensibilidad sola y explican por qué la localización por componente no generaliza a magnitudes no vistas (enlaza con Chen et al., 2025, con Zhang et al., 2026 y con los grupos por simulación de Dieste-Velasco, 2021), y el circuito realista se comporta distinto del de referencia (H5).
