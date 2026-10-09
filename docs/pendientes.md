# Pendientes y puntos por contrastar

Lista única de todo lo que queda abierto en la línea. Cada punto indica qué falta, cómo cerrarlo y si obliga a volver a simular el dataset. El diseño y la procedencia de cada valor están en [circuit.md](circuit.md); el plan, en [linea_diagnostico_fallos_frontend_ecg.md](linea_diagnostico_fallos_frontend_ecg.md).

**Sobre la columna "¿Re-simular?":** si solo cambia un límite de especificación basta `ecgfd relabel --data <carpeta>`, que recalcula las etiquetas con los valores ya guardados. Si cambia el circuito, una red de ensayo o los electrodos, hay que generar el dataset de nuevo.

## 1. Contraste con fuentes

Revisado con la norma UNE-EN 60601-2-25:2016 (idéntica a IEC 60601-2-25:2011) y los dos artículos de `docs/normative_and_papers/`.

### Cerrado

| # | Punto | Resultado |
|---|---|---|
| 1.1 | Respuesta en frecuencia de 40 a 150 Hz | Confirmado: +10 % / −30 % (tabla 201.107, ensayos B y C). Añadido el ensayo D: hasta 500 Hz la respuesta no puede superar +10 % |
| 1.2 | Margen dinámico de entrada | Confirmado: ±5 mV, también con ±300 mV de offset (201.12.4.107.2). La ganancia elegida es válida |
| 1.3 | Resto de límites | Confirmados todos: amplitud 5 %, 0,67–40 Hz ±10 %, impulso 0,1 mV y 0,30 mV/s, rechazo con 10 V eficaces y 1 mV pico a valle, ruido 30 µV, impedancia de entrada 2,5 MΩ (pérdida ≤ 20 %) |
| 1.3b | Montaje de los ensayos | Ajustado al texto: rechazo en modo común a 50 y 60 Hz, con el desequilibrio en cada cable por turno, sin offset y con ±300 mV; ruido con la red en todos los cables; impedancia de entrada con ±300 mV |
| 1.6 | Electrodos secos | Sustituidos los rangos genéricos por las medianas medidas de seis materiales (*Scientific Reports* 2024, tabla 3) |
| 1.9 | Muestreo y cuantificación | Comprobado contra 201.12.4.107.3: 1000 muestras/s (mínimo 500) y 2,9 µV por LSB referidos a la entrada (máximo 5 µV) |

### Sigue abierto

Ninguno de los dos artículos da estos datos, así que quedan como supuestos declarados.

| # | Punto | Estado actual | Cómo cerrarlo | ¿Re-simular? |
|---|---|---|---|---|
| 1.5 | Electrodo de gel | Rd ‖ Cd es la red de 51 kΩ ‖ 47 nF que la norma usa para representar el electrodo; la resistencia serie de 300 Ω es un supuesto. El artículo de *Sensors* 2022 no mide electrodos de gel | Buscar una fuente con parámetros medidos de Ag/AgCl con gel, o declarar que se usa la red de la norma | Sí |
| 1.7 | Dispersión de los parámetros de electrodo | Factor 2 en torno a cada mediana y ±5 mV en el potencial de media celda, supuestos. El artículo de 2024 solo publica medianas en la tabla; los datos por sujeto están en IEEE DataPort y Mendeley Data | Calcular la dispersión real con esos datos abiertos | Sí |
| 1.7b | Medida de los electrodos secos | Los parámetros se ajustaron a una medida entre dos electrodos; si representan el par y no un solo electrodo, Rd sería la mitad y Cd el doble | Aclararlo con el método del artículo o con sus datos | Sí |
| 1.4 | Simplificaciones de los ensayos | Ruido como 6,6 × valor eficaz en vez de pico a valle en 10 s; margen dinámico calculado desde el punto de trabajo y no con la señal de 40 Hz desplazada; línea base leída 50 ms tras el impulso; respuesta en frecuencia por el método sinusoidal y de impulso, no por los ECG de calibración | Decidir si se aceptan y declararlas en el artículo | Sí, si se cambia algún ensayo |
| 1.8 | Rechazo en modo común del INA333 a ganancia 4 | 102 dB interpolado entre los datos de ganancia 1 y 10; el macromodelo de TI da 100 dB | Aceptar la interpolación o consultar a TI | Sí, si cambia |
| 1.10 | Modo monitorización | Solo se aplica la norma de diagnóstico; el plan cita también IEC 60601-2-27, que no está en la carpeta | Decidir si se añade como segundo juego de límites | No |

## 2. Decisiones de diseño abiertas

| # | Decisión | Situación | Opciones | ¿Re-simular? |
|---|---|---|---|---|
| 2.1 | **Electrodos secos porosos frente a la red de entrada** | Con los valores medidos, un circuito sano ve en servicio entre 0,31 y 0,61 de la ganancia nominal con tela conductora (0,11–0,21 a 150 Hz) y entre 0,70 y 0,87 con polímero; con gel y metales sólidos, 0,94–1,02. Lo causan los 10 MΩ de polarización y el condensador diferencial de 1 nF | (a) Limitar la familia seca a acero, plata y platino; (b) rediseñar la entrada para electrodos secos (más impedancia, menos capacidad, o electrodos activos); (c) dejarlo así y tratarlo como dificultad del problema. El dataset `v1` incluye los seis materiales | (a) no: basta filtrar por `electrode_kind`; (b) sí; (c) no |
| 2.2 | Inyección de las señales de autotest | `Vcal`, `Vcmt` e `Ilo` son fuentes ideales | Dejarlo como hipótesis declarada, o diseñar el circuito real de inyección | Sí, si se diseña |
| 2.3 | Operacionales discretos | Modelo genérico; no hay pieza elegida | Elegir pieza y ajustar offset, ancho de banda y ruido | Sí |
| 2.4 | Nodos de continua adicionales | `ina_out` y `rld_out` suponen canales de ADC libres; solo entran en el conjunto `C1x` | Decidir si el equipo los tendría | No |
| 2.5 | Tono de 0,05 Hz en C2 | Es el que ve el paso alto, pero medirlo lleva decenas de segundos | Mantenerlo, o confiar en la cola de la respuesta al pulso | No |
| 2.6 | PySpice | Sustituido por `spicefault` (≥ 0.3), la librería extraída de este repositorio; la tabla de herramientas del plan ya lo recoge | Confirmar el cambio | No |
| 2.7 | Modelo del circuito abierto | 1 GΩ en serie, en lugar de los 10 MΩ de la versión 1 | Confirmar | Sí, si cambia |
| 2.8 | Fallos que afectan a la seguridad | R9 o R1/R2 en corto dejan el circuito apto según IEC 60601-2-25, pero eliminan la limitación de corriente hacia el paciente | Añadir un criterio de seguridad (corriente de paciente, IEC 60601-1) al nivel funcional, o declararlo como limitación | No, si se calcula con los valores guardados; sí, si hace falta una simulación nueva |
| 2.10 | Medida de modo común (C2) | El tono de 0,1 V inyectado por la pierna derecha no ve ningún componente con el ruido supuesto, y los fallos de la pierna derecha que incumplen el rechazo en modo común no se detectan (circuito de referencia, R9 abierta en el integrado) | Subir la amplitud o promediar más el tono, o medir la salida de la pierna derecha (`rld_out`, ya en C1x) | No: amplitud, ruido y promediado se aplican al cargar los datos |
| 2.9 | Desequilibrio de clases por origen | 5.000 sanos y 2.200 de electrodo frente a unos 57.000 de circuito | Ponderar las clases o submuestrear en E6, o generar más casos sanos y de electrodo | Solo si se generan más casos |

## 3. Sin implementar

Nada de esto impide avanzar; son ampliaciones.

| # | Elemento | Nota |
|---|---|---|
| 3.1 | Medida de contacto en continua (variante de C4) | Solo existe la variante en alterna |
| 3.2 | Fallos de offset negativo en operacionales | Solo se inyecta offset positivo |
| 3.3 | Filtro de muesca de 50 Hz | Era opcional en la versión 1 del plan |
| 3.4 | Corriente de polarización, ruido 1/f y velocidad de subida del INA333 | No modelados |
| 3.5 | Tonos medidos en transitorio | Las ganancias de C2 y C4 son de pequeña señal; el modelo de medida recorta al rango del ADC |
| 3.6 | Modelo de ruido de medida | Primera aproximación; debe seguir el procedimiento real de adquisición cuando se defina |
| 3.7 | Ruido del macromodelo de TI | Dio valores no creíbles en ngspice y no se comparó |
| 3.8 | Métricas de testabilidad | E2 usa un criterio univariante (conservador) con mediana y rango intercuartílico, umbral de 3 desviaciones y una desviación de referencia del 10 %. Con media y desviación típica, los fallos que saturan la salida parecían indetectables; por eso se usan estadísticos robustos y se añade un detector de límites; los grupos transitivos encadenan componentes, por eso se informa también de la confusión directa. La separabilidad de E9 (distancia entre centroides sobre dispersión) se dispara con medidas saturadas, como pasa con C4. Conviene contrastarlas con los resultados de E5 |

## 4. Trabajo pendiente por fases

| Fase | Pendiente |
|---|---|
| 1. Lecturas y posicionamiento | Todas las lecturas y el borrador de introducción |
| 4. Dataset | Hecho (versión `data/v1`). Queda publicarlo en Zenodo, en la fase 8 |
| 5. Testabilidad | Hecha: H3 apoyada, H5 apoyada en parte (resultado en el plan). Contrastar con E5 |
| 6. Modelos | E3–E6 corren con modelos sin ajustar y una sola partición; faltan validación para hiperparámetros, particiones repetidas, bandas de guarda en E3 y clases fusionadas por grupo de ambigüedad en E5 |
| 7. Robustez | Faltan E7 y E8 (ya existe la partición por magnitud no vista) |
| 8. Publicación | Todo |

## 5. Del estado del arte

Recogidos de [SOTA/sota_diagnostico_fallos_frontend_ecg.md](SOTA/sota_diagnostico_fallos_frontend_ecg.md), apartado 5.

- Confirmar el hueco principal con una búsqueda en Scopus o Web of Science.
- Localizar el artículo al que responde *Electrocardiogram failure in the operating room – manufacturer's comment* (Anaesthesia, 2018), como posible motivación clínica.
- Verificar los datos bibliográficos de las referencias (algunas fechas de OpenAlex no coinciden con las del DOI).

## 6. Repositorio

- No hay ningún commit todavía.
- La licencia MIT tiene un titular genérico; falta poner el nombre real y, si se quiere, un fichero de cita.
- Decidir si se versiona `results/openalex_results/` (9 MB) y los ficheros de `docs/SOTA/` (uno de ellos pesa 2,8 MB).
- `docs/normative_and_papers/` está excluida de git: la norma es una copia con licencia de uso de AENOR y no puede publicarse. Los dos artículos son de acceso abierto y podrían versionarse aparte.
