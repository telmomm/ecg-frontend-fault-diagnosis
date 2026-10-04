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
| 1.5 | ~~Electrodo de gel~~ | **Decidido:** se declara que el electrodo de gel se representa con la red de 51 kΩ ‖ 47 nF que IEC 60601-2-25 pone en serie con cada cable; la resistencia serie de 300 Ω y la dispersión entre electrodos son supuestos del estudio | — | No |
| 1.7 | ~~Dispersión de los parámetros de electrodo~~ | **Resuelto con los datos abiertos del artículo de 2024:** cada caso con electrodo seco usa la resistencia de contacto medida en uno de los seis sujetos. Siguen siendo supuestos el factor 2 entre los electrodos de un mismo caso y los ±5 mV de media celda | — | Hecho en la versión 2 del dataset |
| 1.7b | ~~Medida de los electrodos secos~~ | **Resuelto:** la medida es de un par de electrodos en serie; los valores por electrodo son la mitad de las resistencias y el doble de la capacidad | — | Hecho en la versión 2 del dataset |
| 1.4 | ~~Simplificaciones de los ensayos~~ | **Aceptadas**; se declaran en el artículo: ruido como 6,6 × valor eficaz, margen dinámico desde el punto de trabajo, línea base leída 50 ms tras el impulso, respuesta en frecuencia por el método sinusoidal y de impulso | — | No |
| 1.8 | ~~Rechazo en modo común del INA333 a ganancia 4~~ | **Aceptada** la interpolación (102 dB; el macromodelo de TI da 100 dB) | — | No |
| 1.10 | ~~Modo monitorización~~ | **Decidido:** solo se aplica la norma de diagnóstico | — | No |

## 2. Decisiones de diseño abiertas

| # | Decisión | Situación | Opciones | ¿Re-simular? |
|---|---|---|---|---|
| 2.1 | ~~Electrodos secos porosos frente a la red de entrada~~ | **Decidido: opción (c).** Se mantienen los seis materiales y la atenuación con electrodos porosos se trata como dificultad del problema. La variante sin porosos (`--electrode-kinds`, etiqueta `gel-solid`) y la de tipo de electrodo conocido (`--known-electrode`) quedan como análisis de sensibilidad | — | No |
| 2.2 | ~~Inyección de las señales de autotest~~ | **Decidido:** `Vcal`, `Vcmt` e `Ilo` son fuentes ideales y se declaran como hipótesis del estudio | — | No |
| 2.3 | ~~Operacionales discretos~~ | **Decidido:** modelo genérico, declarado como hipótesis; no se elige pieza | — | No |
| 2.4 | ~~Nodos de continua adicionales~~ | **Decidido:** la salida del amplificador de instrumentación (`ina_out`) entra en el conjunto principal de medidas (C1); la salida de la pierna derecha (`rld_out`) sigue solo en `C1x` | — | No |
| 2.5 | ~~Tono de 0,05 Hz en C2~~ | **Resuelto por E8:** no aporta ni a la decisión ni a la localización y consume 80 de los 97 segundos del autotest | Quitarlo de la recomendación de medidas | No |
| 2.6 | ~~PySpice~~ | **Decidido:** lanzador propio de ngspice; tabla de herramientas del plan actualizada | — | No |
| 2.7 | ~~Modelo del circuito abierto~~ | **Decidido:** 1 GΩ en serie | — | No |
| 2.8 | ~~Fallos que afectan a la seguridad~~ | **Decidido:** se declara como limitación. R9 o R1/R2 en corto dejan el circuito apto según la norma de prestaciones aunque eliminan la limitación de corriente hacia el paciente; el nivel funcional no incluye criterios de seguridad eléctrica (IEC 60601-1), y se comenta en la discusión | — | No |
| 2.13 | ~~Localización por grupo de ambigüedad~~ | **Implementado:** los grupos se obtienen de la sensibilidad, sin mirar los fallos (integrado: R5+R6+R11+R12, C5+R10, R13+R14, R15+R16). E5 da el resultado por grupo como principal y E7 puntúa también por grupo | — | No |
| 2.14 | ~~Nodo de salida del amplificador de instrumentación~~ | **Decidido:** ver 2.4 | — | No |
| 2.11 | ~~Electrodo desconectado frente a R1/R2 abierta~~ | **Decidido:** un mismo grupo. En E6, R1 o R2 abierta cuenta como la clase electrodo/cable | — | No |
| 2.12 | Regresión de especificaciones con banda de guarda | Los objetivos recortados producen muchos empates y la banda de guarda pasa de aceptar demasiado a rechazarlo casi todo | Usar regresión cuantílica o un clasificador por especificación si se quiere un punto de operación fino | No |
| 2.10 | ~~Medida de modo común (C2)~~ | **Resuelto por E7:** subir la amplitud del tono a 1 V o alargarlo diez veces no hace detectables los fallos de la pierna derecha (se escapa alrededor de la mitad). Se mantiene el tono por defecto y se recomienda leer la salida de la pierna derecha si se quieren cubrir esos fallos | — | No |
| 2.9 | Desequilibrio de clases por origen | 5.000 sanos y 2.200 de electrodo frente a unos 57.000 de circuito. E6 ya pondera las clases y define el origen por lo que hay que hacer, con lo que el reparto pasa a ser 48.000 / 13.400 / 2.200 | Generar más casos de electrodo si se quiere afinar H4 | Solo si se generan más casos |

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
| 3.9 | CNN 1D sobre la respuesta al pulso | Revisada (normalización con el rango del ADC, cinco bloques y cabeza densa sin promediado temporal, planificación del ritmo de aprendizaje, GPU del Mac). En validación sube de 0,11 a unos 0,40 de F1 por componente, pero sigue por debajo del bosque aleatorio sobre los siete descriptores del pulso (0,61). No es aportación del trabajo; se informa tal cual y no se ajusta más |
| 3.8 | Métricas de testabilidad | E2 usa un criterio univariante (conservador) con mediana y rango intercuartílico, umbral de 3 desviaciones y una desviación de referencia del 10 %. Con media y desviación típica, los fallos que saturan la salida parecían indetectables; por eso se usan estadísticos robustos y se añade un detector de límites; los grupos transitivos encadenan componentes, por eso se informa también de la confusión directa. La separabilidad de E9 (distancia entre centroides sobre dispersión) se dispara con medidas saturadas, como pasa con C4. Conviene contrastarlas con los resultados de E5 |

## 4. Trabajo pendiente por fases

| Fase | Pendiente |
|---|---|
| 1. Lecturas y posicionamiento | Cerrada. Notas y borrador (`docs/lecturas_fase1.md`, `docs/paper/introduction_draft.md`) contrastados con los textos de los once trabajos. Para la redacción final: cinco referencias de apoyo citadas por su resumen (9, 10, 15, 18, 19) y dos afirmaciones sin referencia, listadas al final del borrador |
| 4. Dataset | Hecho (versión `data/v2`). Queda publicarlo en Zenodo, en la fase 8 |
| 5. Testabilidad | Hecha: H3 y H5 apoyadas (resultado en el plan) |
| 6. Modelos | Hecha: H1, H2 y H4 evaluadas, con las variantes de electrodos (resultado en el plan) |
| 7. Robustez | Hecha (resultado en el plan) |
| 8. Publicación | Todo |

## 5. Del estado del arte

Recogidos de [SOTA/sota_diagnostico_fallos_frontend_ecg.md](SOTA/sota_diagnostico_fallos_frontend_ecg.md), apartado 5.

- Confirmar el hueco principal con una búsqueda en Scopus o Web of Science.
- Localizar el artículo al que responde *Electrocardiogram failure in the operating room – manufacturer's comment* (Anaesthesia, 2018), como posible motivación clínica.
- Verificar los datos bibliográficos de las referencias (algunas fechas de OpenAlex no coinciden con las del DOI).

## 6. Repositorio

- Licencia MIT con el autor como titular. Versión actual 0.1.0; la 1.0.0 se reserva para cuando sea estable.
- Versión 2 del dataset generada y ensayos relanzados; el plan recoge sus resultados. Los manifiestos de `data/v2` apuntan al commit `6a49f58`, pero se generaron con cambios aún sin confirmar: conviene hacer commit del estado actual y anotarlo.
- Decidir si se versiona `results/openalex_results/` (9 MB) y los ficheros de `docs/SOTA/` (uno de ellos pesa 2,8 MB).
- `docs/normative_and_papers/` está excluida de git: la norma es una copia con licencia de uso de AENOR y no puede publicarse. Los dos artículos son de acceso abierto y podrían versionarse aparte.
