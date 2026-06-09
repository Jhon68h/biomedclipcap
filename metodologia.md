# Metodología y configuración experimental

## 1. Diseño general del estudio

Se planteó un estudio experimental comparativo para evaluar la capacidad de un modelo multimodal de generar descripciones clínicas a partir de imágenes de colonoscopia. El modelo sigue el enfoque de *prefix captioning*: una red visual extrae la representación semántica del frame, un módulo de proyección transforma dicha representación en un prefijo compatible con un modelo de lenguaje y GPT-2 genera el reporte clínico de manera autorregresiva.

La variable experimental principal fue el **backbone visual**. Se compararon tres alternativas manteniendo constante el conjunto de datos, la partición de los casos, el módulo de proyección, el modelo de lenguaje, los hiperparámetros de entrenamiento y el procedimiento de evaluación. De esta forma, las diferencias observadas pueden atribuirse principalmente a la representación visual producida por cada backbone.

Las tres configuraciones evaluadas fueron:

| Configuración | Backbone visual | Característica principal |
| --- | --- | --- |
| BioMedCLIP | ViT-B/16, resolución 224, preentrenado en información biomédica | Representación especializada en el dominio médico |
| CLIP-ViT | ViT-B/32 de CLIP | Representación visual general basada en Transformer |
| CLIP-RN101 | ResNet-101 de CLIP | Representación visual general basada en convoluciones |

## 2. Conjunto de datos

El conjunto experimental estuvo compuesto por **11.400 frames de colonoscopia**, distribuidos de forma balanceada:

- 5.700 frames positivos con presencia de pólipo.
- 5.700 frames negativos sin presencia de pólipo.
- 77 casos clínicos en total.

Cada frame se vinculó con una descripción textual de referencia. Las descripciones negativas expresan la ausencia de pólipos. Las positivas incluyen, cuando la información está disponible, atributos como:

- tipo histológico o clase de lesión;
- morfología según la clasificación de París;
- tamaño en milímetros;
- localización anatómica en el colon.

Las captions se formularon mediante una estructura clínica relativamente controlada. Esto permite entrenar el componente generativo y, posteriormente, recuperar de forma sistemática los atributos clínicos presentes en el texto.

## 3. Preparación de las imágenes

Todas las imágenes se cargaron en formato RGB. Para cada backbone se aplicó el preprocesamiento definido por su modelo preentrenado, incluyendo redimensionamiento, recorte y normalización de intensidad.

Cada frame fue procesado una sola vez por el backbone correspondiente para obtener un embedding visual de 512 dimensiones. Posteriormente, los embeddings fueron normalizados mediante norma L2. La extracción anticipada de representaciones permitió desacoplar el procesamiento visual del entrenamiento del generador y aseguró que cada configuración utilizara representaciones fijas durante el aprendizaje.

No se realizó ajuste fino de los backbones visuales. Por tanto, el experimento compara directamente la utilidad de sus espacios de representación preentrenados para el dominio de colonoscopia.

## 4. Estrategia de partición

Se utilizó una validación cruzada de **dos folds con separación por caso**. Todos los frames pertenecientes a un mismo caso fueron asignados íntegramente a entrenamiento o validación dentro de cada fold.

Esta decisión evita que frames del mismo procedimiento o paciente aparezcan simultáneamente en ambos subconjuntos, lo que reduciría artificialmente la dificultad de la evaluación debido a la similitud visual entre frames consecutivos.

La semilla aleatoria utilizada para construir la partición fue **42**. Los folds resultantes fueron:

| Fold | Casos de validación | Frames positivos | Frames negativos | Total de validación |
| --- | ---: | ---: | ---: | ---: |
| Fold 1 | 39 | 2.888 | 2.850 | 5.738 |
| Fold 2 | 38 | 2.812 | 2.850 | 5.662 |

Los subconjuntos son complementarios: los casos usados para validación en un fold forman parte del entrenamiento en el otro. Se verificó que el solapamiento de casos entre entrenamiento y validación dentro de cada fold fuera igual a cero.

La partición se mantuvo idéntica para BioMedCLIP, CLIP-ViT y CLIP-RN101, garantizando una comparación pareada entre backbones.

## 5. Arquitectura multimodal

La arquitectura se compone de tres bloques:

1. **Encoder visual:** produce un embedding de 512 dimensiones para cada frame.
2. **Mapper Transformer:** convierte el embedding visual en una secuencia de prefijos.
3. **GPT-2:** genera la descripción clínica condicionada por los prefijos visuales.

El mapper recibe el embedding visual y lo proyecta a diez representaciones intermedias. Estas representaciones se combinan con diez vectores de prefijo aprendibles y se procesan mediante un Transformer de ocho capas y ocho cabezas de atención. La salida consiste en una secuencia de **10 embeddings de 768 dimensiones**, compatible con el espacio interno de GPT-2.

Los embeddings proyectados se anteponen a los tokens de la descripción. De esta manera, GPT-2 interpreta la información visual como contexto inicial para producir la secuencia textual.

## 6. Estrategia de entrenamiento

Se utilizó la variante de entrenamiento de solo prefijo. Los pesos de GPT-2 y del backbone visual permanecieron congelados; únicamente se actualizaron los parámetros del mapper Transformer.

Esta estrategia tiene tres propósitos:

- conservar el conocimiento lingüístico del modelo de lenguaje;
- reducir el costo computacional;
- aislar el aprendizaje de la correspondencia entre representación visual y reporte clínico.

La función objetivo fue la entropía cruzada autorregresiva. Para cada posición textual, el modelo debía predecir el siguiente token de la caption de referencia condicionado por el prefijo visual y los tokens anteriores. Los tokens de relleno se excluyeron del cálculo de la pérdida.

### Hiperparámetros

| Parámetro | Valor |
| --- | ---: |
| Número de folds | 2 |
| Semilla | 42 |
| Épocas | 15 |
| Tamaño de lote | 4 |
| Optimizador | AdamW |
| Tasa de aprendizaje | 2 × 10⁻⁵ |
| Pasos de calentamiento | 5.000 |
| Planificador | Decaimiento lineal con calentamiento |
| Longitud del prefijo GPT-2 | 10 |
| Longitud de la proyección visual | 10 |
| Tipo de mapper | Transformer |
| Capas del mapper | 8 |
| Cabezas de atención | 8 |
| Dimensión del embedding visual | 512 |
| Dimensión interna de GPT-2 | 768 |
| Normalización del embedding visual | Norma L2 |
| Ajuste del backbone | No |
| Ajuste de GPT-2 | No |
| Frecuencia de guardado | Un checkpoint por época |

Para cada backbone se entrenó un modelo independiente en cada fold, para un total de **seis entrenamientos principales**.

## 7. Selección del modelo e inferencia

Se conservaron checkpoints de todas las épocas. La configuración principal reportada utiliza el checkpoint de la época 15, identificado internamente como época 14 debido a la indexación desde cero.

Durante la inferencia, cada frame de validación fue procesado con el mismo backbone empleado durante el entrenamiento. El embedding obtenido se normalizó, se transformó mediante el mapper entrenado y se utilizó como prefijo de GPT-2.

La generación se realizó mediante **beam search** con las siguientes condiciones:

| Parámetro de generación | Valor |
| --- | ---: |
| Tamaño del beam | 5 |
| Longitud máxima | 67 tokens |
| Temperatura | 1,0 |
| Token de finalización | Punto (`.`) |

De las cinco secuencias candidatas se seleccionó la de mayor puntuación normalizada. Cada frame produjo una única caption final.

## 8. Protocolo de evaluación

La evaluación se realizó en dos niveles complementarios:

1. detección binaria de presencia o ausencia de pólipo;
2. calidad del reporte clínico generado.

Las predicciones de los dos folds se reunieron para calcular el desempeño global sobre los 11.400 frames. Además, se calculó la desviación estándar poblacional de cada métrica entre los dos folds.

### 8.1 Evaluación binaria por frame

La etiqueta predicha se obtuvo interpretando el contenido de la caption:

- una expresión explícita de ausencia de pólipo se clasificó como negativa;
- cualquier otra descripción válida se clasificó como positiva.

Con estas etiquetas se construyó la matriz de confusión y se calcularon:

- **Exactitud:** proporción total de predicciones correctas.
- **Precisión:** proporción de predicciones positivas que realmente contienen pólipo.
- **Sensibilidad o recall:** proporción de frames positivos detectados correctamente.
- **F1:** media armónica entre precisión y sensibilidad.
- **Especificidad:** proporción de frames negativos identificados correctamente.

Esta evaluación determina si el texto generado comunica correctamente la presencia o ausencia de lesión, aunque el modelo haya sido entrenado como generador y no como clasificador convencional.

### 8.2 Evaluación de generación textual

La similitud entre la caption generada y la referencia se evaluó mediante precisión modificada de n-gramas:

- **BLEU-1:** coincidencia de unigramas.
- **BLEU-4:** coincidencia de secuencias de cuatro tokens.

Antes de la comparación, los textos se normalizaron a minúsculas, se eliminaron espacios redundantes y se retiraron marcas especiales del modelo de lenguaje.

Los valores empleados corresponden a precisión modificada de n-gramas a nivel de corpus. No incluyen penalización por brevedad, por lo que deben interpretarse como medidas de solapamiento léxico y no como una implementación completa del BLEU canónico.

### 8.3 Evaluación de contenido clínico

Para determinar si las captions conservaban la información clínicamente relevante, se extrajeron atributos tanto de la referencia como de la predicción mediante reglas textuales normalizadas.

Se evaluaron:

- **Exactitud del tipo de lesión:** coincidencia entre el tipo de lesión de referencia y el generado.
- **Exactitud de localización:** coincidencia de la región anatómica.
- **Exactitud de morfología de París:** coincidencia de la categoría morfológica.
- **MAE de tamaño:** error absoluto medio, en milímetros, entre el tamaño de referencia y el tamaño generado.

Cada métrica clínica se calculó únicamente en las muestras donde el atributo correspondiente estaba presente en la referencia. Para el MAE también fue necesario que el tamaño pudiera recuperarse de la predicción.

## 9. Comparación entre configuraciones

El protocolo se diseñó como una comparación controlada. Entre las tres variantes solo cambia el backbone visual. Se mantuvieron constantes:

- las imágenes y captions;
- los casos asignados a cada fold;
- el preprocesamiento propio de cada modelo;
- la dimensión de salida usada por el mapper;
- la arquitectura del mapper;
- GPT-2;
- el número de épocas y el tamaño de lote;
- el optimizador y la tasa de aprendizaje;
- el método de generación;
- las métricas y reglas de extracción clínica.

El desempeño se comparó mediante las medias globales y la variabilidad entre folds. Se consideró preferible una configuración con mayor exactitud, precisión, sensibilidad, F1, especificidad, BLEU y exactitudes clínicas, junto con un menor MAE de tamaño.

## 10. Evaluación complementaria sobre videos reales

Además de la validación cruzada, el sistema contempla una evaluación complementaria sobre videos completos de colonoscopia. Los videos se descomponen en frames, se genera una caption para cada uno y las predicciones se comparan con anotaciones temporales de presencia de lesión.

En este escenario se calculan nuevamente verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos. Cuando existe información clínica de referencia para la lesión, también pueden compararse tamaño, localización e histología.

Esta etapa debe considerarse una evaluación externa y temporalmente correlacionada: los frames consecutivos de un video no son muestras independientes. Por ello, sus resultados deben reportarse por video y, cuando sea posible, complementarse con métricas por lesión o por evento, además de las métricas por frame.

## 11. Consideraciones metodológicas

La separación por caso reduce la fuga de información entre entrenamiento y validación y representa mejor la generalización a procedimientos no vistos. Sin embargo, deben considerarse las siguientes limitaciones:

- las captions presentan una estructura lingüística controlada y poca variación estilística;
- la etiqueta binaria se deriva del texto generado;
- las métricas clínicas dependen de la extracción correcta de atributos mediante reglas;
- la precisión modificada de n-gramas utilizada no incorpora penalización por brevedad;
- dos folds permiten usar todos los casos, pero ofrecen una estimación limitada de la variabilidad;
- la evaluación por frame en video puede favorecer secuencias con muchos frames visualmente similares.

En consecuencia, la metodología permite comparar de manera consistente los tres backbones, pero una validación clínica posterior debería incluir más casos externos, análisis por lesión y revisión cualitativa por especialistas.
