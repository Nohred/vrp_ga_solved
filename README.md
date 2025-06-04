# VRP-GA-Solved

Este repositorio contiene una implementación desde cero de un **algoritmo genético (GA)** diseñado para resolver instancias simplificadas del **Problema de Ruteo de Vehículos** (VRP). El enfoque se basa en representar soluciones como rutas factibles que cubren todos los nodos de una red, minimizando la distancia total recorrida y penalizando soluciones inválidas.

## 📌 Características principales

- Soporte para instancias VRP en formato `.dat` con matrices de distancias explícitas.
- Evaluación de soluciones basada en:
  - Distancia total recorrida (ida y vuelta al depósito).
  - Penalización por nodos no visitados o repetidos.
  - Penalización opcional por número de rutas (vehículos).
- Operadores genéticos personalizados:
  - Crossover por rutas completas.
  - Crossover PMX (Partially Mapped Crossover).
  - Mutación por intercambio y relocalización de nodos.
- Configuraciones fácilmente ajustables:
  - Tamaño de población.
  - Número de generaciones.
  - Tasa de mutación.
  - Tamaño de élite.
- Registro del historial evolutivo (fitness por generación).
- Scripts listos para pruebas y análisis comparativos.

## 🧠 Objetivo del proyecto

Explorar y analizar el comportamiento de diferentes configuraciones de un algoritmo genético aplicado al VRP, evaluando el impacto de los operadores de cruce, el tamaño poblacional y la duración evolutiva sobre la calidad y estabilidad de las soluciones.

## 📁 Estructura del proyecto

```

vrp\_ga\_solved/
│
├── model.py              # Script principal con el algoritmo genético
├── VRParser.py           # Lector de instancias .dat en formato VRP
├── data/                 # Instancias VRP (archivos .dat)
│   └── A045-03f.dat      # Instancia principal utilizada
├── results/              # Carpeta sugerida para guardar salidas y gráficas
└── README.md             # Descripción del proyecto

````

## ▶️ Ejecución básica

```bash
python model.py
````

Esto ejecutará el algoritmo con la configuración por defecto definida en `model.py`. Puedes modificar los parámetros (`generations`, `mutation_rate`, etc.) directamente desde el código o a través de configuraciones externas.

## 📊 Resultados

Los resultados de múltiples configuraciones se encuentran documentados en el informe LaTeX asociado. El algoritmo ha logrado soluciones con menos de 30 unidades de diferencia respecto al óptimo reportado en literatura para la instancia `A045-03f`.

## 📌 Notas

* El enfoque no impone un número fijo de vehículos, lo cual permite que el algoritmo optimice la estructura de rutas libremente.
* El cruce tiene un área de oportunidad importante, ya que en muchos casos la mutación resultó ser el principal mecanismo de mejora.
* La aleatoriedad del GA obliga a realizar múltiples ejecuciones para obtener resultados representativos.


## 🧑‍💻 Autor

* **Juan Ulises Calderón Huerta** – [ju.calderonhuerta@ugto.mx](mailto:ju.calderonhuerta@ugto.mx)
  Universidad de Guanajuato, DICIS


---

¿Quieres que también genere automáticamente este `README.md` como archivo para que solo tengas que hacer `git add README.md` y `git push`?
```
