# Aprendizaje reforzado aplicado al balanceo de sistemas públicos de bicicletas

**Tesis de posgrado - Maestría en Ciencia de Datos**

**Universidad Austral - Facultad de Ingeniería**

*Alumno:* Lic. Joaquín Bermejo

*Director:* Dr. Enrique Gabriel Baquela

### Contenido

Este repositorio contiene 12 archivos que contemplan todo el análisis de datos realizado en las aplicaciones de la tesis.

#### Experimentos

* **a1.ipynb**: Experimento A1 (Comparación Monte Carlo vs Q-learning).
* **a2.ipynb**: Experimento A2 (Estudio de convergencia al incorporar temporalidad en el estado).
* **b1.ipynb**: Experimento B1 (Ajuste de hiperparámetros).
* **b2.ipynb**: Experimento B2 (Esquemas de recompensa).
* **c1.ipynb**: Experimento C1 (Saturación de demanda).
* **c2.ipynb**: Experimento C2 (Demanda asimétrica).
* **c3.ipynb**: Experimento C3 (Dinámicas de viaje inestables).

#### Scripts auxiliares

* **agentes.py**: Definición de los 6 agentes utilizados en el proyecto.
* **entorno.py**: Definición del entorno del MDP (con variantes para los distintos experimentos).
* **train.py**: Funciones para entrenar algoritmos y evaluar su desempeño.
