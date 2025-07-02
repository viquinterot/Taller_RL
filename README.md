# Taller de Aprendizaje por Refuerzo: Péndulo y Super Mario Bros

**Autor:** Víctor Quintero  
**Materia:** Pensamiento de Máquina (Doctorado en IA)

---

## Resumen General

Este repositorio contiene las soluciones para el taller de Aprendizaje por Refuerzo (RL), enfocado en dos entornos clásicos con desafíos distintos: `Pendulum-v1` (control continuo) y `gym-super-mario-bros` (control desde pixeles con recompensas dispersas).

El objetivo es investigar, implementar y evaluar las arquitecturas de Deep Learning y los algoritmos de RL más apropiados para cada caso, demostrando el aprendizaje a través de métricas y visualizaciones.

---

## Proyecto 1: Control Continuo con `Pendulum-v1`

El primer proyecto se centra en el problema de control clásico de balancear un péndulo. El agente debe aprender a aplicar un torque continuo para mantener el péndulo en posición vertical.

-   **Entorno:** `Pendulum-v1` de Gymnasium.
-   **Desafío Principal:** Espacio de acción continuo.
-   **Algoritmo Investigado:** **DDPG (Deep Deterministic Policy Gradient)**, un algoritmo Actor-Crítico ideal para este tipo de problemas.
-   **Framework Utilizado:** **TF-Agents**.
-   **Arquitectura de Red:** Redes Neuronales Densas (MLP) tanto para el Actor como para el Crítico.

### Resultados Clave (Péndulo)

La curva de recompensa muestra una mejora constante, y el video comparativo evidencia cómo el agente entrenado logra estabilizar el péndulo de manera eficiente, a diferencia de un agente aleatorio.

| Gráfica de Recompensa (Péndulo) | Agente Entrenado vs. Aleatorio (Péndulo) |
| :---------------------------------: | :----------------------------------------: |
| ![Curva de Recompensa](ruta/a/grafica_pendulo.png) | *(GIF o imagen del video comparativo)* |

▶️ **Ver el cuaderno completo:** [`pendulum-v1-con-tf-agents-v-ctor-quintero.ipynb`](./ruta/a/tu/cuaderno_pendulo.ipynb)

---

## Proyecto 2: Control desde Píxeles con `Super Mario Bros`

El segundo proyecto aborda un problema significativamente más complejo: aprender a jugar Super Mario Bros directamente desde los píxeles de la pantalla.

-   **Entorno:** `gym-super-mario-bros`.
-   **Desafíos Principales:**
    1.  Espacio de estados de alta dimensionalidad (imágenes).
    2.  Recompensas extremadamente dispersas (el agente solo sabe que ha hecho algo bien al final del nivel).
-   **Algoritmo Investigado:** **PPO (Proximal Policy Optimization)**, un algoritmo robusto y eficiente para entornos complejos.
-   **Framework Utilizado:** **Stable-Baselines3**.
-   **Arquitectura de Red:** Red Neuronal Convolucional (CNN) para procesar las entradas visuales.

### Técnicas Avanzadas Implementadas

Para superar el desafío de las recompensas dispersas, se implementaron varias técnicas avanzadas:

-   **Ingeniería de Recompensas (Reward Shaping):** Se creó una función de recompensa densa para guiar al agente con señales de aprendizaje frecuentes (recompensas por avanzar, penalizaciones por morir, etc.).
-   **Exploración Basada en Novedad:** Se añadió una bonificación intrínseca para incentivar al agente a explorar nuevas áreas del mapa.

### Resultados Clave (Mario Bros)

El agente entrenado aprende políticas de comportamiento coherentes, logrando navegar los niveles, saltar sobre enemigos y evitar obstáculos.

Este proyecto se ejecutó en local ya que a pesar de usar Kaggle no fue compatible con la libreria tensorflow 2.18 y decía que usaba la GPU pero no era así. 

Se adjunta los modelos entrenados en cada caso. SAC; PPO, y el modelo final que recorrió 8890 pasos. 
