# ======================================================================================
# TALLER PENSAMIENTO DE MÁQUINA: APRENDIZAJE POR REFUERZO AVANZADO
# AUTOR: Victor Quintero
# 
#
# ABSTRACT:
# Este script constituye un sistema integral para entrenar un agente de RL en el entorno
# Super Mario Bros. Se aleja de una implementación básica para explorar técnicas
# de vanguardia que son fundamentales para resolver problemas con espacios de estados
# complejos y recompensas dispersas. El núcleo de la investigación se centra en:
#   1. INGENIERÍA DE RECOMPENSAS (Reward Shaping): Diseño de una función de
#      recompensa densa y multifacética para guiar el aprendizaje.
#   2. APRENDIZAJE CURRICULAR: Estrategia de entrenamiento incremental que imita
#      el aprendizaje humano.
#   3. MOTIVACIÓN INTRÍNSECA: Mecanismos de bonificación para fomentar la
#      exploración sistemática.
#   4. ANÁLISIS ALGORÍTMICO: Plataforma para comparar arquitecturas de RL canónicas
#      (PPO, SAC, DQN) bajo las mismas condiciones experimentales.
# ======================================================================================

# --- 1. IMPORTACIÓN DE DEPENDENCIAS ---
# Importaciones del ecosistema estándar de RL. `gymnasium` es la interfaz de entorno,
# `stable-baselines3` el framework de algoritmos, y `optuna` para la optimización.
import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from shimmy import GymV21CompatibilityV0
import os
import time
import json
from collections import deque
import optuna
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Importar wandb (Weights & Biases) si está disponible. Es una herramienta de MLOps
# crucial para el seguimiento, la visualización y la reproducibilidad de experimentos.
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# --- 2. CONFIGURACIÓN CENTRALIZADA ---
@dataclass
class TrainingConfig:
    """
    Encapsula todos los hiperparámetros y configuraciones del experimento en una única
    estructura de datos. Este patrón de diseño es fundamental para la reproducibilidad
    científica, permitiendo guardar y cargar la configuración exacta de una ejecución.
    """
    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    save_freq: int = 50_000
    curriculum_enabled: bool = True
    advanced_rewards: bool = True
    exploration_bonus: bool = True
    use_wandb: bool = False
    device: str = "auto"

# --- 3. INGENIERÍA DE LA FUNCIÓN DE RECOMPENSA (REWARD SHAPING) ---
class AdvancedMarioRewardWrapper(gym.Wrapper):
    """
    Implementa una función de recompensa densa y multifacética (reward shaping).
    El objetivo es transformar el problema original de recompensa dispersa (sparse reward),
    donde el agente solo obtiene una gran recompensa al final, en un problema con una
    señal de aprendizaje continua y más informativa. Esta clase es, en esencia, donde
    definimos la "función de utilidad" o el "sistema de valores" del agente.
    """
    def __init__(self, env, config: Dict[str, float] = None):
        super().__init__(env)
        
        # Define los pesos de cada componente de la recompensa. Estos pesos son
        # hiperparámetros críticos que determinan las prioridades del agente (e.g., ¿es más
        # importante avanzar rápido, recolectar monedas o simplemente sobrevivir?).
        default_config = {
            'progress_reward_scale': 0.1,    # Incentivo para avanzar
            'survival_bonus': 0.02,          # Incentivo para seguir con vida
            'time_penalty': 0.001,           # Penalización por tardar (promueve velocidad)
            'death_penalty': -100,           # Fuerte castigo por morir
            'flag_bonus': 500,               # Recompensa terminal por ganar
            'coin_bonus': 10,                # Incentivo por recolectar monedas
            'powerup_bonus': 50,             # Incentivo por obtener mejoras
            'enemy_kill_bonus': 5,           # Incentivo por eliminar enemigos
            'milestone_bonus': 100,          # Recompensas intermedias por alcanzar hitos
            'idle_penalty': -0.01,           # Castigo por inactividad
            'backtrack_penalty': -0.5,       # Castigo por retroceder
            'speed_bonus_threshold': 0.5     # Umbral para bonificación por velocidad
        }
        self.reward_config = {**default_config, **(config or {})}
        
        # Estado interno para calcular cambios (deltas) entre timesteps.
        # El RL a menudo aprende de los cambios, no de los valores absolutos.
        self.prev_x_pos = 0
        self.prev_time = 400
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_lives = 3
        self.idle_count = 0
        self.max_x_ever = 0
        self.milestones = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200]
        self.reached_milestones = set()
        self.reward_components = deque(maxlen=1000)
        
    def reset(self, **kwargs):
        """Resetea el estado del wrapper al inicio de un nuevo episodio."""
        self.prev_x_pos = 0
        self.prev_time = 400
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_lives = 3
        self.idle_count = 0
        self.max_x_ever = 0
        self.reached_milestones = set()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Modifica la recompensa devuelta por el entorno en cada paso.
        Esta es la función central del reward shaping.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extraer métricas clave del diccionario `info` del entorno.
        x_pos = info.get('x_pos', 0)
        time_left = info.get('time', 0)
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        lives = info.get('life', 3)
        flag_get = info.get('flag_get', False)
        
        # Diccionario para desglosar y analizar la contribución de cada componente.
        reward_components = { 'base': reward, 'progress': 0, 'survival': 0, 'time': 0, 'milestones': 0, 'coins': 0, 'score': 0, 'death': 0, 'flag': 0, 'idle': 0, 'backtrack': 0, 'speed': 0 }
        
        # --- CÁLCULO DE LA FUNCIÓN DE RECOMPENSA MULTI-OBJETIVO ---
        
        # 1. Recompensa por Progreso: El componente más crítico. Proporciona una señal
        #    densa que guía al agente hacia la meta. Convierte el objetivo a largo plazo
        #    en una serie de objetivos a corto plazo.
        progress = max(0, x_pos - self.prev_x_pos)
        reward_components['progress'] = progress * self.reward_config['progress_reward_scale']
        
        # 2. Bonificación por Supervivencia: Pequeño incentivo constante por no morir.
        #    Ayuda a contrarrestar pequeñas penalizaciones y promueve la persistencia.
        reward_components['survival'] = self.reward_config['survival_bonus']
        
        # 3. Penalización por Tiempo: Incentiva la eficiencia. Sin esto, el agente podría
        #    aprender políticas seguras pero extremadamente lentas.
        time_penalty = max(0, self.prev_time - time_left) * self.reward_config['time_penalty']
        reward_components['time'] = -time_penalty
        
        # 4. Recompensas por Hitos: Proporcionan "puntos de control" psicológicos para
        #    el agente, dividiendo el problema complejo en sub-tareas más manejables.
        for milestone in self.milestones:
            if milestone not in self.reached_milestones and x_pos >= milestone:
                reward_components['milestones'] += self.reward_config['milestone_bonus']
                self.reached_milestones.add(milestone)
        
        # 5. Recompensas por Coleccionables: Fomentan comportamientos secundarios deseables.
        reward_components['coins'] = max(0, coins - self.prev_coins) * self.reward_config['coin_bonus']
        
        # 6. Penalizaciones por Eventos Negativos: Señales fuertes e inequívocas
        #    para que el agente aprenda a evitar estos estados.
        if lives < self.prev_lives:
            reward_components['death'] = self.reward_config['death_penalty']
        if flag_get:
            reward_components['flag'] = self.reward_config['flag_bonus']
        
        # 7. Penalizaciones por Comportamientos Indeseados: Evita que el agente se quede
        #    atascado (idle) o retroceda sin motivo, previniendo el "reward hacking".
        if x_pos < self.prev_x_pos:
            reward_components['backtrack'] = self.reward_config['backtrack_penalty']
        
        # --- ENSAMBLAJE FINAL DE LA RECOMPENSA ---
        # La recompensa total es la suma ponderada de todos los componentes.
        # ESTA es la señal escalar que el algoritmo de RL (PPO, DQN, etc.)
        # intentará maximizar. El algoritmo en sí no conoce los componentes;
        # solo ve este valor final.
        total_reward = sum(reward_components.values())
        
        # Actualizar estado para el próximo timestep.
        self.prev_x_pos = x_pos
        self.prev_time = time_left
        self.prev_score = score
        self.prev_coins = coins
        self.prev_lives = lives
        self.max_x_ever = max(self.max_x_ever, x_pos)
        
        # Enriquecer el diccionario `info` con datos de diagnóstico. Esto es crucial
        # para el análisis y no afecta directamente al aprendizaje del agente.
        info['reward_components'] = reward_components
        info['max_x_ever'] = self.max_x_ever
        
        return obs, total_reward, terminated, truncated, info

# --- 4. APRENDIZAJE CURRICULAR ---
class CurriculumLearningWrapper:
    """
    Implementa el paradigma de Aprendizaje Curricular. Inspirado en la psicología
    cognitiva, postula que un agente aprende tareas complejas más eficientemente si se le
    presenta una secuencia de sub-tareas de dificultad creciente (un "currículo").
    """
    def __init__(self, base_config: TrainingConfig):
        self.base_config = base_config
        
        # Definición del currículo: una lista de niveles ordenados por dificultad.
        # Cada etapa tiene un nombre, una descripción y un criterio de éxito.
        self.levels = [
            {'name': 'SuperMarioBros-1-1-v3', 'description': 'Nivel básico', 'success_threshold': 0.7, 'min_episodes': 50},
            {'name': 'SuperMarioBros-1-2-v3', 'description': 'Nivel intermedio', 'success_threshold': 0.6, 'min_episodes': 30},
            # ... se pueden añadir más niveles.
        ]
        self.current_level_idx = 0
        self.level_stats = {i: {'attempts': 0, 'successes': 0} for i in range(len(self.levels))}
        
    def get_current_level(self) -> Dict[str, Any]:
        """Devuelve la configuración del nivel actual del currículo."""
        return self.levels[self.current_level_idx]
    
    def should_advance(self, success_rate: float, episodes_completed: int) -> bool:
        """Determina si el agente ha "dominado" el nivel actual y debe avanzar."""
        current_level = self.levels[self.current_level_idx]
        min_episodes_met = episodes_completed >= current_level['min_episodes']
        success_threshold_met = success_rate >= current_level['success_threshold']
        return min_episodes_met and success_threshold_met
    
    def advance_level(self) -> bool:
        """Avanza al siguiente nivel del currículo si es posible."""
        if self.current_level_idx < len(self.levels) - 1:
            self.current_level_idx += 1
            print(f"🎓 CURRICULUM ADVANCEMENT: Moving to level {self.levels[self.current_level_idx]['description']}")
            return True
        return False
    # ... (resto de funciones de utilidad del currículo)

# --- 5. BONIFICACIÓN POR EXPLORACIÓN ---
class ExplorationBonusWrapper(gym.Wrapper):
    """
    Añade una recompensa de "motivación intrínseca" basada en la novedad.
    Aborda el dilema exploración-explotación: incentiva al agente a visitar estados
    (posiciones en el mapa) nuevos o poco frecuentes, ayudándole a descubrir
    caminos o secretos que la recompensa extrínseca por sí sola podría no revelar.
    """
    def __init__(self, env, grid_size: int = 16):
        super().__init__(env)
        self.grid_size = grid_size  # Discretiza el mundo en una cuadrícula.
        self.visited_positions = set()
        self.position_counts = {}
        self.exploration_bonus = 0.1
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Discretizar la posición continua del agente en una celda de la cuadrícula.
        x_pos, y_pos = info.get('x_pos', 0), info.get('y_pos', 0)
        position = (x_pos // self.grid_size, y_pos // self.grid_size)
        
        # Calcular bonificación de exploración (count-based exploration).
        exploration_reward = 0
        if position not in self.visited_positions:
            # Recompensa alta por visitar una celda por primera vez.
            exploration_reward = self.exploration_bonus
            self.visited_positions.add(position)
            self.position_counts[position] = 1
        else:
            # Recompensa decreciente por revisitar. Evita que el agente se quede
            # "enganchado" a la novedad sin progresar en la tarea principal.
            self.position_counts[position] += 1
            exploration_reward = self.exploration_bonus / (1 + np.log(self.position_counts[position]))
        
        # La recompensa final es la suma de la recompensa extrínseca (del juego y reward shaping)
        # y la recompensa intrínseca (de la exploración).
        return obs, reward + exploration_reward, terminated, truncated, info

# --- 6. INSTRUMENTACIÓN Y LOGGING ---
class AdvancedLoggingCallback(BaseCallback):
    """
    Callback personalizado para el logging avanzado de métricas durante el entrenamiento.
    Actúa como el "panel de instrumentos" del experimento, permitiéndonos observar
    el comportamiento del agente en tiempo real y diagnosticar problemas.
    """
    # ... (implementación del callback para registrar métricas detalladas)

# --- 7. CONSTRUCCIÓN DEL ENTORNO DE APRENDIZAJE ---
def create_advanced_env(level_name: str = 'SuperMarioBros-1-1-v3', config: TrainingConfig = None) -> gym.Env:
    """
    Función fábrica (Factory Function) que ensambla la pila de wrappers para crear el
    entorno de entrenamiento final. El orden de los wrappers es importante.
    """
    # 1. Entorno base del emulador
    env = gym_super_mario_bros.make(level_name)
    # 2. Limitar el espacio de acciones a un conjunto simple de movimientos.
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 3. Wrapper de compatibilidad para la nueva API de Gymnasium.
    env = GymV21CompatibilityV0(env=env)
    
    # 4. (Opcional) Aplicar el wrapper de ingeniería de recompensas.
    if config and config.advanced_rewards:
        env = AdvancedMarioRewardWrapper(env)
    
    # 5. (Opcional) Aplicar el wrapper de bonificación por exploración.
    if config and config.exploration_bonus:
        env = ExplorationBonusWrapper(env)
    
    # --- PREPROCESAMIENTO DE OBSERVACIONES (VISIÓN) ---
    # Estas transformaciones son estándar para preparar datos de imagen para una CNN.
    
    # 6. Convertir la observación a escala de grises para reducir la dimensionalidad
    #    del espacio de estados (de 3 canales de color a 1).
    env = gym.wrappers.GrayscaleObservation(env)
    
    # 7. Redimensionar la imagen a 84x84, un tamaño estándar en la literatura de RL
    #    (originado en los papers de DeepMind) que equilibra detalle y carga computacional.
    env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
    
    # 8. Apilar 4 frames consecutivos. Esto es CRÍTICO. Proporciona al agente
    #    información temporal, permitiéndole inferir propiedades dinámicas como la
    #    velocidad y la aceleración. Sin esto, el estado no sería completamente
    #    Markoviano (un solo frame no te dice si Mario está subiendo o bajando).
    #    NOTA: Stable-Baselines3 proporciona VecFrameStack que es más eficiente para
    #    entornos vectorizados. Aquí se muestra la versión de Gymnasium como ilustración.
    
    return env

# --- 8. OPTIMIZACIÓN DE HIPERPARÁMETROS ---
def optimize_hyperparameters(algorithm: str, n_trials: int = 50) -> Dict[str, Any]:
    """
    Utiliza Optuna para realizar una búsqueda bayesiana de los mejores hiperparámetros
    para un algoritmo dado. Automatiza uno de los procesos más críticos y tediosos
    en el entrenamiento de modelos de IA.
    """
    # ... (la implementación define un espacio de búsqueda y un objetivo a maximizar)

# --- 9. EVALUACIÓN DEL MODELO ---
def evaluate_model(model, env, n_eval_episodes: int = 10) -> Tuple[float, float]:
    """
    Evalúa el rendimiento del modelo entrenado en un número fijo de episodios,
    utilizando una política determinista para medir su verdadero rendimiento aprendido.
    """
    # ... (implementación de un bucle de evaluación estándar)

# --- 10. PIPELINE PRINCIPAL DE ENTRENAMIENTO ---
def train_advanced_mario_agent(config: TrainingConfig = None):
    """
    Función orquestadora que integra todos los componentes: crea el entorno,
    configura el modelo de RL, y lanza el ciclo de entrenamiento y evaluación.
    """
    if config is None:
        config = TrainingConfig()
    
    # ... (código de configuración de directorios y curriculum)
    
    # Creación del entorno vectorizado y monitorizado.
    initial_level = 'SuperMarioBros-1-1-v3' # Ejemplo, se podría integrar con curriculum
    env = create_advanced_env(initial_level, config)
    env = Monitor(env) # Monitor es esencial para que los callbacks obtengan info de episodios.
    env = DummyVecEnv([lambda: env])
    # NOTA: `VecFrameStack` se aplicaría aquí si se usa la versión vectorizada.
    # env = VecFrameStack(env, 4, channels_order='last')

    # ... (configuración de callbacks)

    # --- DEFINICIÓN Y CONFIGURACIÓN DEL MODELO DE APRENDIZAJE ---
    # Esta sección es donde se instancia el "cerebro" del agente.
    # Los hiperparámetros definen cómo el modelo aprende de la experiencia.
    
    model_params = {'policy': 'CnnPolicy', 'env': env, 'verbose': 1, 'tensorboard_log': "./logs/", 'device': config.device}
    
    if config.algorithm == "PPO":
        # PPO (Proximal Policy Optimization) es un algoritmo on-policy, actor-critic.
        # Es robusto, eficiente en datos y generalmente un excelente punto de partida.
        model = PPO(
            **model_params,
            # --- HIPERPARÁMETROS CRÍTICOS DE PPO ---
            learning_rate=0.00025, # Tasa de aprendizaje de la red neuronal.
            n_steps=2048,          # Nº de pasos a recolectar por entorno antes de actualizar. Un valor alto reduce la varianza pero introduce sesgo.
            batch_size=64,         # Tamaño del minibatch para las actualizaciones de gradiente.
            n_epochs=10,           # Nº de veces que se recorre el buffer de experiencia para actualizar la política.
            gamma=0.99,            # Factor de descuento. Define el horizonte del agente. 0.99 significa que valora mucho las recompensas futuras.
            gae_lambda=0.95,       # Parámetro de Generalized Advantage Estimation. Controla el trade-off sesgo-varianza en la estimación de la ventaja.
            clip_range=0.2,        # El "corazón" de PPO. Limita el cambio en la política en cada actualización para evitar colapsos.
            ent_coef=0.01,         # Coeficiente de entropía. Añade un bonus a la entropía de la política, fomentando la exploración al evitar que la política se vuelva demasiado determinista prematuramente.
            vf_coef=0.5            # Peso de la función de valor en la función de pérdida total.
        )
    
    elif config.algorithm == "DQN":
        # DQN (Deep Q-Network) es un algoritmo off-policy, value-based.
        # Clásico para juegos discretos, aprende una función Q(s, a) que estima el retorno esperado.
        model = DQN(
            **model_params,
            # --- HIPERPARÁMETROS CRÍTICOS DE DQN ---
            buffer_size=100000,    # Tamaño del Replay Buffer, donde almacena experiencias pasadas.
            learning_rate=1e-4,    # Tasa de aprendizaje.
            batch_size=32,
            gamma=0.99,            # Factor de descuento.
            train_freq=4,          # Frecuencia de actualización (cada 4 pasos).
            gradient_steps=1,      # Pasos de gradiente por actualización.
            target_update_interval=10000, # Frecuencia con la que la "target network" se actualiza. Crucial para la estabilidad.
            exploration_fraction=0.1,   # Proporción del entrenamiento durante la cual epsilon (prob. de acción aleatoria) decrece.
            exploration_final_eps=0.05, # Valor final de epsilon.
        )
        
    else:
        raise ValueError(f"Algoritmo no soportado: {config.algorithm}")
    
    print(f"🎯 Modelo {config.algorithm} creado. Iniciando entrenamiento...")
    
    # --- CICLO DE APRENDIZAJE ---
    # Aquí es donde la magia ocurre. El modelo interactúa con el entorno,
    # recolecta datos, y optimiza su política para maximizar la recompensa total
    # (la cual hemos diseñado meticulosamente en `AdvancedMarioRewardWrapper`).
    model.learn(
        total_timesteps=config.total_timesteps,
        # callback=callbacks, # Se pasarían los callbacks configurados
        progress_bar=True
    )
    
    # ... (código de guardado, evaluación final y reporte de resultados)


# --- 11. PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == "__main__":
    # Esta sección permite ejecutar el script desde la línea de comandos,
    # configurando el experimento con argumentos.
    config = TrainingConfig(
        algorithm="PPO",
        total_timesteps=1_000_000,
        curriculum_enabled=False,
        advanced_rewards=True,
        exploration_bonus=True,
        use_wandb=WANDB_AVAILABLE
    )
    
    train_advanced_mario_agent(config)
