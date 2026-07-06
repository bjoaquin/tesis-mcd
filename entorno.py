import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from itertools import permutations


class Estacion():
    def __init__(self, capacidad, bicis):
        self.capacidad = capacidad
        self.bicis = bicis
    
    def emitir_bici(self):
        if self.bicis > 0:
            self.bicis -= 1
            return True
        return False
    
    def recibir_bici(self):
        if self.bicis < self.capacidad:
            self.bicis += 1
            return True
        return False



class Viaje():
    def __init__(self, origen, destino, inicio, fin):
        self.origen = origen
        self.destino = destino
        self.inicio = inicio
        self.fin = fin
        self.finalizado = False
        self.tipo = "entorno" # viajes simulados (a diferencia de acciones del agente)



class Entorno(gym.Env):
    def __init__(self, tamanos, costos, tiempos, probs):
        super(Entorno, self).__init__()

        self.n_estaciones     = tamanos['n_estaciones']
        self.anclas_por_est   = tamanos['anclas_por_est']
        self.bicis_por_est    = tamanos['bicis_por_est']
        self.bicis_por_accion = tamanos['bicis_por_accion']
        
        self.costo_accion          = costos['accion']
        self.costo_accion_invalida = costos['accion_invalida']
        self.costo_est_vacia       = costos['est_vacia']
        self.costo_est_llena       = costos['est_llena']
        
        self.t_viaje  = tiempos['viaje']
        self.t_accion = tiempos['accion']
        self.t_update = tiempos['update'] # cada cuanto actualizar el tiempo de estado
        self.T        = tiempos['episodio']

        self.probs_origen  = probs['origen']
        self.probs_destino = probs['destino']
        
        self.action_space = spaces.Discrete( # nP2 + 1 (ese 1 es la inaccion)
            self.n_estaciones * (self.n_estaciones-1) + 1
        )
        self.observation_space = spaces.MultiDiscrete( # c/ estacion puede tener entre 0 y N bicis
            self.n_estaciones*[self.anclas_por_est+1] + [tiempos['estado']]
        )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(self.n_estaciones*[self.bicis_por_est] + [0], dtype=np.int32) # Estado inicial
        
        self.t = 0
        self.insatisfechos = 0
        self.prolongados = 0
        self.t_desbalanceo = 0
        self.viajes_por_fin = defaultdict(list)
        self.n_viajes = 0
        self.agente_libre_en = 0

        self.estaciones = dict()
        for e in range(self.n_estaciones):
            self.estaciones[e] = Estacion(capacidad=self.anclas_por_est, bicis=self.bicis_por_est)
        self.idx_estaciones = list(self.estaciones.keys())

        return np.array([self.state], dtype=np.float32), {}


    def step(self, action):
        reward = 0
        self.agente_libre_en = self.t

        # 1. Aplicar accion del agente
        self.actions = [None] + list(permutations(range(self.n_estaciones), 2))
        move = self.actions[action]
        if move is not None:
            origen, destino = move
        
            if self.estaciones[origen].bicis < self.bicis_por_accion:
                reward -= self.costo_accion_invalida  # accion invalida
            else:
                reward -= self.costo_accion  # costo por operacion
                self.agente_libre_en = self.t + self.t_accion
                
                for _ in range(self.bicis_por_accion):
                    self.estaciones[origen].emitir_bici()
                    self.state[origen] -= 1
                    viaje = Viaje(origen, destino, self.t, self.t + self.t_accion)
                    viaje.tipo = "agente"
                    self.viajes_por_fin[self.t + self.t_accion].append(viaje)

        # 2. Actualizar viajes (de entorno y de agente)
        while self.t <= self.agente_libre_en:

            # Finalizar viajes que terminan ahora
            for viaje in self.viajes_por_fin[self.t]:
                
                finalizado = self.estaciones[viaje.destino].recibir_bici()
                viaje.finalizado = finalizado

                if not finalizado:  # estacion llena
                    if viaje.tipo == "entorno":
                        nuevo_destino = random.choice(self.idx_estaciones)
                        viaje_extendido = Viaje(viaje.origen, nuevo_destino, viaje.inicio, self.t+self.t_viaje)
                        self.viajes_por_fin[self.t+self.t_viaje].append(viaje_extendido)
                        self.prolongados += 1
                        reward -= self.costo_est_llena  # penalizacion por estacion llena
                    else:
                        viaje_extendido = Viaje(viaje.origen, viaje.destino, viaje.inicio, self.t+1)
                        self.viajes_por_fin[self.t+1].append(viaje_extendido)
                        self.agente_libre_en += (self.agente_libre_en == self.t)  # no actualiza si ya lo hizo antes
                else: # finalizacion exitosa
                    self.state[viaje.destino] += 1
            
            del self.viajes_por_fin[self.t]

            # Emitir nuevos viajes (con cierta probabilidad)
            for e in self.idx_estaciones:
                
                if random.random() < self.probs_origen[e]:
                    destino = np.random.choice(self.idx_estaciones, p=self.probs_destino[e])
                    inicio_exitoso = self.estaciones[e].emitir_bici()
                    
                    if inicio_exitoso:
                        self.state[e] -= 1
                        viaje = Viaje(e, destino, self.t, self.t+self.t_viaje)
                        self.viajes_por_fin[self.t+self.t_viaje].append(viaje)
                        self.n_viajes += 1
                    else:  # estacion vacia
                        self.insatisfechos += 1
                        reward -= self.costo_est_vacia  # penalizacion por estacion vacia

            # Finalizar viajes recien generados (solo aplica si t_viaje=0)
            for viaje in self.viajes_por_fin[self.t]:
                
                finalizado = self.estaciones[viaje.destino].recibir_bici()
                viaje.finalizado = finalizado

                if not finalizado:  # estacion llena
                    nuevo_destino = random.choice(self.idx_estaciones)
                    viaje_extendido = Viaje(viaje.origen, nuevo_destino, viaje.inicio, self.t+self.t_viaje)
                    self.viajes_por_fin[self.t+self.t_viaje].append(viaje_extendido)
                    self.prolongados += 1
                    reward -= self.costo_est_llena  # penalizacion por estacion llena
                else: # finalizacion exitosa
                    self.state[viaje.destino] += 1


            # Contar (tiempo de) estaciones desbalanceadas
            desbalanceadas = sum(
                (e.bicis == 0) or (e.bicis == e.capacidad)
                for e in self.estaciones.values()
            )
            self.t_desbalanceo += desbalanceadas

            self.t += 1
        
            # Actualizar tiempo de estado
            if self.t >= self.state[-1] + self.t_update:
                self.state[-1] += 1

        # 3. Verificar termination
        terminated = self.t >= self.T # fin del episodio
        truncated = False
        
        return np.array([self.state], dtype=np.float32), reward, terminated, truncated, {}