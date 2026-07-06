import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy


def train(env, agent, params=None, N=10**4, verbose=True):
    """
    Entrena el agente durante N episodios y devuelve metricas de performance.
    """
    obs, info = env.reset()

    metricas = {
        'rewards': [],
        'n_viajes': [],
        'insatisfechos': [],
        'prolongados': [],
        't_desbalanceo': []
    }

    counters = {
        'acciones': Counter(),
        'estados': Counter()
    }

    inicio = time()

    for episode in range(N):
        obs, info = env.reset()
        terminated = False
        episode_reward = 0
        
        while not terminated:
            
            state = tuple(obs[0])
            counters['estados'][state] += 1
            
            action = agent.choose_action(state)
            counters['acciones'][action] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            next_state = tuple(obs[0])
            agent.learn(state, action, reward, next_state, terminated)

            episode_reward += reward

        agent.epsilon = max(params['eps_end'], agent.epsilon * params['eps_step'])
        agent.alpha = max(params['alpha_end'], agent.alpha * params['alpha_step'])
        
        metricas['rewards'].append(episode_reward)
        metricas['n_viajes'].append(env.n_viajes)
        metricas['insatisfechos'].append(env.insatisfechos)
        metricas['prolongados'].append(env.prolongados)
        metricas['t_desbalanceo'].append(env.t_desbalanceo)

    fin = time()
    
    tiempo_mins = (fin - inicio)/60

    # Imprimir resultados
    if verbose:
        prop_insatisfechos = 100 * np.array(metricas['insatisfechos']) / np.array(metricas['n_viajes'])
        prop_prolongados   = 100 * np.array(metricas['prolongados']) / np.array(metricas['n_viajes'])
        prop_desbalanceo   = 100 * np.array(metricas['t_desbalanceo']) / (env.T * env.n_estaciones)

        def estadisticas_texto(lista_metricas):
            return f"{np.mean(lista_metricas[-1000:]):.1f} ± {np.std(lista_metricas[-1000:]):.1f}"

        print(f"Entrenamiento finalizado en {tiempo_mins:.1f} minutos.")
        print(f"# Insatisfechos: {estadisticas_texto(metricas['insatisfechos'])} ({estadisticas_texto(prop_insatisfechos)}%)")
        print(f"# Prolongados: {estadisticas_texto(metricas['prolongados'])} ({estadisticas_texto(prop_prolongados)}%)")
        print(f"Tiempo desbalanceo: {estadisticas_texto(metricas['t_desbalanceo'])} ({estadisticas_texto(prop_desbalanceo)}%)")
        print(f"Recompensa: {estadisticas_texto(metricas['rewards'])}")

    return tiempo_mins, metricas, counters, None



def test(env, qtable, N=10**4, verbose=True):
    """
    Evalua el desempeño del agente post-aprendizaje.
    """
    obs, info = env.reset()

    metricas = {
        'rewards': [],
        'n_viajes': [],
        'insatisfechos': [],
        'prolongados': [],
        't_desbalanceo': []
    }

    counters = {
        'acciones': Counter(),
        'estados': Counter()
    }

    inicio = time()

    for episode in range(N):
        obs, info = env.reset()
        terminated = False
        episode_reward = 0
        
        while not terminated:
            
            state = tuple(obs[0])
            counters['estados'][state] += 1
            
            action = np.argmax(qtable[state]) if state in qtable.keys() else 0
            counters['acciones'][action] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
        
        metricas['rewards'].append(episode_reward)
        metricas['n_viajes'].append(env.n_viajes)
        metricas['insatisfechos'].append(env.insatisfechos)
        metricas['prolongados'].append(env.prolongados)
        metricas['t_desbalanceo'].append(env.t_desbalanceo)

    fin = time()
    
    tiempo_mins = (fin - inicio)/60

    # Imprimir resultados
    if verbose:
        prop_insatisfechos = 100 * np.array(metricas['insatisfechos']) / np.array(metricas['n_viajes'])
        prop_prolongados   = 100 * np.array(metricas['prolongados']) / np.array(metricas['n_viajes'])
        prop_desbalanceo   = 100 * np.array(metricas['t_desbalanceo']) / (env.T * env.n_estaciones)

        def estadisticas_texto(lista_metricas):
            return f"{np.mean(lista_metricas[-1000:]):.1f} ± {np.std(lista_metricas[-1000:]):.1f}"

        print(f"Entrenamiento finalizado en {tiempo_mins:.1f} minutos.")
        print(f"# Insatisfechos: {estadisticas_texto(metricas['insatisfechos'])} ({estadisticas_texto(prop_insatisfechos)}%)")
        print(f"# Prolongados: {estadisticas_texto(metricas['prolongados'])} ({estadisticas_texto(prop_prolongados)}%)")
        print(f"Tiempo desbalanceo: {estadisticas_texto(metricas['t_desbalanceo'])} ({estadisticas_texto(prop_desbalanceo)}%)")
        print(f"Recompensa: {estadisticas_texto(metricas['rewards'])}")

    return tiempo_mins, metricas, counters, None



def train_test(env, agent, params=None, N=10**4, keep_best=True, verbose=True):
    """
    Similar a train pero alterna episodios con epsilon=0 (greedy) para testear la politica aprendida
    """
    obs, info = env.reset()

    metricas = {
        'rewards': [],
        'n_viajes': [],
        'insatisfechos': [],
        'prolongados': [],
        't_desbalanceo': []
    }

    counters = {
        'acciones': Counter(),
        'estados': Counter()
    }
    
    prev_epsilon = agent.epsilon

    if keep_best:
        best_avg_reward = -999
        best_qtable = dict()

    inicio = time()

    for episode in range(2*N):
        obs, info = env.reset()
        terminated = False
        episode_reward = 0
        
        while not terminated:
            if episode % 2 == 0:
                agent.epsilon = prev_epsilon
            else:
                agent.epsilon = 0

            state = tuple(obs[0])
            counters['estados'][state] += 1
            
            action = agent.choose_action(state)
            counters['acciones'][action] += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if episode % 2 == 0:
                next_state = tuple(obs[0])
                agent.learn(state, action, reward, next_state, terminated)

            episode_reward += reward

        if episode % 2 == 0:
            prev_epsilon = max(params['eps_end'], agent.epsilon * params['eps_step'])
            agent.alpha = max(params['alpha_end'], agent.alpha * params['alpha_step'])
        
        metricas['rewards'].append(episode_reward)
        metricas['n_viajes'].append(env.n_viajes)
        metricas['insatisfechos'].append(env.insatisfechos)
        metricas['prolongados'].append(env.prolongados)
        metricas['t_desbalanceo'].append(env.t_desbalanceo)

        if keep_best and len(metricas['rewards']) >= 200 and episode % 2 != 0:
            current_avg_reward = np.mean(metricas['rewards'][-199::2])
            if current_avg_reward > best_avg_reward:
                best_qtable = deepcopy(agent.q_table)
                best_avg_reward = current_avg_reward

    fin = time()
    
    tiempo_mins = (fin - inicio)/60

    print(f"Entrenamiento finalizado en {tiempo_mins:.1f} minutos.")

    final_qtable = best_qtable if keep_best else None

    return tiempo_mins, metricas, counters, final_qtable



def train_plot(rewards, window_size=100):
    weights = np.ones(window_size) / window_size
    sma = np.convolve(rewards, weights, mode='valid')
    plt.figure(figsize=(18,12))
    plt.plot(range(len(rewards)-window_size+1), sma, color='C1')
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.grid()
    plt.show()

def train_test_plot(rewards, window_size=100):
    weights = np.ones(window_size) / window_size
    sma_train = np.convolve(rewards[::2], weights, mode='valid')
    sma_test = np.convolve(rewards[1::2], weights, mode='valid')
    plt.figure(figsize=(18,12))
    plt.plot(range(len(rewards)//2-window_size+1), sma_train, color='C1', label='Entrenamiento')
    plt.plot(range(len(rewards)//2-window_size+1), sma_test, color='C0', label='Validación')
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.ylim((-400,0))
    plt.legend()
    plt.grid()
    plt.show()