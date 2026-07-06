import random
import numpy as np
from collections import defaultdict
from itertools import permutations

class NullAgent:
    def __init__(self, *args):
        self.alpha, self.gamma, self.epsilon = 0, 0, 0

    def choose_action(self, state):
        return 0

    def learn(self, *args):
        pass



class RandomAgent:
    def __init__(self, action_size, *args):
        self.action_size = action_size
        self.alpha, self.gamma, self.epsilon = 0, 0, 0

    def choose_action(self, state):
        return random.randint(0, self.action_size - 1)

    def learn(self, *args):
        pass



class HeuristicAgent:
    def __init__(self, action_size, *args):
        self.n_estaciones = int(( 1 + (4*action_size - 3)**0.5 ) / 2) # inverse of |A|-1 = nP2
        self.action_space = [None] + list(permutations(range(self.n_estaciones), 2))
        self.alpha, self.gamma, self.epsilon = 0, 0, 0

    def choose_action(self, state):
        s = state[:-1]
        if max(s) > 7 and min(s) < 3:
            return self.action_space.index((np.argmax(s), np.argmin(s)))
        else:
            return 0

    def learn(self, *args):
        pass



class HeuristicAgent2:
    def __init__(self, action_size, *args):
        self.n_estaciones = int(( 1 + (4*action_size - 3)**0.5 ) / 2) # inverse of |A|-1 = nP2
        self.action_space = [None] + list(permutations(range(self.n_estaciones), 2))
        self.alpha, self.gamma, self.epsilon = 0, 0, 0

    def choose_action(self, state):
        s = state[:-1]
        if max(s) >= 7 and min(s) <= 1:
            return self.action_space.index((np.argmax(s), np.argmin(s)))
        else:
            return 0

    def learn(self, *args):
        pass



class MonteCarloAgent:
    def __init__(self, action_size, alpha=0.1, gamma=1.0, epsilon=0.9):
        self.action_size = action_size
        self.alpha = 1  # no se usa este parametro
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = defaultdict(lambda: np.full(self.action_size, 0.0))
        self.returns_count = defaultdict(lambda: np.zeros(action_size))

        self.episode = [] # Experiencias del episodio actual

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, terminated):
        self.episode.append((state, action, reward))
        
        if terminated:  # MC solo aprende al final del episodio
            returns = []
            G = 0

            for state, action, reward in reversed(self.episode):
                G = reward + self.gamma * G
                returns.append(G)
            returns.reverse()
            
            visited = set()  # Actualizar primera visita de cada (s,a)

            for (state, action, reward), G in zip(self.episode, returns):

                if (state, action) in visited:
                    continue

                visited.add((state, action))

                self.returns_count[state][action] += 1
                n = self.returns_count[state][action]

                self.q_table[state][action] += (G - self.q_table[state][action]) / n

            self.episode.clear()



class QLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=1.0, epsilon=0.9):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = defaultdict(lambda: np.full(self.action_size, 0.0))


    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore: random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: best known action


    def learn(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state][action]
        
        if terminated:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * best_next_q
        
        # Update rule
        self.q_table[state][action] += self.alpha * (target - current_q)



class GuidedQLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=1.0, epsilon=0.9):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_estaciones = ( 1 + (4*action_size - 3)**0.5 ) // 2
        self.action_space = [None] + list(permutations(range(self.n_estaciones), 2))
        
        self.q_table = defaultdict(lambda: np.full(self.action_size, 0.0))


    def choose_action(self, state):
        
        def heuristica(state):
            s = state[:-1]
            if max(s) > 7 and min(s) < 3:
                return self.action_space.index((np.argmax(s), np.argmin(s)))
            else:
                return 0
        
        if random.uniform(0, 1) < self.epsilon:
            if random.uniform(0, 1) < 0.8:
                return heuristica(state)
            else:
                return random.randint(0, self.action_size - 1)  # Explore: random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: best known action


    def learn(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state][action]
        
        if terminated:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * best_next_q
        
        # Q-Learning update formula (Temporal Difference)
        new_q = current_q + self.alpha * (target - current_q)
        
        # Update the table
        self.q_table[state][action] = new_q