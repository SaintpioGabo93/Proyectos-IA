       # ----- Parte 1 ----------------- #

# Implementamos las librerías.
import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


# Creamos la arquitectura de la IA

class Red_Neuronal(nn.Module):

    def __init__(self,tamanio_accion):
        # Programamos las capas de convolución, para este ejemplo vamos a crear la capa de flattening en una linea diferente
        super(Red_Neuronal, self).__init__()
        self.convolucion_1 = nn.Conv2d(in_channels= 4, out_channels= 32, kernel_size= (3, 3), stride= 2)
        self.convolucion_2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= (3, 3), stride= 2)
        self.convolucion_3 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= (3, 3), stride= 2)
        # Flattening
        self.flatten = nn.Flatten()
        # Red Neuronal
        self.fc_1 = nn.Linear(512, 128)
        # Esta linea nos va a sacar dos tensores, uno de valores de acciones, y o el otro será crítico o de estado
        self.fc_2accion = nn.Linear(128, tamanio_accion) # Este no prove de una predicción del valor más probable del estado actual
        self.fc_2_state = nn.Linear(128, 1) # El tamaño debe se 1, porque sólo es tensor n1


    def forward(self, estado):

        # Retropropagación a las capas convolucionales
        x = self.convolucion_1(estado)
        x = F.relu(x)
        x = self.convolucion_2(x)
        x = F.relu(x)
        x = self.convolucion_3(x)
        x = F.relu(x)
        # Retropropagación capa flattening
        x = self.flatten(x) # Esto fue lo que teníamos anteriormente → x = x.view(x.size(0),-1)
        # Retropropagación a las capas neuronales
        x = self.fc_1(x)
        x = F.relu(x)
        # Obtención de los valores de acción (Q), y los valores de estado (criticos)
        valores_accion = self.fc_2accion(x)
        valor_estado = self.fc_2_state(x)[0]
        return valores_accion, valor_estado

        # ----- Paso 2 ---------- #

# Preparamos nuestro entorno
'''
Este código va a construir nuestro entorno, y aestá hecho, construido, ni le muevas padrino, :v
'''
class PreprocessAtari(ObservationWrapper):

  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self):
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset()
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    self.frames = self.observation(obs)

def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()

forma_estado = env.observation_space.shape
numero_acciones = env.action_space.n
print("Forma del Estado:", forma_estado)
print("Número de Acciones:", numero_acciones)
print("Nombre de acciones:", env.env.env.get_action_meanings())

# Inicialización de los Hiperparámetros

tasa_aprendizaje = 1e-4
factor_descuento = 0.99
'''
El número de entornos va a a ser muy importante para este método, porque vamos a entrenara 3 agente, en tres entornos 
distintos
'''
numero_entornos = 20

# Implementamos nuestro método A3C

class Robot():
    def __init__(self, tamanio_accion):
        self.dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tamanio_accion = tamanio_accion
        '''
        En este método no vamos a hacer dos cerebros como anteriormente, que era el de objetivos y estados, vamos a 
        implementar nuestro A3C
        '''
        self.Red_Neuronal = Red_Neuronal(tamanio_accion).to(self.dispositivo)
        self.optimizador = torch.optim.Adam(self.Red_Neuronal.parameters(), lr= tasa_aprendizaje)

    #Ahora vamos a implementar el método softmax
    def act(self, estado):
        if estado.ndim == 3:
            estado = [estado]# Así creamos la dimensión adecuada para nuestro batch
        # Convertimos en un tensor de torch
        estado = torch.tensor(estado, dtype= torch.float32, device= self.dispositivo)
        valores_accion, _ = self.Red_Neuronal(estado)
        # Programamos el softmax
        poliza = F.softmax(valores_accion, dim= -1)
        return np.array([np.random.choice(len(p), p = p) for p in poliza.detach().cpu().numpy()])

    def paso(self, estado, accion, recompensa, sig_estado, hecho):

        tamanio_batch = estado.shape[0]
        estado = torch.tensor(estado, dtype= torch.float32, device= self.dispositivo)
        sig_estado = torch.tensor(sig_estado, dtype= torch.float32, device= self.dispositivo)
        recompensa = torch.tensor(recompensa, dtype=torch.float32, device=self.dispositivo)
        hecho = torch.tensor(hecho, dtype=torch.bool, device=self.dispositivo).to(dtype= torch.float32)
        valores_accion, valor_estado = self.Red_Neuronal(estado)
        _, valor_sig_estado = self.Red_Neuronal(sig_estado)
        valor_estado_objetivo = recompensa + factor_descuento*valor_sig_estado*(1-hecho) # Ecuación de Bellmann
        ventaja =valor_estado_objetivo - valor_estado
        proba = F.softmax(valores_accion, dim= -1)
        logproba = F.log_softmax(valores_accion, dim=-1)
        entropia = -torch.sum(proba*logproba, axis= -1)
        index_batch = np.arange(tamanio_batch)
        logp_acciones = logproba[index_batch, accion]
        perdida_actor = -(logp_acciones*ventaja.detach()).mean() - 0.001*entropia.mean()
        perdida_critica = F.mse_loss(valor_estado_objetivo.detach(), valor_estado)
        total_perdida = perdida_actor + perdida_critica
        self.optimizador.zero_grad()
        total_perdida.backward()
        self.optimizador.step()

# Inicializamos nuestro agente

robot = Robot(numero_acciones)

# Evaluar nuestro agente A3C con ciertos episodios

def evaluar(robot, entorno, n_episodios = 1):

    recompensa_episodios = []
    for _ in range(n_episodios):

        estado, _ = entorno.reset()
        total_recompensa = 0
        while True:
            accion = robot.act(estado)# Hacemos que nuestro agente ejecute la acción
            estado, recompensa, hecho, info, _ = entorno.step(accion[0])
            total_recompensa += recompensa
            if hecho:
                break
        recompensa_episodios.append(total_recompensa)
    return recompensa_episodios

# Metodo Asincrono
class Entorno_Batch: # Esta va a ser la clase que cree diferentes entornos
    def __init__(self, n_entornos = 0):
        self.entornos = [make_env() for _ in range(n_entornos)]

    def reset(self):
        _estados = []
        for env in self.entornos:
            _estados.append(env.reset()[0])

        return np.array(_estados)


    def paso(self, acciones):

        sig_estados, recompensas, hechos, infos, _ = map(np.array, zip(*[env.step(a)for env,a in zip(self.entornos, acciones)]))
        for i in range(len(self.entornos)):
            if hechos[i]:
                sig_estados[i] = self.entornos[i].reset()[0]

        return sig_estados, recompensas, hechos, infos



# Entrenamiento de la IA con el método A3C

import tqdm

entorno_batch = Entorno_Batch(numero_entornos)
estado_batch = entorno_batch.reset()

# Barra de progreso, para ver como va nuestro entrenamiento
with tqdm.trange(0,3001) as barra_progreso:

    for i in barra_progreso:
        acciones_batch = robot.act(estado_batch)
        sig_estado_batch, recompensa_batch, hechos_batch, _ = entorno_batch.paso(acciones_batch)
        recompensa_batch *= 0.01
        robot.paso(estado_batch, acciones_batch, recompensa_batch, sig_estado_batch, hechos_batch)
        estado_batch = sig_estado_batch

        if i % 10000 == 0:
            print('Recompensa promedio del robot', np.mean(evaluar(robot, env, n_episodios= 10)))


        # -------- Part 3 - Visualizing the results -------- #

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  env.close()
  imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(robot, env)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()
