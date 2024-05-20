        # ----- Parte 1: Construcción de la IA ------ #

# Importamos las librerías que vamos a utilizar

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

'''
Vamos a construir el cerebro y los ojos del robot, el cerebro va a ser similar, ahora solo tenemos que implementar 
los ojos, que van a ser las capas convolucionales, las imagenes de entrada, van a ser los estados.
'''
# Creamos la IA
class Red_Neuronal(nn.Module):
    def __init__(self, tamanio_accion, seed = 42):

        super(Red_Neuronal, self).__init__() # Hasta aquí hemos activado correctamente las herencias del nn.Module
        self.semilla = torch.manual_seed(seed)

        # Vamos a programar la convolución, que serían los ojos de nuestro robot
        self.convolucion_1 = nn.Conv2d(3,32, kernel_size= 8, stride= 4)# Sus argumentos serían los canales de entrada, que en este caso al tratarse de RGB son tres, canales de salida 32, tamanio_kernel 8x8, tamanio de paso del kernel o stride = 4
        self.batch_norm_layer_1 = nn.BatchNorm2d(32) # Batch normalization layer en 2d, a este le tenemos que meter el número de canales que obtivimos en la capa anterior, 32
        # Hacemos el mismo procedimiento 3 veces para tener 3 capas de convolución
        self.convolucion_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.batch_norm_layer_2 = nn.BatchNorm2d(64)
        self.convolucion_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.batch_norm_layer_3 = nn.BatchNorm2d(64)
        self.convolucion_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.batch_norm_layer_4 = nn.BatchNorm2d(128)
        # A Continuacion programamos la red neuronal
        '''
        Para programar la red neuronal, tenemos que hacer el proceso de flattengin, que es crear un solo vector unidimensional
        para que pueda ser procesado por la red neuronal
        (La formula es input size - tamaño kernel + 2*padding)/stride
        '''
        self.fcl_1 = nn.Linear(10*10*128, 512) #los argumentos son el tamanio de salida, numero de características de salida,
        self.fcl_2 = nn.Linear(512, 256)  # Full connection layer 2
        self.fcl_3 = nn.Linear(256, tamanio_accion)

    def forward(self, estado): # Bien importante, el estaod el la imagen del entorno, en este caso, del juego pac-man
        '''
        A continuación vamos a propagar la convolución al batch norm layer, para que proceda a relaizar esta convolución
        y se se descomponta la imagen
        '''
        # Las capas convolutivas de los ojos
        x  = F.relu(self.batch_norm_layer_1(self.convolucion_1(estado)))
        x = F.relu(self.batch_norm_layer_2(self.convolucion_2(x)))
        x = F.relu(self.batch_norm_layer_3(self.convolucion_3(x)))
        x = F.relu(self.batch_norm_layer_4(self.convolucion_4(x)))

        # Ahora tenemos que aplanar o flattening x para que pueda entrar en nuestra red neuronal
        x = x.view(x.size(0),-1) # Así se aplana nuestro tensor a una sola dimensión
        # Ya podemos propagar el vector obtenido en las capas de la red neuronal
        x = F.relu(self.fcl_1(x))
        x = F.relu(self.fcl_2(x))
        return self.fcl_3(x)



            # ----- Parte 2: Entrenar IA -------- #
# Importar el entorno virtual
import gymnasium as gym

env = gym.make('MsPacmanDeterministic-v0', full_action_space= False)  # Con esta variable creamos nuestro entoro que viene en la librería gymnasium
forma_del_estado = env.observation_space.shape
tamanio_del_estado = env.observation_space.shape[0]
numero_acciones = env.action_space.n

# Inicializar los hiperparámetros

tasa_aprendizaje = 5e-4 # Este valor se obtuvo a traves de la experimentación
minibatch_tamanio = 64 # Se refiere a las observaciónes usadas en un paso del entrenamiento
factor_descuento = 0.99

# Preprocesamiento de frames
'''
Este paso es para que las imagenes de entrada se puedan convertir en tensores para de PyTorch que puedan ser aceptados 
por la red neuronal de nuestra IA
'''
from PIL import Image
from torchvision import transforms

def preprocesamiento_frame(frame):
    frame = Image.fromarray(frame)# Debemos convertir nuestro arreglo de numpy en un objeto de PIL IMAGE
    preprecesamiento = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])# Ahora crearemos un objeto de preprocesamiento de nuestros frames en formato de un pil image, reduciremos las dimensiones con las convoluciones
    return preprecesamiento(frame).unsqueeze(0) # Recordar que con este método ponemos el return en su batch correspondiente

# Construcción de nuestro agente

class Robot():

    def __init__(self,tamanio_accion):
        self.dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tamanio_accion = tamanio_accion
        # Ahora toca implementar el Q-Learning
        self.Qredneuronal_local = Red_Neuronal(tamanio_accion).to(self.dispositivo)
        self.Qredneuronal_objetivo = Red_Neuronal(tamanio_accion).to(self.dispositivo)
        self.optimizador = optim.Adam(self.Qredneuronal_local.parameters(), lr= tasa_aprendizaje)
        self.memoria = deque(maxlen= 10000)# Va a ser diferente para la red convolucional

    def paso(self,estado, accion, recompensa, sig_estado, logrado):
        estado = preprocesamiento_frame(estado)
        sig_estado = preprocesamiento_frame(sig_estado)
        self.memoria.append((estado, accion, recompensa, sig_estado, logrado))
        if len(self.memoria) > minibatch_tamanio:
            experiencias = random.sample(self.memoria, k= minibatch_tamanio)
            self.aprendizaje(experiencias,factor_descuento)

    def act(self, estado, epsilon = 0.): # Esta función nos sirve para la política de selección de acción, de acuerdo con el método epsilon greedy
        estado = preprocesamiento_frame(estado).to(self.dispositivo) # En este caso utilizamos nuestra funciónde preprocesamiento
        self.Qredneuronal_local.eval()
        with torch.no_grad(): # Con esto deshabilitamos cualquier calculo de gradiente de la ligreria PyTorch
            valores_acciones = self.Qredneuronal_local(estado)
        self.Qredneuronal_local.train()
        if random.random() > epsilon:   # Con esto seleccionamos si el valor generado es mayor a epsilon, y así seleccionamos la acción con el valor Q más alto
            return np.argmax(valores_acciones.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.tamanio_accion))

    def aprendizaje(self, experiencias, factor_descuento):
        estados, acciones, recompensas, sig_estado, logrados = zip(*experiencias) # Con el zip, juntamos en pilas la información y cuando pase a los pasos del q learning con el * se van a desempaquetar
            # A continuación vamos a poner los valores con los que calculabamos la muestra en la función reejecucion de memoria, debemso respetar el orde con el que declaramos las variables anteriores

        estados = torch.from_numpy(np.vstack(estados)).float().to(self.dispositivo)
        acciones = torch.from_numpy(np.vstack(acciones)).long().to(self.dispositivo)
        recompensas = torch.from_numpy(np.vstack(recompensas)).float().to(self.dispositivo)
        sig_estado = torch.from_numpy(np.vstack(sig_estado)).float().to(self.dispositivo)
        logrados = torch.from_numpy(np.vstack(logrados).astype(np.uint8)).float().to(self.dispositivo)

        sig_Qobjetivo = self.Qredneuronal_objetivo(sig_estado).detach().max(1)[0].unsqueeze(1) #La función detach separa el tensor, y la función max, nos ayuda a seleccionar el tensor con el que queremos trabajar
        q_objetivos = recompensas + (factor_descuento*sig_Qobjetivo*(1-logrados))
        q_expectado = self.Qredneuronal_local(estados).gather(1, acciones)
        perdida = F.mse_loss(q_expectado,q_objetivos) #mse = mean square error
        self.optimizador.zero_grad()
        perdida.backward() # Así propagamos retrogradamente
        self.optimizador.step() # eso sólo hace una sola optimización


# Inicializar la clase DCQN

robot = Robot(numero_acciones)

# Entrenamiento del Robot DCQN

numero_episodios = 2000
numer_maximo_de_pasostiempo_por_episodio = 10000
valor_epsilon_inicio = 1.0
valor_epsilon_final = 0.01
valor_epsilon_corrupcion =0.995
epsilon = valor_epsilon_inicio
record_de_cien_episodios = deque(maxlen= 100)

for episodio in range(1, numero_episodios + 1):

    estado, _ = env.reset()
    record = 0

    for t in range(numer_maximo_de_pasostiempo_por_episodio):

        accion = robot.act(estado, epsilon)
        sig_estado, recompensa, logrado, _, _ = env.step(accion)
        robot.paso(estado, accion, recompensa, sig_estado, logrado)
        estado = sig_estado
        record += recompensa
        if logrado:
            break

    record_de_cien_episodios.append(record)
    epsilon = max(valor_epsilon_final, valor_epsilon_corrupcion*epsilon)
    print('\rEpisodio {}\tScore Promedio: {:.2f}'.format(episodio, np.mean(record_de_cien_episodios)), end="")
    if episodio % 100 == 0:
        print('\rEpisodio {}\tScore Promedio: {:.2f}'.format(episodio, np.mean(record_de_cien_episodios)))
    if np.mean(record_de_cien_episodios) >= 500.0:
        print('\nEntorno Resuelto en: {:d} episodios!\tScore Promedio: {:.2f}'.format(episodio - 100,
                                                                                     np.mean(record_de_cien_episodios)))
        torch.save(robot.Qredneuronal_local.state_dict(), 'checkpoint.pth')
        break


import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    estado, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()  # Obtén el frame como un arreglo numpy
        # Asegúrate de que el tamaño del frame sea divisible por 16
        height, width, _ = frame.shape
        new_height = (height // 16 + 1) * 16
        new_width = (width // 16 + 1) * 16
        frame_resized = frame[:new_height, :new_width, :]
        frames.append(frame_resized)
        action = agent.act(estado)
        estado, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(robot, 'MsPacmanDeterministic-v0')
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

