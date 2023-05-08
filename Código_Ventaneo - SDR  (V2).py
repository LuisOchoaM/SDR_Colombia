#!/usr/bin/env python
# coding: utf-8

# ## Implememtació de ventaneo  - SDR

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.10.4.0

#%matplotlib inline
#%matplotlib qt

#Se importan todas las librerías necesarias, esto lo hace automáticamente GNU-Radio al generar la plantilla
from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
import sip
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.fft import logpwrfft
import threading
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename


from gnuradio import qtgui

#Dentro de esta función es donde se definen todos los metodos y atributos.
class DataGNU(gr.top_block, Qt.QWidget): 

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "DataGNU")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.variable_function = variable_function = 0
        self.samp_rate = samp_rate = 1e6
        self.nfft = nfft = 4096
        self.Frecuencia = Frecuencia = 100e6 
        self.Average = Average = 0.05
        self.frec_val = self.Frecuencia  #Frecuencia central
        self.Bw = 40e6             #Ancho de banda


        ##################################################
        # Blocks
        ##################################################
        self.blocks_probe = blocks.probe_signal_vf(nfft)
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            nfft,
            0,
            1.0,
            "Frequency [Hz]",
            "Power [dBm]",
            "",
            1, # Number of inputs
            None # parent
        )
        
        
        #########################################################################
                                  # Implememtació de ventaneo 
        ###########################################################################

        def _variable_function_probe():
            while True:
                sint1 = self.blocks_probe.level() #Se gurdan los datos en el vector sint
                if np.std(sint1)!= 0: 
                    
                    #Se define el eje de las frecuencias, con respecto a la frecuencia central y el ancho de banda.
                    x_f = np.linspace(self.frec_val-(self.Bw//2),self.frec_val+(self.Bw//2),self.nfft) 
                    
                    # Crear un arreglo para almacenar las mediciones
                    mediciones = []
                    # Tomar las mediciones
                    colores = ['g', 'y', 'r', 'c', 'orange'] # lista de colores
                    for i in range (5):
                        medicion = self.blocks_probe.level() #Se gurdan los datos en el vector sint
                        mediciones.append(medicion)
                        
                        #plot
                        plt.plot(x_f, medicion, color=colores[i], label=f"Espectro_Signal {i+1}")  #plot(x_f, y_FTT) 
                        print(len(medicion))
                        time.sleep(2)
                    
                    #plt.close()
                    plt.savefig(f"Espectroooo.png")
                    # Sumar las sublistas punto a punto
                    suma = [sum(x) for x in zip(*mediciones)]
                    # Calcular el promedio de la suma
                    sint = [x / len(mediciones) for x in suma] #Señal Promedio
                    
                    plt.plot(x_f, sint, color='b', label=f"Espectro_Signal_Promedio")  #plot(x_f, y_FTT) 
                    plt.ylim([-100,-50])   #Escala eje Y
                    plt.xlabel('Frequency [Hz]')
                    plt.ylabel('Power [dBm]')
                    plt.legend(loc = "upper right") #leyendas
                    plt.grid()
                    plt.rcParams["figure.figsize"] = (14, 10) #tamano de la grafica
                    plt.savefig(f"Espectro_Prom.png")
                    
                    ventana=30 #Tamaño de la ventana
                    overlap=0 #Size del solapamiento
                    size_ventana=0    #A partir de esta longitud se toma la nueva ventana en cada iteraccíon 
                    umbral_spectrum=np.mean(sint) + 5 #Potencia promedio del especteo más 5 dB por encima del piso del ruido como lo recomienda la ITU 
                    Delta=self.Bw/(1e3*self.nfft) #en KHz
                    
                    #Ptencia promedio del ruido en todo el espectro:  -86.20302343554795  (dBm)
                    
                    print("Tamaño de la FFT:", self.nfft, " --> ",self.Bw/1e6, "MHz. "" --> ",Delta, "KHz entre muestras.")
                    print("Tamaño de la ventana:", ventana, " --> ",Delta*ventana/1e3, "MHz. ")
                    print("Tamaño del solapamiento:", overlap," --> ",Delta*overlap/1e3, "MHz. " "\r\n")
                    
                    dataFrame=[]            #Base de datos
                    datos=[]                #Datos por cada ventana
                    ventanas=[]             #Lista con los datos de potencia [dBm] de cada ventana temporalmente
                    vect_frecuencias=[]     #Lista con los datos de freciencia[Hz] de cada ventana temporalmente
                    frecuencias=[]          #Lista de todas las frecuencias obtenidas en los puntos maximos de potencia
                    diff_umbral=[]
                    num_ventana=[]
                    i=1                     #Contador de ventanas
                    
                    #Escala de la grafica en el eje Y (dBm)
                    y_max=-50
                    y_min=-100
                    
                    while((size_ventana)<len(sint)):
                        if size_ventana==0:
                            ventanas= sint[0:ventana] #se toma el primer segmento de ventana sin traslape
                            vect_frecuencias=x_f[0:ventana]
                            #se agrega al Dataframe las potencias que superen un umbral definido.
                            umbral=np.mean(ventanas) + 4 #Potencia promedio de las ventanas más 5 dB
                            diff_umbral.append(umbral - umbral_spectrum)     #se gurda la diferencia entre umbral estatico definido y el umbral dinamico en cada  ventana
                            num_ventana.append(i)                            #se obtine el # de la ventana y se agrga a una lista
                            if (np.max(ventanas)>=umbral):
                                datos.append(i)
                                datos.append(np.mean(ventanas))                         #Potencia promedio
                                datos.append(umbral)
                                datos.append(np.sum([n**2 for n in ventanas])/len(sint))#Potencia acumulada por ventana
                                datos.append(np.max(ventanas))                          #Potencia Máxima
                                datos.append(vect_frecuencias[ventanas.index(np.max(ventanas))]) #se obtiene la frecuencia que corresponde al maximo
                                datos.append(0)
                                frecuencias.append(vect_frecuencias[ventanas.index(np.max(ventanas))]) #también se gurdar las frecuencias en un lista a parte
                                dataFrame.append(datos)                                #Se agrega los datos obtenidos de cada ventana al DataFrame
                                
                                plt.figure(figsize=(7,3))
                                plt.scatter(vect_frecuencias[ventanas.index(np.max(ventanas))],np.max(ventanas),color="r",marker="v",label = "P_Máxima") #se grafica el punto maximo de potencia
                                plt.axhline(umbral,color='b', linestyle='--',label = "Umbral")   #se grafica el promedio de la potencia
                                plt.axhline(np.mean(ventanas),color='y', linestyle='--',label = "P_Promedio")   #se grafica el promedio de la potencia
                                plt.plot(x_f[0:ventana], sint[0:ventana],color="g",label = "Espectro_Signal")  #plot(x, y_FTT) 
                                plt.ylim([y_min,y_max])
                                plt.xlabel('Frequency [Hz]')
                                plt.ylabel('Power [dBm]')
                                plt.legend(loc = "upper right")
                                print("Ventana-> [{} - {}] MHz.".format(x_f[0]/1e6,x_f[ventana]/1e6)," -> Tamaño = {}".format(Delta*ventana/1e3), "MHz. ")
                                #Frecuencia inicial y final de la ventana
                                plt.grid()
                                plt.show()
                                
                            i+=1
                            size_ventana+=ventana-overlap

                        elif(len(sint[size_ventana:size_ventana + ventana])==ventana and (size_ventana+ventana)<len(sint)-1):
                            ventanas= sint[size_ventana:size_ventana + ventana] #se toma el segmento de ventana con traslape
                            vect_frecuencias=x_f[size_ventana:size_ventana + ventana]
                            
                            #se cuentan las frecuencias repetidas
                            if vect_frecuencias[ventanas.index(np.max(ventanas))] not in frecuencias:
                                #se agrega al Dataframe las potencias que superen un umbral definido.
                                umbral=np.mean(ventanas) + 4 #Potencia promedio de las ventanas más 5 dB
                                diff_umbral.append(umbral - umbral_spectrum) # se calcula la diferencia entre umbral constante y dinamico
                                num_ventana.append(i)            
                                if (np.max(ventanas)>=umbral):
                                    datos=[]
                                    datos.append(i)
                                    datos.append(np.mean(ventanas))
                                    datos.append(umbral)
                                    datos.append(np.sum([n**2 for n in ventanas])/len(sint)) #Potencia acumulada por ventana
                                    datos.append(np.max(ventanas))                           #
                                    datos.append(vect_frecuencias[ventanas.index(np.max(ventanas))]) #se obtiene la frecuencia que corresponde al maximo
                                    datos.append(0)    #para conservar el tamaño de la lista y poder agregarla al dataFrame
                                    frecuencias.append(vect_frecuencias[ventanas.index(np.max(ventanas))]) #también se gurdar las frecuencias en un lista a parte
                                    dataFrame.append(datos)
                                    
                                    plt.figure(figsize=(7,3))
                                    plt.scatter(vect_frecuencias[ventanas.index(np.max(ventanas))],np.max(ventanas),color="r",marker="v",label = "P_Máxima") #se grafica el punto maximo de potencia
                                    plt.axhline(umbral,color='b', linestyle='--',label = "Umbral")   #se grafica el promedio de la potencia
                                    plt.axhline(np.mean(ventanas),color='y', linestyle='--',label = "P_Promedio")   #se grafica el promedio de la potencia
                                    plt.plot(x_f[size_ventana:size_ventana + ventana],sint[size_ventana:size_ventana + ventana],color="g",label = "Espectro_Signal")  #plot(x, y_FTT)
                                    plt.ylim([y_min,y_max])
                                    plt.xlabel('Frequency [Hz]')
                                    plt.ylabel('Power [dBm]')
                                    plt.legend(loc = "upper right")
                                    print("Ventana-> [{} - {}] MHz.".format(x_f[size_ventana]/1e6,x_f[size_ventana + ventana]/1e6)," -> Tamaño = {}".format(Delta*ventana/1e3), "MHz. ") #Se imprime la frecuencia inicial y final de cada ventana
                                    plt.grid()
                                    plt.show()

                            else:
                                datos[0]=i   #las ventanas repetidas  tambien se enúmeran
                                datos[-1]+=1 #se cuentan los valores de potencia repetidas

                            i+=1
                            size_ventana+=ventana-overlap

                            
                        elif(size_ventana<len(sint)-1):     #si las mustras que faltan es menor a tamaño de la ventana se grafican a parte
                            ventanas= sint[size_ventana:len(sint)]
                            vect_frecuencias=x_f[size_ventana:len(sint)]
                            
                            #se cuentan las frecuencias repetidas
                            if vect_frecuencias[ventanas.index(np.max(ventanas))] not in frecuencias:
                                #se agrega al Dataframe las potencias que superen un umbral definido.
                                umbral=np.mean(ventanas) + 4 #Potencia promedio de las ventanas más 5 dB 
                                diff_umbral.append(umbral - umbral_spectrum)  
                                num_ventana.append(i) #se obtine el # de la ventana y se agrga a una lista
                                if (np.max(ventanas)>=umbral):
                                    datos=[]
                                    datos.append(i)
                                    datos.append(np.mean(ventanas))
                                    datos.append(umbral)
                                    datos.append(np.sum([n**2 for n in ventanas])/len(sint)) #Potencia acumulada por ventana
                                    datos.append(np.max(ventanas))
                                    datos.append(vect_frecuencias[ventanas.index(np.max(ventanas))]) #se obtiene la frecuencia que corresponde al maximo
                                    datos.append(0)     #para conservar el tamaño de la lista y poder agregarla al dataFrame
                                    dataFrame.append(datos)
                                    frecuencias.append(vect_frecuencias[ventanas.index(np.max(ventanas))]) #también se gurdar las frecuencias en un lista a parte

                                    plt.figure(figsize=(7,3))   
                                    plt.scatter(vect_frecuencias[ventanas.index(np.max(ventanas))],np.max(ventanas),color="r",marker="v",label = "P_Máxima") #se grafica el punto maximo de potencia
                                    plt.axhline(umbral,color='b', linestyle='--',label = "Umbral")   #se grafica el promedio de la potencia
                                    plt.axhline(np.mean(ventanas),color='y', linestyle='--',label = "P_Promedio")   #se grafica el promedio de la potencia  
                                    plt.plot(x_f[size_ventana:len(sint)], sint[size_ventana:len(sint)],color="g",label = "Espectro_Signal")  #plot(x_f, y_FTT) 
                                    plt.ylim([y_min,y_max])
                                    plt.xlabel('Frequency [Hz]')
                                    plt.ylabel('Power [dBm]')
                                    plt.legend(loc = "upper right")
                                    print("Ventana-> [{} - {}] MHz.".format(x_f[size_ventana]/1e6,x_f[len(sint)-1]/1e6)," -> Tamaño = {}".format(Delta*ventana/1e3), "MHz. ")  #Frecuencia inicial y final de la ventana
                                    plt.grid()
                                    plt.show()
                            else:
                                datos[0]=i
                                datos[-1]+=1

                            size_ventana=len(sint)
                            

                             
                    #Se crea el DataFrame        
                    columnas = ['#Ventana','P_Promedio (dBm)','Umbral (dBm)',"P_Acumulada (dBm)", 'P_Maxima (dBm)','Frecuencia (Hz)','#Repeticiones']
                    df = pd.DataFrame(dataFrame, columns=columnas)
                    
                    #A continuación se eliminan las frecuencias adyacentes con valores muy muy cercanos entre ellas, ya que corresponden 
                    #a mismo pico y solo se deja el valor de frecuencia en la que se haya obtenido el mayor valor de potencia.
                    
                    EPSILON = 200e3   #se define un unmbral para decidir que tan cercanas son 2 frecuencias adyacentes
                    diferencias=(df['Frecuencia (Hz)'].diff().abs() < EPSILON).astype(int) #se calcula la diferencia entre frecuencias consecutivas
                    diferencias=diferencias.tolist()
                    indice=[indice for indice, dato in enumerate(diferencias) if dato == 1] #posición en las que se encuentran esas frecuencias en el DataFrame
                    
                    #se comparan esas frecuencias en el DataFrane con su frecuencia adyasente anterior y se elimina la que tenga menor valor de potencia
                    for i in indice: 
                        if df['P_Maxima (dBm)'][i]>=df['P_Maxima (dBm)'][i-1]:
                            df=df.drop([i-1],axis=0)    
                            
                        else:
                            df=df.drop([i],axis=0)
                            
                    df.reset_index(drop=True, inplace=True) #Se reordenan los indices del DataFrame despues de eliminar las filas de frecuencia repetidas
                    
                    #Se calcula el Ancho de banda aproximado de cada señal detectada.
                    pbw = []
                    for i in range(len(df['P_Maxima (dBm)'])):
                        pot_max = df['P_Maxima (dBm)'][i]
                        indice = sint.index(pot_max) #indice d ela potencia maxima d ecada señal detectada
                        pot_umbral = pot_max - 6  # umbral de potencia
                        #se cuentan los indices que estań por encima del umbral alrededor de cada potencia maxima y se convierten a frecuencia
                        pbw_left = np.where(sint[:indice] < pot_umbral)[0][-1] #retorna una tupla [0]= array de indices; [-1]=último indice
                        pbw_right = np.where(sint[indice:] < pot_umbral)[0][0] + indice #para obtener el índice real de este valor en la lista completa, se debe sumar el índice 
                        pbw.append((pbw_right - pbw_left) * Delta)
                        
                    #se calcula la SNR de cada señal detectada
                    def calculate_snr(signal_power_dbm, noise_power_dbm):
                        signal_power_watts = 10**((signal_power_dbm - 30) / 10)  # Convertir dBm a Watts
                        noise_power_watts = 10**((noise_power_dbm - 30) / 10)  # Convertir dBm a Watts
                        return 10 * np.log10(signal_power_watts / noise_power_watts)


                    loc = len(df.columns) - 1  # Índice de la penúltima columna
                    df.insert(loc, 'BW (KHz)', pbw)
                    df.insert(loc+1, 'SNR', calculate_snr(df['P_Maxima (dBm)'], df['P_Promedio (dBm)']))
                    df = df.round(3)  # se reondea todos los números a 2 cifras decimales.
                    df.index = df.index + 1 #Para que el indice del dataFrame empiece en 1 y no en 0
                    
                    print("Potencia de Umbral: ",umbral_spectrum)
                    print("P_Promedio del espectro: ",np.mean(sint),' (dBm)')
                    print("P_Promedio de las P_Maximas: ",np.mean(df['P_Maxima (dBm)']),' (dBm)')
                    #print("Varianza: ",df['P_Maxima (dBm)'].var()) #Del los puntos de potencia máxima
                    print("Desviació estandar P_Maximas: ",np.sqrt(df['P_Maxima (dBm)'].var()), ' (dBm)''\r\n') #Del los puntos de potencia máxima
                    
                    #Se grafica todo el espectro 
                    datos=[]
                    datos.append(np.mean(sint))
                    datos.append(np.max(sint))
                    datos.append(x_f[sint.index(np.max(sint))]) #se obtiene la frecuencia que corresponde al maximo
                    
                    plt.rcParams["figure.figsize"] = (14, 10) #tamano de la grafica
                    
                    # se cre una lista de etiquetas para enumerar los picos detectados
                    for i, (x,y) in enumerate(zip(df['Frecuencia (Hz)'], df['P_Maxima (dBm)'])):
                        plt.annotate(f"{i+1}", (x,y))


                    # Graficar los puntos de frecuencia en el eje x con una altura de cero en el eje y
                    
                    #plt.axhline(np.mean(df['P_Promedio (dBm)']),color='k', linestyle='--',label = "P_Promedio_Ventanas")   #se grafica el promedio de todas las potencias promedios de cada ventana
                    plt.axhline(np.mean(sint),color='k', linestyle='--',label = "P_Promedio")   #se grafica el promedio de la potencia de toda la señal
                    plt.axhline(np.mean(df['P_Maxima (dBm)']),color='g', linestyle='--',label = "P_Máxima_Promedio")   #se grafica el promedio de la potencia maxima de cada ventana
                    #plt.scatter(df['Frecuencia (Hz)'], ceros, color="r", marker="v", label="Frecuencias")
                    plt.scatter(df['Frecuencia (Hz)'],df['P_Maxima (dBm)'],color="r",marker="v",label = "P_Detectada")
                    #plt.scatter(x_f[sint.index(np.max(sint))],np.max(sint),color="r",marker="v",label = "P_Máxima") #se grafica el punto maximo de potencia
                    plt.plot(x_f,sint,color='b',label = "Espectro_Signal")  #plot(x_f, y_FTT) 
                    #plt.axhline(np.mean(df['P_Maxima (dBm)']) + np.sqrt(df['P_Maxima (dBm)'].var()),color='y', linestyle='--',label = "Desv_Estandar")   #se grafica el promedio de la potencia maxima de cada ventana
                    #plt.axhline(np.mean(df['P_Maxima (dBm)']) - np.sqrt(df['P_Maxima (dBm)'].var()),color='y', linestyle='--')   #se grafica el promedio de la potencia maxima de cada ventana
                    #plt.axhline(umbral ,color='r', linestyle='--',label = "Umbral") #Umbral
                
                    plt.ylim([y_min,y_max])   #Escala eje Y
                    plt.xlabel('Frequency [Hz]')
                    plt.ylabel('Power [dBm]')
                    plt.legend(loc = "upper right",ncol=2) #leyendas
                    plt.grid()
                    #plt.show()
                    
                    #Se genener una imagen.jpg del espectro
                    plt.savefig("Espectro.png")
                    
                    #Se genera un .csv,HTML del DataFrame y se imprime en pantalla el DataFrame
                    df.to_csv('datos.csv', index=False) #
                    df.to_html('/home/luis/Grupo de Investigación GITA/Notebook Jupyter/DataFrame.html')
                    print(df)
                    
                    break  #Como la fuente es un archivo y no el USRP solo se ejecuta una vez el ciclo while
                try:
                  try:
                    self.doc.add_next_tick_callback(functools.partial(self.set_variable_function,val))
                  except AttributeError:
                    self.set_variable_function(val)
                except AttributeError:
                  pass
                time.sleep(5) #Se inicia la lectura de datos depsues de varios segundos.
                
        _variable_function_thread = threading.Thread(target=_variable_function_probe)
        _variable_function_thread.daemon = True
        _variable_function_thread.start()


        self.qtgui_vector_sink_f_0.set_update_time(0.10)
        self.qtgui_vector_sink_f_0.set_y_axis((-140), 10)
        self.qtgui_vector_sink_f_0.enable_autoscale(False)
        self.qtgui_vector_sink_f_0.enable_grid(False)
        self.qtgui_vector_sink_f_0.set_x_axis_units("")
        self.qtgui_vector_sink_f_0.set_y_axis_units("")
        self.qtgui_vector_sink_f_0.set_ref_level(0)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_f_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_f_0.set_line_label(i, labels[i])
            self.qtgui_vector_sink_f_0.set_line_width(i, widths[i])
            self.qtgui_vector_sink_f_0.set_line_color(i, colors[i])
            self.qtgui_vector_sink_f_0.set_line_alpha(i, alphas[i])

        self._qtgui_vector_sink_f_0_win = sip.wrapinstance(self.qtgui_vector_sink_f_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_vector_sink_f_0_win)
        self.logpwrfft_x_0 = logpwrfft.logpwrfft_c(
            sample_rate=samp_rate,
            fft_size=nfft,
            ref_scale=2,
            frame_rate=30,
            avg_alpha=Average,
            average=True,
            shift=True)
        
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        ##################################################
        # Ruta del archivo
        ##################################################
        filename =askopenfilename() # show an "Open" dialog box and return
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, filename, True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.logpwrfft_x_0, 0))
        self.connect((self.logpwrfft_x_0, 0), (self.blocks_probe, 0))
        self.connect((self.logpwrfft_x_0, 0), (self.qtgui_vector_sink_f_0, 0))
        

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "DataGNU")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.logpwrfft_x_0.set_sample_rate(self.samp_rate)

def main(top_block_cls=DataGNU, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()


# In[ ]:




