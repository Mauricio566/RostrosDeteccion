# Importamos las librerias
import cv2

# Realizamos VideoCaptura
cap = cv2.VideoCapture(0)

# Leemos el modelo(arquitectura y pesos(parametros aprendidos durante el entrenamiento))
#net es un objeto, osea una instancia de la clase cv2.dnn.Net 
net = cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
layer_names = net.getLayerNames()
print("layer names ",layer_names)
# Parametros del modelo
# Tamaño
anchonet = 300
altonet = 300
# Valores medios de los canales de color
media = [104, 117, 123]
umbral = 0.7
num = 0
#bucle infinito que captura continuamente frames de la camara y se realizan
#ciertas operaciones sobre cada frame
while True:
    # Leemos los frames(returna 2 valores)
    ret, frame = cap.read()
    num = num+1
    #print("frame numero ", num, frame)

    # Si hay error(si ret es falso)
    if not ret:
        break

    # se voltea cada frame horizontalmente
    frame = cv2.flip(frame, 1)

    # Extraemos info de los frames
    #Estas líneas extraen las dimensiones del frame. frame.shape devuelve una tupla que contiene la altura, el ancho y los canales de color del frame. 
    altoframe = frame.shape[0]#altura de cada frame
    anchoframe = frame.shape[1]#ancho de cada frame
    print("alto y ancho" ,altoframe,anchoframe)

    # Preprocesamos la imagen(cada frame en el proceso iterativo)
    # Images - Factor de escala - tamaño - media de color - Formato de color(BGR-RGB) - Recorte
    #blob devuelve cada uno de los frames preprocesados
    blob = cv2.dnn.blobFromImage(frame, 1.0, (anchonet, altonet), media, swapRB = False, crop = False)
    #print("shape :)", blob.shape)
    # Corremos el modelo
    # y aqui pasamos cada frame a net(nuestra red neuronal)
    net.setInput(blob)
    #entregamos y capturamos detecciones
    #detecciones devuelve las coordenadas en X y Y donde se crea una especie de 
    #armario con un numero de cajones basado en la cantidad de rostros detectados
    detecciones = net.forward()
    #print("deteccion ",detecciones)
    #print("numero de dimensiones ",detecciones.ndim)
    #print("detecciones shape .)", detecciones.shape)#(output= (1,1,200,7)
    #print("dimension ",detecciones[0])
    #print("Segunda dimensión: ", detecciones[0][0])
    #tercera_dimension_completa = detecciones[0, 0, :, :]
    #print(tercera_dimension_completa)
    #print("Tercera dimensión: ", detecciones[0][0][1])
    #print("Cuarta dimensión: ", detecciones[0][0][1][2])
    # Iteramos(obtenemos indice de cada capa)
    for i in range(detecciones.shape[2]):#indices(i) desde 0 hasta 200(0,1,2..200)
        # Extraemos la confianza de esa deteccion
        conf_detect = detecciones[0,0,i,2]#(i)iteraciones 2(4ta dimension)nivel de confianza
        # Si superamos el umbral (70% de probabilidad de que sea un rostro)
        if conf_detect > umbral:#evaluacion realizada para cada iteracion del bucle
            # Extraemos las coordenadas
            xmin = int(detecciones[0, 0, i, 3] * anchoframe)
            ymin = int(detecciones[0, 0, i, 4] * altoframe)
            xmax = int(detecciones[0, 0, i, 5] * anchoframe)
            ymax = int(detecciones[0, 0, i, 6] * altoframe)

            # Dibujamos el rectangulo en base a las coordenadas
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            # Texto que vamos a mostrar
            label = "Confianza de deteccion: %.4f" % conf_detect

            # Tamaño del fondo del label o texto
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            #rectangulo que va sobre el texto 
            cv2.rectangle(frame, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin + base_line),
                          (0,0,0), cv2.FILLED)
            # Colocamos el texto
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("DETECCION DE ROSTROS", frame)

    t = cv2.waitKey(1)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
