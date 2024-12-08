import cv2
import face_recognition as fr
import os
import numpy
import mysql.connector as mysql
from datetime import datetime

#crear una base de datos

conn = mysql.connect(host = "localhost", user = "root", password = "", port = 3306, database = "empresa")

ruta = "Empleados"

mis_imagenes = []

nombres_empleados = []
"""colocamos la lista de directorios que nos da el comando dentro de una variable."""
lista_empleados = os.listdir(ruta)

#print(lista_empleados)

"""en el siguiente bucle for se leen una por una las imagenes segun una lista de nombres extraida del directorio 
donde tenemos la imagenes. 
"""
for nombre in lista_empleados:
    #guardamos la imagen leida en una variable (extraemos la imagen con el metodo imread pasando como parametro la ruta de las imagenes.)
    imagen_actual = cv2.imread(f"{ruta}\{nombre}")
    #agregamos la imagen en una lista.
    mis_imagenes.append(imagen_actual)
    #agregamos los nombres de las imagenes sin extension dentro de una lista para rescatar solo los nombres de los empleados.
    nombres_empleados.append(os.path.splitext(nombre)[0])
    

    
"""en esta funcion codificar que pasamos como parametro la lista de imagenes previamente leidas
codificamos las iamgenes para reconocer la cara dentro de ellas
"""
def codificar(imagenes):

    #crear una lista nueva
    lista_codificada = []

    #pasar todas las imagenes a rgb.
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        #print(imagen)
        #codificar.
        codificado = fr.face_encodings(imagen)[0]
        #print(codificado)
        #agregrar a la lista.
        lista_codificada.append(codificado)
    #devolver la lista codificada.
    return lista_codificada


lista_empleados_codificada = codificar(mis_imagenes)

#print(len(lista_empleados_codificada))

#sacar una foto con la camapara del dispositivo.
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#leer la imagen.
estado, imagen = captura.read()

def fecha_actual():
    hora_actual = datetime.now()
    hora_actual = hora_actual.strftime('%H:%M:%S')
    print(hora_actual)
    return hora_actual

def registrar_asistencia_empleado(nombre_emp, fecha):
    cursor = conn.cursor()
    query = """INSERT INTO asistencia (nombre, hora) VALUES (%s,%s)"""
    cursor.execute(query, (nombre_emp, fecha))
    conn.commit()
    cursor.close()

if not estado:
    print("Error al tomar la foto.")
else:
    #reconocer cara en captura.
    cara_captura = fr.face_locations(imagen)

    #codificar cara capturada.
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)

    for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
        coincidencia = fr.compare_faces(lista_empleados_codificada, caracodif)
        distancias = fr.face_distance(lista_empleados_codificada, caracodif)

        print(distancias)
        print(coincidencia)

        indice_coincidencias = numpy.argmin(distancias)

        if distancias[indice_coincidencias] > 0.6:
            print("la cara no coincide con ninguna de la base de datos.")
        
        else:

            print("coincidencia detectada.")

            nombre = nombres_empleados[indice_coincidencias]

            y1, x2, y2, x1 = caraubic
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(imagen, nombre, (x1 +6, y2 -6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255))

            cv2.imshow("imagen capturada", imagen)
            cv2.waitKey(0)
            fecha = fecha_actual()
            registrar_asistencia_empleado(nombre, fecha)