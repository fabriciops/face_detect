# USAGE
# import the necessary packages
# detect.py

from flask import Response, request
import json, time, logging, argparse
from flask import Flask
from numpy import array
import numpy
import cv2
import os

cascade_filename = "classificadores/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_filename)

app = Flask(__name__)
mylog = logging.getLogger('werkzeug')
mylog.disabled = True

@app.route("/rosto", methods=["GET", "POST"])
def obterrosto():
    
    start = time.time()
    imagem = request.files["imagem"]

    if imagem is None:
        logging.debug('Imagem não identificada')
        print('Imagem não identificada')
        return Response(json.loads({ "mensagem": "Imagem não identificada"}), 400, mimetype='application/json')
        
    
    img = cv2.imdecode(numpy.frombuffer(imagem.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    
    resultadoDeteccao = detectarRostos(img)
    if resultadoDeteccao is None:
        logging.debug('Nenhuma Face encontrada')
        logging.debug('Tempo Para Obter o Rosto: {: }'.format(time.time()-start))
        return Response(json.dumps({ "mensagem": "Nenhum Rosto identificado"}), 400, mimetype='application/json')

    
    logging.debug('Tempo obterrosto: {: } segs.'.format(time.time()-start))
    return Response(json.dumps({ "rostos": resultadoDeteccao}), 200, mimetype='application/json')

def detectarRostos(img):
    global faceCascade

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(20, 20)
    )

    if(len(faces)):
        rostos = []
        for (x,y,w,h) in faces:
            rostos.append({"coordenadas": [int(x), int(y), int(w), int(h)], "reconhecido": False})
        return rostos
    else:
        return None   


if __name__ == '__main__':	

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--log", type=str, default="INFO", help="Type log level (DEBUG, INFO, WARNING, ERROR, CRiTICAL)")
    ap.add_argument("-i", "--ip", type=str, default="0.0.0.0", help="Type IP for the API")
    ap.add_argument("-p", "--port", type=int, default=5001, help="Type port where API is listening from")

    args = vars(ap.parse_args())

    LEVELS = {'debug': logging.DEBUG,
                'info': logging.INFO,
                'warning': logging.WARNING,
                'error': logging.ERROR,
                'critical': logging.CRITICAL}
    level = LEVELS.get(args["log"], logging.INFO)

    #config logging
    logging.basicConfig(filename='log/detect.log', format='[%(asctime)s] {%(pathname)s:%(lineno)d} [%(levelname)s] %(message)s', level=level)

    ip = args["ip"]
    port = args["port"]
    #frame_count = 90
    print('\nAPI Face Detect started. Ip: {} - Port: {}\n\n' .format(ip, port))
    logging.info('API Face Detect started. Ip: {} - Port: {}' .format(ip, port))
    app.run(host=ip, port=port)

