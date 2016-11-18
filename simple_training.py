#!/usr/bin/python
import os
import sys
import glob
import dlib
from skimage import io

def simple_training(trainpath="Training/",nbthreads=4,cvalue=5):
    train_folder=trainpath 
    #Para treinarmos nosso dataset, chamaremos a classe train_simple_object_detector() com todos os seus valores default (que são razoavelmente bons)
    options = dlib.simple_object_detector_training_options()
    #Cartas são de certa forma simétricas, então, melhoraremos a qualidade do resultado mudando o padrão de add_left_right_image_flips para True
    options.add_left_right_image_flips = True
    #O valor de C encoraja um fitting melhor nos dados, mas um C muito grande encoraja overfitting, valor 5 ainda é experimental, precisamos de mais teste
    options.C = cvalue
    # Quantos Threads temos disponíveis para treinar em paralelo?
    options.num_threads = nbthreads
    #Vamos acompanhar o processo
    options.be_verbose = True
    #Concatene o diretório do treino e seu .xml correspondente numa string
    training_xml_path = os.path.join(train_folder, "training.xml")
    #testing_xml_path = os.path.join(train_folder, "testing.xml")
    #Finalmente chamamos a função que faz o treino de fato. Salva o resultado  em um detector .svm
    # a partir do nosso .xml (Dados do treino supervisionado)
    dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)
    print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
    print("Process done sucesssfully")

simple_training("Training/",8)
