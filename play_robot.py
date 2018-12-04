# -*- coding: utf-8 -*-

"""Ce fichier contient le code principal du jeu.
Exécutez-le avec Python pour lancer le jeu.
"""

from actions import *

continuer_a_jouer = True
while continuer_a_jouer:
    cartes = chargement_des_cartes()
    labyrinthe = choisir_une_partie(cartes)
    afficher_labyrinthe(labyrinthe)
    jouer(labyrinthe)
    afficher_labyrinthe(labyrinthe)
    print("Votre nombre de points est {}".format(labyrinthe.nombre_points))
    continuer_a_jouer = (input('\nVoulez-vous faire une autre partie ?\n ').lower() in ["oui", "o"])