# -*- coding: utf-8 -*-

"""Ce module contient la classe Carte."""
from labyrinthe import *
import pickle


def creer_labyrinthe_depuis_chaine(chaine):
    grille = []
    for line in chaine.upper().split("\n"):
        grille.append([line[ii] for ii in range(len(line))])
    robot = []
    obstacles = []
    portes = []
    sorties = []
    places_libres=[]

    if chaine.upper().count("X") != 1:
        print("Il faut un unique robot pour créer un labyrinthe")
    else:
        for ii in range(0, len(grille)):
            if "X" in (grille[ii]):
                robot = (ii, "".join(grille[ii]).find("X"))
            obstacles += [(ii, jj) for jj, char in enumerate(grille[ii]) if char == "O"]
            portes += [(ii, jj) for jj, char in enumerate(grille[ii]) if char == "."]
            sorties += [(ii, jj) for jj, char in enumerate(grille[ii]) if char == "U"]
            places_libres += [(ii, jj) for jj, char in enumerate(grille[ii]) if char == " "]

    """ renvoie un objet de type labyrinthe"""
    return Labyrinthe(robot, obstacles, portes, sorties, grille, places_libres)

class Carte:
    """Objet de transition entre un fichier et un labyrinthe."""

    def __init__(self, nom, chaine):
        self.nom = nom
        self.chaine = chaine
        self.labyrinthe = creer_labyrinthe_depuis_chaine(chaine)

    def __repr__(self):
        return "<Carte {}>".format(self.nom)
