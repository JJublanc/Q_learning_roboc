from actions import *
from carte import *
from labyrinthe import *

# On choisit une carte
carte_choisie = chargement_des_cartes()[0]

# On imprime la carte
afficher_labyrinthe(carte_choisie.labyrinthe)
carte_choisie.labyrinthe.robot

# On effectue un movement vers l'ouest et on l'affiche
carte_choisie.labyrinthe.executer_instruction("o1")
afficher_labyrinthe(carte_choisie.labyrinthe)

# On effectue un movement vers le nord et on l'affiche
carte_choisie.labyrinthe.executer_instruction("n1")
afficher_labyrinthe(carte_choisie.labyrinthe)
carte_choisie.labyrinthe.robot

# On effectue un movement vers l'est et on l'affiche
carte_choisie.labyrinthe.executer_instruction("e1")
afficher_labyrinthe(carte_choisie.labyrinthe)

# On effectue un movement vers le sud et on l'affiche
carte_choisie.labyrinthe.executer_instruction("s3")
afficher_labyrinthe(carte_choisie.labyrinthe)