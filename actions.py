# -*- coding: utf-8 -*-

"""
Ce script contient les fonctions qui sont appelées lors du lancement et du déroulement du jeu
Ces fonctions exécutent les actions nécessaires au déroulement de la partie
"""

from carte import *
from labyrinthe import *

# Charger les cartes existantes
def chargement_des_cartes():
    cartes = []
    for nom_fichier in os.listdir("cartes"):
        if nom_fichier.endswith(".txt"):
            chemin = os.path.join("cartes", nom_fichier)
            nom_carte = nom_fichier[:-3].lower()
            with open(chemin, "r") as fichier:
                contenu = fichier.read()
                cartes.append(Carte(nom_carte, contenu))
    return cartes


# Afficher une liste de cartes
def afficher_cartes(cartes):
    print("Voici les labyrinthes existants :")
    for ii, carte in enumerate(cartes):
        print("  {} - {}".format(ii + 1, carte.nom))


# Afficher un labyrinthe
def afficher_labyrinthe(labyrinthe):
    print("\n" + '\n'.join(["".join(line) for line in labyrinthe.grille]) + "\n")


# Choisir une carte parmi une liste de carte
def choisir_une_carte(cartes):
    carte_choisie = False
    while not carte_choisie:
        afficher_cartes(cartes)
        try:
            numero_labyrinthe_choisi = int(input("Choisissez un labyrinthe (indiquer le numéro du labyrinthe) : "))
            carte = cartes[numero_labyrinthe_choisi - 1]
            carte_choisie = True
            assert numero_labyrinthe_choisi > 0
        except:
            print("\n \nEntrer un entier entre 1 et {}".format(len(cartes)))
            carte_choisie = False
    return carte

# Choisir de reprendre une partie en cours ou non
# Si on lance une partie nouvelle : choisir la carte pour une partie nouvelle
def choisir_une_partie(cartes):
    reprendre_partie_en_cours = ""
    """
    S'il y a une partie en cours on propose au joueur de la reprendre
    """
    if os.path.isfile("partie_en_cours"):
        while not (reprendre_partie_en_cours in ["oui", "o", "non", "n"]):
            reprendre_partie_en_cours = input("Il y a une partie en cours. Souhaitez-vous la reprendre ? ")
    """
    Si le joueur souhaite reprendre la partie en cours on charge le labyrinthe à partir du fichier "partie_en_cours
    sinon on lui propose de choisir l'une des cartes disponibles"
    """
    if reprendre_partie_en_cours.lower() in ["o", "oui"]:
        with open("partie_en_cours", 'rb') as partie_en_cours:
            partie_en_cours_depickler = pickle.Unpickler(partie_en_cours)
            labyrinthe = partie_en_cours_depickler.load()
    else:
        carte_choisie = choisir_une_carte(cartes)
        labyrinthe = carte_choisie.labyrinthe
    return labyrinthe

# Lancer une partie à partir du labyrinthe choisi
# la méthode executer_instruction() renvoie True seulement si la partie prend fin (le joueur a gagné ou décide d'arrêter)
def jouer(labyrinthe):
    arreter = False
    while not arreter:
        instruction = input("Donner votre instruction au robot pour le faire avancer ")
        arreter = labyrinthe.executer_instruction(instruction) # on demande une instruction au joueur et execute
                                                               # l'instruction donnée
        if not arreter:
            afficher_labyrinthe(labyrinthe)
