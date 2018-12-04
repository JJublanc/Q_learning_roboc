from tkinter import *
import random
from time import sleep
from actions import *
import pandas as pd
import numpy as np


def exploiting_robot(q=pd.DataFrame(columns=["s", "a", "q"]),
                  labyrinthe=chargement_des_cartes()[0].labyrinthe,
                  epsilon=1):
    """
    Initialisation des paramètres
    """
    try:
        Q = q
        assert isinstance(Q, pd.DataFrame)
        assert [q.columns] == ["s", "a", "q"]
    except AssertionError:
        print("Q doit être un dataFrame avec trois colonnes nommées s, a et q")
        Q = pd.DataFrame(columns=["s", "a", "q"])

    # on place le robot au hasard dans un emplacement vide
    placement_robot = random.choice(labyrinthe.places_libres)
    labyrinthe.changer_position(placement_robot[0], placement_robot[1])

    # on initialise la fin de partie à False
    fin_partie = False

    # on initialise le nombre d'essais
    nb_essais = 1

    # on initialise la sortie graphique
    fenetre = Tk()
    var = StringVar()
    var.set(labyrinthe.grille)
    labyrinthe_sortie = Label(fenetre, textvariable=var, anchor="e", width=20)
    labyrinthe_sortie.pack()
    s = '\n'.join(["".join(line) for line in labyrinthe.grille])

    # on joue
    while not fin_partie:
        # Etat de l'environnement : ici c'est le labyrinthe (position des obstacles, portes, sorties et robot)

        rand = np.random.choice((0, 1), p=[epsilon, (1 - epsilon)])

        # si l'état est connu et qu'on est dans une configuration greedy on choisit
        # l'action maximisant l'espérance de gain sinon on effectue un mouvement au hasard
        if (s in list(Q.s)) & (rand == 1):
            if max(Q[Q.s == s].q) > 0:
                max_q = max(Q[Q.s == s].q)
                print(max_q)
                a = random.choice(Q[(Q.s == s) & (Q.q == max_q)])
            else:
                a = random.choice(['n', 's', 'e', 'o'])
        else:
            a = random.choice(['n', 's', 'e', 'o'])
        # TODO transformer le choix de l'action en fonction

        # on met à jour l'état du labyrinthe
        s = '\n'.join(["".join(line) for line in labyrinthe.grille])

        # on effectue l'action choisie, ici on bouge (on tente de bouger) le robot
        fin_partie = labyrinthe.executer_instruction(a)
        var.set(s)
        fenetre.update()
        sleep(0.1)

        # TODO idem transformer en fonction

        if fin_partie:
            var.set("Félicitations ! Vous avez mis {} coups pour réussir".format(nb_essais))
        else:
            nb_essais += 1

exploiting_robot()

labyrinthe=chargement_des_cartes()[0].labyrinthe
