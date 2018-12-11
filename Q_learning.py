import random
from actions import *
import pandas as pd
import numpy as np


def Q_learning_test(nb_iter=1, q=pd.DataFrame(columns=["s", "a", "q"])):
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

    lamb = 0.9  # facteur de discount
    alpha = 0.8  # taux d'apprentissage
    epsilon = 0.1  # paramètre epsilon pour la epsilon-greedy policy
    nombre_essais = []
  
    """
    Boucle pour plusieurs parties
    """
    for ii in range(nb_iter):
        cartes = chargement_des_cartes()
        labyrinthe = cartes[0].labyrinthe
        placement_robot = random.choice(labyrinthe.places_libres)
        labyrinthe.changer_position(placement_robot[0], placement_robot[1])
        fin_partie = False
        nbnb_essais = 0

        """
        Boucle pour une partie
        """
        while not fin_partie:
            # Etat de l'environnement : ici c'est le labyrinthe (position des obstacles, portes, sorties et robot)
            s = '\n'.join(["".join(line) for line in labyrinthe.grille])

            rand = np.random.choice((0, 1), p=[epsilon, (1 - epsilon)])
            # si l'état est connu et qu'on est dans une configuration greedy on choisi
            # l'action maximisant l'espérance de gain
            # sinon on effectue un mouvement au hasard
            if (s in list(Q.s)) & (rand == 1):
                if max(Q[Q.s == s].q) > 0:
                    max_q = max(Q[Q.s == s].q)
                    print(max_q)
                    a = Q[(Q.s == s) & (Q.q == max_q)].a[0]
                else:
                    a = random.choice(['n', 's', 'e', 'o'])
            else:
                a = random.choice(['n', 's', 'e', 'o'])

            # on effectue l'action choisie, ici on bouge (on tente de bouger) le robot
            afficher_labyrinthe(labyrinthe)
            fin_partie = labyrinthe.executer_instruction(a)
            afficher_labyrinthe(labyrinthe)

            if fin_partie:
                print("stop")

            # on récupère la récompense imédiate
            r = labyrinthe.nombre_points
            # on récupère le nouvel état de l'environnement
            s_prim = '\n'.join(["".join(line) for line in labyrinthe.grille])

            # On calcule la target qui est égale à :
            #  - la récompense immédiate
            #  - à laquelle s'ajoute la valeur q maximale attendue dans le nouvel état s_prim
            # Si on ne peut pas calculer l'espérance de gain on l'estime à 0
            if s_prim in list(Q.s):
                target = r + lamb * max(Q[Q.s == s_prim].q)
            else:
                target = r

            # On met à jour la valeur q pour le couple s,a (état-action)
            if sum(Q[(Q.s == s)].loc[:, 'a'].isin([a])) > 0:
                Q.loc[(Q.s == s) & (Q.a == a), 'q'] = (alpha * Q[(Q.s == s) & (Q.a == a)].q
                                                       + (1 - alpha) * target)
                print("mise à jour {}".format(Q.loc[(Q.s == s) & (Q.a == a), 'q']))
            else:
                df = pd.DataFrame([[s, a, target]], columns=["s", "a", "q"])
                print("ajout {}".format(df.s))
                Q = Q.append(df)
            nbnb_essais += 1
        nombre_essais.append(nbnb_essais)
    return nombre_essais, Q


resultats, Q_sortie = Q_learning_test(2)

# TODO empecher d'avoir des labyrinthe avec des lignes de tailles différentes ou des colonnes de tailles differentes
