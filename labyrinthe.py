# -*- coding: utf-8 -*-

"""Ce module contient la classe Labyrinthe."""
import pickle
import os


class Labyrinthe:
    """Classe représentant un labyrinthe."""

    def __init__(self, robot, obstacles, portes, sorties, grille, places_libres):
        self.elements_labyrinthe = ["X", "O", " ", "U", "."]
        self.directions_autorisees = ["n", "s", "e", "o"]
        self.commande_fin_partie = ["q"]
        self.robot = robot
        self.obstacles = obstacles
        self.portes = portes
        self.sorties = sorties
        self.places_libres = places_libres
        self.grille = grille
        self.message_erreur_instruction = "\n\nIl faut donner une instruction valide du type xy, avec x " \
                                          "une direction parmi {0} et y un entier indiquant le " \
                                          "nombre de pas à effectuer dans cette direction." \
                                          "\nVous pouvez également quitter le jeu en " \
                                          "saisissant {1}.\n\n" \
            .format(self.directions_autorisees, self.commande_fin_partie[0])
        self.message_obstacles = "Il y a des obstacles sur le chemin. Choisissez un autre mouvement."
        self.message_partie_gagnee = "\n" \
                                     "\n***************" \
                                     "\nFélicitation !" \
                                     "\nVous avez gagné" \
                                     "\n***************"
        self.message_fin_partie = "\nLa partie est finie. Vous pourrez la reprendre plus tard.\n"
        self.nombre_points = 0

    def executer_instruction(self, instruction):
        arreter = False

        """On lève une exception si l'instruction pour le mouvement n'est pas conforme."""
        try:
            direction = instruction[0].lower()
            """
            On lève une exception si la direction n'est pas dans la liste direction autorisées
            ET n'est pas dans la liste des instructions permettant de quitter la partie
            """
            assert (direction in self.directions_autorisees) | \
                   (direction in self.commande_fin_partie)

            """
            Si la direction est q on arrête la partie et on l'enregistre
            Sinon on tente d'effectuer le mouvement correspondant à l'instruction
            """
            if direction == "q":
                arreter = self.fin_de_partie()
            else:
                if len(instruction) == 1:
                    nombre_pas = 1
                else:
                    nombre_pas = int(instruction[1:len(instruction)])
                arreter = self.effectuer_mouvement(direction, nombre_pas)
        except ValueError:
            print(self.message_erreur_instruction)
        except AssertionError:
            print(self.message_erreur_instruction)

        """On renvoit True si la partie prend fin"""
        return arreter

    def effectuer_mouvement(self, direction, nombre_pas):
        robot_sorti = False

        """On enregistre les coordonnées que l'instruction cherche à atteindre."""
        if direction == "n":
            ligne_a_atteindre = max((self.robot[0] - nombre_pas), 0)
            colone_a_atteindre = self.robot[1]

        if direction == "s":
            ligne_a_atteindre = min((self.robot[0] + nombre_pas), len(self.grille)-1)
            colone_a_atteindre = self.robot[1]

        if direction == "e":
            ligne_a_atteindre = self.robot[0]
            colone_a_atteindre = min((self.robot[1] + nombre_pas), len(self.grille[self.robot[0]])-1)

        if direction == "o":
            ligne_a_atteindre = self.robot[0]
            colone_a_atteindre = max((self.robot[1] - nombre_pas), 0)

        """On enregistre deux informations : 
            1/ le parcours est-il dégagé (i.e pas d'obstacle) ? 
            2/ la position d'arrivée est-elle une sortie ?"""
        aucun_osbtacle_sur_parcours = (self.nombre_obstacles_sur_parcours(ligne_a_atteindre, colone_a_atteindre) == 0)
        position_arrivee_sortie = ((ligne_a_atteindre, colone_a_atteindre) in self.sorties)

        """S'il n'y a pas d'obstacle on effectue le mouvement"""
        if aucun_osbtacle_sur_parcours:
            self.changer_position(ligne_a_atteindre, colone_a_atteindre)
        else:
            print(self.message_obstacles)

        """Si n'y a pas d'obstacle et que la position finale est une sortie on finit la partie"""
        if aucun_osbtacle_sur_parcours & position_arrivee_sortie:
            robot_sorti = self.partie_gagnee()

        """la méthode renvoie un booleen qui vaut True si on arrête la partie ou si on est sorti du labyrinthe."""
        return robot_sorti

    def nombre_obstacles_sur_parcours(self, ligne_a_atteindre, colone_a_atteindre):
        """
        :param ligne_a_atteindre: numéro de la ligne où nous amène l'instruction
        :param colone_a_atteindre: numéro de la colonne où nous amène l'instruction
        :return: le nombre d'ostacles sur le parcours
        """
        x1 = min(ligne_a_atteindre, self.robot[0])
        x2 = max(ligne_a_atteindre, self.robot[0])
        y1 = min(colone_a_atteindre, self.robot[1])
        y2 = max(colone_a_atteindre, self.robot[1])

        """
        x1, x2 : sont les numéros des deux lignes les plus éloignées sur le parcours avec x1 <= x2
        y1, y2 : sont les numéros des deux colonnes les plus éloignées sur le parcours avec y1 <= y2
        NB : dans notre cas si x1<x2 alors y1=y2 et inversement (i.e il n'y a pas de déplacement en diagonale)
        """

        obstacle_x = [(ii, self.robot[1]) in self.obstacles for ii in range(x1, x2 + 1)]
        obstacle_y = [(self.robot[0], ii) in self.obstacles for ii in range(y1, y2 + 1)]

        return sum(obstacle_x) + sum(obstacle_y)

    def changer_position(self, ligne_a_atteindre, colone_a_atteindre):
        """On efface le robot de la grille sans oublier de remettre le
        symbole de la porte s'il y en avait une à notre arrivée"""
        if (self.robot[0], self.robot[1]) in self.portes:
            self.grille[self.robot[0]][self.robot[1]] = "."
        else:
            self.grille[self.robot[0]][self.robot[1]] = " "

        """On change la position du robot"""
        self.robot = [ligne_a_atteindre, colone_a_atteindre]

        """On positionne le robot sur la case à atteindre"""
        self.grille[self.robot[0]][self.robot[1]] = "X"

        """On enregistre la partie"""
        self.enregistrer_partie()

    def fin_de_partie(self):
        print(self.message_fin_partie)
        self.enregistrer_partie()
        return True

    def partie_gagnee(self):
        print(self.message_partie_gagnee)
        os.remove("partie_en_cours")
        self.nombre_points = 1
        return True

    def enregistrer_partie(self):
        with open("partie_en_cours", "wb") as partie_en_cours:
            partie_en_cours_pickler = pickle.Pickler(partie_en_cours)
            partie_en_cours_pickler.dump(self)
