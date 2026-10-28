"""Le rapport d'un run LBL: ce qu'il a fait, en figures.

Ecrit a la toute fin, apres `compile`, un PDF par run dans <DATA_DIR>/reports.
Pilote par le parametre REPORT du wrapper, vrai par defaut.

Ce que le rapport montre:

    la SEQUENCE des vitesses, par instrument, avec ses barres d'erreur
    le PERIODOGRAMME de la serie et celui de la fenetre
    le FIP resistant aux aberrantes, dans ses trois versions (sans coupure,
        coupure douce, coupure dure), parce que ce qui compte est de voir ce
        que le nettoyage fait a la reponse et non la seule reponse nettoyee
    la QUALITE DE LA CORRECTION TELLURIQUE, si elle a ete appliquee: pour la
        meilleure, la mediane et la pire pose en colonne d'eau, puis la meme
        chose en colonne seche

LES ROUTINES VIENNENT DE KOLOA (periodogrammes, FIP, coupures robustes)
plutot que d'etre reecrites ici: elles y sont testees et maintenues. Le
rapport est donc optionnel au sens fort, c'est-a-dire qu'il se saute tout
seul et sans bruit si koloa n'est pas installe, et le run n'en souffre pas.
"""
from lbl.report.main import make_report

__all__ = ['make_report']
