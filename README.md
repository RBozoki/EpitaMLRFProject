# EpitaMLRFProject

Un projet libre par équipe de 2 personnes est à effectuer pour cette session de cours. Le choix des équipes sera pré-défini et aléatoire.
Il se décompose en une petite soutenance orale le 28 Juin à 13h, et l'envoi d'un rapport écrit (incluant lien vers code git) jusqu'au 5 Juillet 23h59.

Nous vous conseillons de commencer le plus tôt possible.
Contenu

Il s'agit de classifier la base de données CIFAR-10, dont la version python est disponible ici.

Voici le plan attendu pour le projet:
1. Introduction et état de l'art

2. Méthode

        3 algo feature extraction: dont un sans extraction (vecteur "aplati" de l'image)
        3 classifieurs: Un paramétrique linéaire, deux non-paramétriques linéaire/non-linéaire (non-linéaire avec data preprocessing est ok)
        entraînement basé descente de gradient stochastique
        Doit inclure les définitions et clairement apparaître les hyper-paramètres vs paramètres d’entraînement    Quelle stratégie multi-classe

3. Expérimentations

        Hardware et softwares utilisés (avec version) et code sur git (plate-forme au choix)
        Analyse de la base de données avec détails techniques (taille image, )
        Pre-processing
        Choix des hyper paramètres (split train/valid/test, taille des mini-batchs, learning rate, kernels utilisés etc...)

4. Résultats

        quantitatif
            indépendance des features
            fonction de coût
            matrice de confusion
            ROC curve
            comparaison entre classifieurs
            comparaison entre algo d'extraction de features
        qualitatif
            visualization de l'espace latent
            frontiere de decision

5. Conclusion et améliorations

Il n'est pas nécessaire d'avoir finalisé le plan au complet pour la soutenance orale. Les parties 1, 2 et 3 sont attendues, avec quelques résultats pour 2 classifieurs et 2 algos de feature extraction.
L'entraide entre équipes est vivement recommandée.


    Tous les documents et internet autorisés (chatbot interdit pour écriture du rapport mais ok pour le reste).
    Programmation Python avec limitation des logiciels à utiliser (scikit-learn, opencv, numpy et matplotlib).
    Pas de deep learning.
    Pas de jupyter notebook pour la librairie, ok pour faire la visu (pour un fichier d'exemple)
    Pour l'architecture de votre projet git, inspirez-vous de ce layout: https://drivendata.github.io/cookiecutter-data-science/
