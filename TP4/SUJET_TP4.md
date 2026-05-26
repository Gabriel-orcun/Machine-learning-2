2.  Profils des districts engagés dans la transition électrique
    Problème métier

Identifier les combinaisons de caractéristiques fréquentes chez les districts déjà engagés dans les bus électriques (committed / operating / delivering).

Dataset recommandé
[
'3b. Number of delivered or operating ESBs',
'3f. Number of ESBs operating',
'4f. Median household income',
'4g. Percent poverty',
'4e. Free/reduced lunch %',
'1p. Locale broad type'
]
Transformation
committed = ESB_committed == 1
operating = ESB_operating > 0
high_income = income > median
low_income = income <= median
urban = locale == "Urban"
high_need = poverty > median
