# Système de scoring

> 🌐 [English](../SCORING.md) · **Français** · [Deutsch](../de/SCORING.md) · [Italiano](../it/SCORING.md) · [Español](../es/SCORING.md) · [Português](../pt/SCORING.md) · [简体中文](../zh/SCORING.md)

Les photos sont classées dans une catégorie, puis notées avec les poids de cette catégorie.

## Fonctionnement du scoring

1. **Détection de la catégorie** - La photo est analysée pour son contenu (visages, tags, données EXIF)
2. **Évaluation des filtres** - Les catégories sont évaluées par ordre de priorité jusqu'à ce qu'une corresponde (un *contexte de notation* peut promouvoir/exclure des catégories par album ou par photo sans modifier l'ordre de base — voir [Contextes de notation](#contextes-de-notation))
3. **Application des poids** - Les poids spécifiques à la catégorie sont appliqués aux métriques
4. **Application des modificateurs** - Bonus, pénalités et indicateurs de comportement appliqués
5. **Score final** - Somme pondérée bornée à la plage 0-10

## Catégories

`scoring_config.json` définit 34 catégories (33 nommées plus `default`), évaluées par ordre de priorité croissant jusqu'à ce qu'une corresponde. La priorité la plus basse l'emporte. La liste complète figure dans le tableau `categories` ; voici les principales :

| Priorité | Catégorie | Méthode de détection |
|----------|----------|------------------|
| 8 | `art` | Tags : painting, statue, drawing, cartoon, anime |
| 10 | `astro` | Étiquettes : aurora, astrophotography, stars, milky way, ET luminance moyenne ≤ 0.30 |
| 15 | `concert` | Tags : concert |
| 35 | `group_portrait` | Ratio visage ≥ 5 % ET is_group_portrait |
| 42 | `silhouette` | A un visage ET `is_silhouette` (le plus grand visage est sombre sur un environnement plus clair) |
| 45 | `portrait` | Ratio visage ≥ 5 %, ni silhouette/groupe/mono |
| 46 | `portrait_bw` | Portrait monochrome (visage ≥ 5 %) |
| 55 | `macro` | Tags : macro, insect, butterfly, dewdrop, ... |
| 65 | `wildlife` | Tags : animal, bird, marine, reptile, primate |
| 80 | `long_exposure` | Obturation 1-10 secondes |
| 85 | `night` | Luminance < 0,15 |
| 88 | `monochrome` | is_monochrome (saturation < 5 %) |
| 95 | `street` | Tags : street, urban_culture |
| 96 | `human_others` | Présence d'un visage ET ratio visage < 5 % |
| 100 | `landscape` | Tags : landscape, mountain, beach, forest, ... |
| 999 | `default` | Repli (aucun filtre) |

Les autres catégories basées sur les tags incluent `aerial`, `food`, `sports`, `vehicle`, `travel`, `fashion`, `candid`, `product`, `architecture`, `urban`, `golden_hour`, `blue_hour`, `cinematic`, `vintage`, `abstract`, `minimalist`, `dramatic` et `weather`.

Sur les profils à étiquetage CLIP (`legacy`/`8gb`, et `16gb`/`24gb` quand leur tagger VLM ne s'exécute pas), un scan multi-passes étiquette chaque photo pendant la passe d'embeddings, avant le calcul de sa catégorie et de son agrégat. La catégorie stockée par un scan utilise donc les étiquettes, comme `--recompute-average`. L'étape d'étiquetage en fin de scan ne rattrape que les photos encore sans étiquette. Une exécution avec un seul `--pass NAME` n'étiquette pas.

**Silhouette.** `is_silhouette` provient d'une règle de pixels fondée sur le visage (voir [Filtres booléens](#filtres-booléens)), pas d'une étiquette ni d'un test d'histogramme. Le drapeau stocké agit sur le score autant que sur la catégorie. Quand `exposure.silhouette_detection` vaut `true` (valeur par défaut), l'agrégat d'une silhouette ignore la pénalité d'écrêtage. Le `exposure_score` stocké dépend de la commande : un scan le stocke avec la pénalité d'écrêtage, tandis que `--recompute-average` le recalcule sans elle pour toute photo marquée `is_silhouette`, quelle que soit la valeur de `exposure.silhouette_detection`. Pour recalculer les drapeaux d'une bibliothèque existante, lancez `python facet.py --recompute-silhouette`, puis `--recompute-average` deux fois (voir [COMMANDS.md](COMMANDS.md)).

**Astro.** Une photo doit avoir une étiquette astro *et* une luminance moyenne d'au plus 0.30 (`luminance_max`) : une scène lumineuse étiquetée `aurora` ou `stars` n'est pas astro, et une photo sans luminance stockée ne correspond jamais. Un `scoring_config.json` avec son propre tableau `categories` garde sa propre entrée `astro` et ne reçoit pas ce filtre tant que vous n'ajoutez pas `"luminance_max": 0.30` à `astro.filters` ; `python facet.py --doctor` vous prévient quand la vôtre en manque.

## Contextes de notation

L'ordre de priorité ci-dessus est global : chaque photo est évaluée par rapport à la même liste. Un **contexte de notation** est un *delta* nommé appliqué à cet ordre de base : il promeut une courte liste de catégories en tête et en exclut d'autres purement et simplement, sans renuméroter quoi que ce soit. `default` (avec `promote`/`excluded` vides) est le contexte neutre, si bien que rien ne change pour une photo tant qu'un contexte ne lui est pas explicitement attribué.

**Ordre effectif** = `promote` (dans l'ordre donné) → l'ordre de priorité global, moins les noms promus et exclus → `default` en dernier. Un nom présent à la fois dans `promote` et dans `excluded` est entièrement retiré — c'est `excluded` qui l'emporte. `ScoringConfig.resolve_context_order()` (`config/scoring_config.py`) calcule et met ce résultat en cache une fois par nom de contexte.

Préréglages livrés par défaut — modifiables depuis l'onglet **Contexte de notation** de la visionneuse (`PUT /api/config/scoring_contexts/{name}`, réservé au mode édition) ou directement dans le JSON ; voir [Contextes de notation](CONFIGURATION.md#contextes-de-notation) pour la référence complète des champs :

| Contexte | Promeut | Exclut |
|---------|----------|----------|
| `default` | — | — |
| `action_stage` | `sports`, `concert`, `candid` | `silhouette` |
| `party_event` | `group_portrait`, `candid`, `food` | — |
| `portrait_session` | `portrait`, `portrait_bw`, `fashion` | — |
| `wildlife` | `wildlife` | — |
| `landscape` | `landscape`, `golden_hour`, `blue_hour` | — |
| `motorsport` | `sports`, `vehicle` | `silhouette` |

Seul le *delta* est modifiable — faites glisser la tête promue dans l'ordre voulu (ou utilisez les boutons Monter/Descendre), basculez l'exclusion d'une catégorie — jamais un ordre complet autonome par contexte : les catégories non promues conservent toujours l'ordre de priorité global, si bien qu'une catégorie ajoutée plus tard ne peut jamais manquer silencieusement dans six listes distinctes. Voir [Modifier un contexte](CONFIGURATION.md#modifier-un-contexte) pour les règles de validation.

Un contexte s'attribue par album (`PUT /api/albums/{id}/scoring_context`, qui le matérialise sur chaque photo membre à cet instant — un instantané, pas un abonnement, pour un album intelligent, voir [Contextes de notation](CONFIGURATION.md#contextes-de-notation)) ou, pour une photo isolée récalcitrante, s'applique comme un remplacement de catégorie persistant (`POST /api/comparison/override_category`). Les deux leviers sont conservés dans une table annexe `photo_scoring_overrides` plutôt que comme des colonnes de `photos` — `save_photo`/`save_photos_batch` écrivent les lignes de photo avec `INSERT OR REPLACE`, ce qui effacerait silencieusement une nouvelle colonne de cette ligne au prochain rescan. Activer un levier laisse l'autre intact, et chacun peut être réinitialisé indépendamment. **Aucun des deux ne prend effet sur des photos déjà notées avant un recalcul** — `python facet.py --recompute-average`, ou `POST /api/scan/recompute` depuis la visionneuse (protégé au niveau inter-processus contre l'exécution simultanée de deux instances — voir [Modifier les priorités nécessite un recalcul](CONFIGURATION.md#réorganiser-la-priorité-globale)). Si `normalization.per_category` est activé, relancez le recalcul deux fois — voir [Normalisation](CONFIGURATION.md#normalisation) pour comprendre pourquoi la première passe normalise par rapport à l'ancienne catégorie de chaque photo.

### Le piège des données EXIF manquantes

Réorganiser — que ce soit en modifiant la priorité globale ou en promouvant via un contexte — ne change que la catégorie *essayée en premier*. Cela ne peut pas faire correspondre les filtres d'une catégorie à une photo qu'ils ne matcheraient pas autrement. `config/category_filter.py:122-128` fait échouer purement et simplement un filtre de plage numérique dès que la valeur sous-jacente de la photo est manquante ou ininterprétable, plutôt que de simplement ignorer cette borne — une valeur manquante et une valeur hors plage sont traitées de façon identique, et la catégorie est écartée dans les deux cas.

Concrètement : `sports` (priorité 71) porte `shutter_speed_max: 0.02`. Une photo de danse prise à une vitesse plus lente que 1/50s, ou sans vitesse d'obturation EXIF lisible du tout, échoue à ce filtre quel que soit l'endroit où `sports` se trouve dans l'ordre d'évaluation — même promue tout en tête par un contexte comme `action_stage`. La photo retombe alors sur la première catégorie suivante qui correspond, typiquement `fashion` (priorité 43, taguée `fashion`, contient un visage) ou `silhouette` (priorité 42, contre-jour avec un visage). **C'est le tout premier réflexe à avoir lorsqu'une photo atterrit dans une catégorie inattendue :** avant de réorganiser ou de promouvoir quoi que ce soit, vérifiez que les filtres numériques de la catégorie visée peuvent réellement correspondre aux données EXIF stockées de la photo, pas seulement à ses tags.

### Le piège du tag manquant

Le piège EXIF ci-dessus suppose que la couche de tags a déjà trouvé une correspondance — `required_tags` a sa propre version du même problème : un filtre ne peut pas correspondre à un tag qui n'existe pas du tout dans le vocabulaire CLIP. Avant le commit `917dd94`, rien dans `scoring_config.json` ne couvrait la danse : une image que le tagger décrivait comme danse, performance ou scène ne correspondait à aucun des `required_tags` de `sports` (`sports`, `motion`, `athlete`, `competition`, `action_sport`) et retombait directement dans le repli `default` — promouvoir `sports` via `action_stage` ne changeait rien, car la promotion ne fait que réordonner *quand* une catégorie est essayée, jamais si ses filtres correspondent.

`sports.tags.dance` porte désormais 11 prompts CLIP (`dance performance`, `dancer on stage`, `ballet dancer`, `contemporary dance`, `ballroom dancing`, `latin dance`, `hip hop dance`, `dance competition`, `dance troupe performing`, `dancer mid leap`, `dancer in motion`), et `dance` a été ajouté à `sports.filters.required_tags`, si bien qu'une photo de danse taguée peut désormais franchir ce filtre. Cela rend `sports` *accessible* pour le contenu de danse — cela ne dispense pas du piège de la vitesse d'obturation ci-dessus : une image de danse à vitesse lente ou sans EXIF peut désormais passer la vérification des tags et échouer quand même à `shutter_speed_max: 0.02`, retombant sur `fashion` ou `silhouette` exactement comme décrit plus haut.

**Les photos existantes conservent les tags qu'elles ont déjà** — le nouveau vocabulaire ne change que ce que le tagger attribue à partir de maintenant. Ré-étiquetez la bibliothèque avec `python facet.py --recompute-tags` pour l'appliquer rétroactivement.

### Réorganiser la priorité globale

`GET/POST /api/config/category_priorities` (réservé au mode édition) lit et réécrit l'ordre de base sur lequel chaque contexte applique son delta. `POST` prend `{"order": [name, ...]}` — une permutation à ensemble égal de tous les noms de catégorie hors `default` — et **permute les valeurs de priorité existantes selon le nouvel ordre** plutôt que de les renuméroter (10/20/30/…) : le multi-ensemble des priorités reste inchangé, si bien que les nombres du tableau ci-dessus restent pertinents et que l'unicité est garantie par construction. `default` (priorité 999) reste fixé en dernière position et n'est pas concerné par la réorganisation. Chaque écriture prend d'abord une copie horodatée `.backup.<timestamp>` de `scoring_config.json` ; cet écrivain et l'éditeur de poids (`update_category_weights`) partagent désormais un même verrou, car la précédente lecture-modification-écriture non protégée permettait à un enregistrement concurrent de l'un d'écraser silencieusement les modifications de l'autre.

Réorganiser ne touche par lui-même la `category` stockée d'aucune photo — lancez un recalcul ensuite (`--recompute-average` ou `POST /api/scan/recompute`) pour l'appliquer.

**Limitation connue :** `api/types.py` construit la liste déroulante de type/filtre de la galerie à partir de `ScoringConfig.get_categories()` une seule fois, à l'import. Une réorganisation de priorité est prise en compte immédiatement pour la correspondance de catégorie proprement dite (chaque appel de notation et de recalcul relit la configuration depuis le disque), mais le menu déroulant de type de la galerie conserve son ancien ordre jusqu'au redémarrage du processus de la visionneuse. Le filtrage lui-même n'est pas affecté — la réorganisation n'ajoute ni ne retire aucun nom de catégorie.

## Définition d'une catégorie

Chaque catégorie dans `scoring_config.json` comporte ces composants :

```json
{
  "name": "portrait",
  "priority": 45,
  "filters": {
    "face_ratio_min": 0.05,
    "has_face": true,
    "is_silhouette": false,
    "is_group_portrait": false,
    "is_monochrome": false
  },
  "weights": {
    "aesthetic_percent": 32,
    "eye_sharpness_percent": 16,
    "face_quality_percent": 14,
    "composition_percent": 12,
    "liqe_percent": 8,
    "exposure_percent": 4,
    "tech_sharpness_percent": 4,
    "color_percent": 4,
    "contrast_percent": 4,
    "aesthetic_iaa_percent": 2
  },
  "modifiers": {
    "bonus": 0.419,
    "_apply_blink_penalty": true,
    "noise_tolerance_multiplier": 0.006,
    "_clipping_multiplier": 0.5
  },
  "tags": {}
}
```

## Référence des filtres

### Filtres de plage numérique

| Filtre | Champ | Description |
|--------|-------|-------------|
| `face_ratio_min` / `face_ratio_max` | `face_ratio` | Surface du visage en fraction (0.0-1.0) |
| `face_count_min` / `face_count_max` | `face_count` | Nombre de visages |
| `iso_min` / `iso_max` | `ISO` | ISO de l'appareil |
| `shutter_speed_min` / `shutter_speed_max` | `shutter_speed` | Temps d'exposition (secondes) |
| `luminance_min` / `luminance_max` | `mean_luminance` | Luminosité (0.0-1.0) |
| `focal_length_min` / `focal_length_max` | `focal_length` | Longueur focale (mm) |
| `f_stop_min` / `f_stop_max` | `f_stop` | Ouverture (nombre f) |

### Filtres booléens

| Filtre | Description |
|--------|-------------|
| `has_face` | Au moins un visage détecté |
| `is_monochrome` | Saturation < 5 % |
| `is_silhouette` | Le plus grand visage est sombre et son environnement clair : la luminosité moyenne (luma, 0-1) des 60 % centraux du cadre du visage est inférieure à 0.12, et la bande autour du cadre (de 0.2 à 0.75 de la taille du cadre vers l'extérieur de chaque côté, rognée à l'image) est plus claire de plus de 0.20. Ni l'étiquette CLIP `silhouette` ni l'histogramme de l'image entière ne le définissent plus. Une photo sans visage n'est jamais une silhouette |
| `is_group_portrait` | face_count >= `min_faces_for_group` (configurable, défaut : 4) |

### Filtres de tags

| Filtre | Description |
|--------|-------------|
| `required_tags` | Liste des tags que la photo doit posséder |
| `excluded_tags` | Liste des tags que la photo ne doit PAS posséder |
| `tag_match_mode` | `"any"` (défaut) ou `"all"` |

## Clés de poids

Tous les poids utilisent le suffixe `_percent`. Ils sont normalisés par `get_weights()`, donc les totaux n'ont pas besoin de valoir exactement 100 — mais les maintenir à 100 conserve les scores sur l'échelle 0-10.

| Clé | Métrique | Source | Idéale pour |
|-----|--------|--------|----------|
| `aesthetic_percent` | Attrait visuel | TOPIQ ou CLIP+MLP | Toutes |
| `quality_percent` | Qualité héritée | Ajouté à `aesthetic_percent` avant le calcul (pas de signal distinct) | — |
| `face_quality_percent` | Netteté du visage | InsightFace | Portraits |
| `eye_sharpness_percent` | Netteté des yeux | Points de repère InsightFace | Portraits |
| `tech_sharpness_percent` | Netteté globale | Variance laplacienne | Paysages |
| `composition_percent` | Composition | SAMP-Net ou règles | Toutes |
| `exposure_percent` | Équilibre de l'exposition | Analyse d'histogramme | Toutes |
| `color_percent` | Harmonie des couleurs | Analyse HSV | Photos couleur |
| `contrast_percent` | Contraste tonal | Étendue de l'histogramme | N&B |
| `dynamic_range_percent` | Plage tonale | Analyse d'histogramme | HDR, paysages |
| `isolation_percent` | Séparation du sujet | Visage vs arrière-plan | Portraits, faune |
| `leading_lines_percent` | Lignes directrices | Détection de contours | Architecture |
| `power_point_percent` | Règle des tiers | Placement du sujet | Toutes |
| `saturation_percent` | Saturation des couleurs | Analyse HSV | Photos vives |
| `noise_percent` | Niveau de bruit | Estimation du bruit | Basse lumière |
| `face_sharpness_percent` | Netteté de la zone du visage | Analyse du visage | Portraits |
| `aesthetic_iaa_percent` | Mérite esthétique artistique | TOPIQ IAA (entraîné sur AVA) | Art, créatif |
| `face_quality_iqa_percent` | Qualité du visage (IQA) | TOPIQ NR-Face | Portraits |
| `liqe_percent` | Score de qualité LIQE | LIQE | Diagnostics |
| `subject_sharpness_percent` | Netteté de la zone du sujet | BiRefNet + Laplacien | Portraits, faune |
| `subject_prominence_percent` | Ratio de surface du sujet | BiRefNet | Macro, faune |
| `subject_placement_percent` | Règle des tiers du sujet | BiRefNet | Toutes |
| `bg_separation_percent` | Séparation de l'arrière-plan | BiRefNet | Portraits, macro |

**`quality_percent` compte dans `aesthetic`.** La métrique de qualité n'a pas de score propre : le `quality_percent` d'une catégorie est donc ajouté à son `aesthetic_percent` avant le calcul. Les poids totalisent toujours 100. Les versions précédentes notaient la part `quality` avec une métrique toujours égale à 0, si bien que cette part était perdue dans l'agrégat. Au prochain `--recompute-average`, une bibliothèque agrégée par une version précédente retrouve cette part. Sur une copie d'une bibliothèque de 129 737 photos, 35 139 agrégats ont augmenté, 544 ont baissé (moyenne +0,087), et aucune catégorie n'a changé.

## Modificateurs

Ajustent le comportement de scoring par catégorie :

| Modificateur | Type | Description |
|----------|------|-------------|
| `bonus` | float | Ajouté au score final (ex. 0.5) |
| `noise_tolerance_multiplier` | float | Échelle de la pénalité de bruit (0.5 = moitié) |
| `_skip_clipping_penalty` | bool | Ignore la pénalité d'écrêtage de l'exposition |
| `_skip_oversaturation_penalty` | bool | Ignore la pénalité de sursaturation |
| `_clipping_multiplier` | float | Échelle de la pénalité d'écrêtage |
| `_apply_blink_penalty` | bool | Applique la pénalité de détection de clignement |

## Dimensions de la saillance du sujet

Quatre dimensions dérivées de la segmentation du sujet par BiRefNet :

| Clé de poids | Métrique | Description |
|-----------|--------|-------------|
| `subject_sharpness_percent` | Netteté du sujet | Qualité de mise au point de la zone du sujet par rapport à l'arrière-plan. Élevée = sujet net, arrière-plan flou. |
| `subject_prominence_percent` | Proéminence du sujet | Surface du sujet en fraction du cadre. Élevée pour la macro et les sujets cadrés serré, faible pour les scènes larges. |
| `subject_placement_percent` | Placement du sujet | Score de règle des tiers pour le centre de masse du sujet. |
| `bg_separation_percent` | Séparation de l'arrière-plan | Différence de gradient de contour à la frontière du sujet (qualité du bokeh). |

Utilisez `subject_sharpness_percent` et `bg_separation_percent` pour le portrait/la faune ; `subject_prominence_percent` pour la macro.

**Pas de sujet, pas de nombre.** Quand aucun sujet n'est trouvé, `subject_sharpness`, `subject_placement` et `bg_separation` sont stockés NULL (tout comme `power_point_score` quand il n'y a pas de cadre de sujet), `subject_prominence` vaut 0.0 et `subject_bbox` est NULL ; les versions antérieures stockaient un 5.0 neutre. Les filtres de plage min/max de la galerie ignorent un NULL sur ces métriques (`min_subject_sharpness=0` ne renvoie plus une photo sans sujet), les statistiques l'excluent des moyennes et l'interface l'affiche comme absent. L'agrégat compte toujours un NULL comme le 5.0 neutre : aucun agrégat ne change. Les lignes existantes gardent leur 5.0 jusqu'à ce que `--recompute-saliency` et `--recompute-composition-cpu` les réécrivent.

**Quand la saillance échoue.** Sur CUDA, la taille de lot de saillance est déduite de la VRAM libre (environ 2,4 GiB par image à 1024 px, entre 1 et 8 images) et divisée par deux après une erreur de mémoire insuffisante, jusqu'à une seule image. Une photo dont la saillance échoue malgré tout journalise une ERROR et ne stocke rien : une photo jamais notée garde NULL dans les cinq colonnes de saillance (`subject_sharpness`, `subject_prominence`, `subject_placement`, `bg_separation`, `subject_bbox`), et `--recompute-saliency` et `--pass saliency` conservent les valeurs stockées d'une photo quand sa saillance échoue. Un nouveau scan complet de la photo (par exemple avec `--force`) stocke NULL pour elle, comme pour une photo jamais notée. Facet n'écrit jamais un nombre neutre à la place d'un résultat en échec. Dans l'agrégat, un NULL compte comme neutre (5,0), exactement comme une photo scannée avant l'existence de la saillance : il n'aide ni ne pénalise le score. `--recompute-saliency` resélectionne une telle photo (voir [Saillance du sujet](COMMANDS.md#saillance-du-sujet)).

## Dimensions IQA supplémentaires

Trois modèles de qualité additionnels :

| Clé de poids | Modèle | Description |
|-----------|-------|-------------|
| `aesthetic_iaa_percent` | TOPIQ IAA | Mérite esthétique entraîné sur AVA, distinct du score esthétique de qualité technique. Idéal pour les catégories art/créatif. |
| `face_quality_iqa_percent` | TOPIQ NR-Face | Évaluation de la qualité de la zone du visage. Idéal pour les catégories portrait. |
| `liqe_percent` | LIQE | Score de qualité accompagné d'un diagnostic de distorsion (flou de bougé, surexposition, bruit). |

Ces modèles s'exécutent dans le cadre du pipeline de scoring par défaut et partagent la VRAM avec TOPIQ. Ajoutez leurs clés de poids à toute catégorie où l'évaluation est utile.

**Quand un modèle supplémentaire ne peut pas noter une photo.** TOPIQ NR-Face a besoin d'un visage que son propre détecteur trouve, et LIQE d'une photo dont le petit côté mesure au moins 224 px ; Aesthetic V2.5, s'il est activé, peut échouer sur n'importe quelle image. Une photo qu'un modèle ne peut pas noter stocke NULL dans la colonne de ce modèle (`face_quality_iqa`, `liqe_score`, `aesthetic_iaa`, `aesthetic_v25`) au lieu d'un 5,0 neutre. Une erreur de mémoire insuffisante du GPU est retentée avec la moitié de la taille de lot, jusqu'à une seule image, avant que la photo ne soit abandonnée ; une erreur de mémoire du CPU n'est pas retentée ainsi, elle suit le chemin ordinaire image par image, et une photo qui échoue encore stocke NULL. Un échec inattendu est journalisé en ERROR avec le nombre de photos concernées, tandis que les deux cas attendus (aucun visage trouvé, petit côté inférieur à 224 px) ne sont totalisés dans le journal qu'à la libération du modèle. `--recompute-iqa` ne retente à chaque exécution aucun de ces deux cas attendus : le cas du petit côté se décide d'après la taille de la miniature stockée, et un refus de NR-Face est mémorisé jusqu'au nouveau scan de la photo. Un échec de mémoire insuffisante ou inattendu est retenté à l'exécution suivante. Dans l'agrégat, un NULL compte comme neutre (5,0) : les scores agrégés, les catégories et les classements ne bougent donc pas. **Un filtre de plage de la galerie sur Esthétique (IAA), Qualité du visage (IQA), LIQE ou Aesthetic V2.5 ne retient plus une photo sans score** : une photo sans visage portait auparavant un 5,0 stocké et correspondait à `min_face_quality_iqa=0`, ce qui n'est plus le cas ; un tri croissant sur Esthétique (IAA), Qualité du visage (IQA) ou LIQE place ces photos en dernier, dans la galerie comme dans les albums (intelligents et partagés compris). Les valeurs stockées par les versions précédentes restent telles quelles tant que `--repair-neutral-iqa` n'a pas effacé le 5,0 neutre de `aesthetic_iaa`, `face_quality_iqa` et `liqe_score` (lancez ensuite `--recompute-iqa` puis `--recompute-average`). Un nouveau scan qui réécrit la photo (par exemple avec `--force`) stocke NULL pour un score que le modèle n'a pas pu produire ; `--recompute-iqa` ne remplit que les colonnes NULL, et `--pass quality-iaa`, `--pass quality-face` et `--pass quality-liqe` conservent la valeur stockée quand un modèle échoue, donc aucun ne remplace un ancien 5,0. Si le modèle de qualité principal (TOPIQ sur les profils `16gb` et `24gb`) échoue pour une photo, celle-ci est enregistrée comme échouée dans `scan_failures`, au lieu d'être enregistrée avec un score esthétique neutre de 5,0. Le scan suivant reprend une photo qui n'est pas encore dans la base ; une photo déjà enregistrée (par exemple après un nouveau scan avec `--force`) garde sa ligne et est retentée par `--retry-failed all` (un `--retry-failed` sans argument ne couvre que le dernier scan ayant enregistré des échecs) ou par un nouveau scan avec `--force`. Avec `--pass quality`, la valeur stockée est conservée et aucune ligne d'échec n'est enregistrée. Voir [Modèles de qualité supplémentaires](COMMANDS.md#modèles-de-qualité-supplémentaires).

### Signaux supplémentaires (hors agrégat par défaut)

| Colonne | Source | Description |
|--------|--------|-------------|
| `aesthetic_clip` | `analyzers/aesthetic_clip.py` + embedding CLIP/SigLIP en cache | Un score esthétique supplémentaire gratuit (0-10) dérivé des embeddings d'image en cache par projection sur un « axe esthétique » construit à partir de prompts texte positifs/négatifs. Aucune inférence d'image supplémentaire au moment du scan. **Ne fait pas** partie de l'`aggregate` par défaut. À renseigner avec `python scripts/compute_aesthetic_clip.py --db <path>`. À évaluer avec `python scripts/benchmark_aesthetic.py --db <path> --ava AVA.txt --photo-dir <dir>`. SRCC AVA ≈ 0,52 sur le jeu de 500 photos `ava_test/` (contre 0,94 pour `aesthetic_iaa`) — utile comme pré-filtre bon marché ou lorsque TOPIQ-IAA n'est pas disponible. |

## Tags de catégorie (vocabulaire CLIP)

Les tags déclenchent les catégories basées sur les tags et sont mis en correspondance via la similarité CLIP :

```json
{
  "tags": {
    "landscape": ["landscape", "scenic view", "nature scene"],
    "mountain": ["mountain", "alpine", "peaks"],
    "beach": ["beach", "ocean", "seaside", "coastal"]
  }
}
```

Chaque clé est le nom canonique du tag, et le tableau contient les synonymes pour la correspondance CLIP.

## Scoring des Meilleurs choix

Le filtre « Meilleurs choix » de la galerie utilise un score pondéré personnalisé :

```json
"top_picks_weights": {
  "aggregate_percent": 30,
  "aesthetic_percent": 28,
  "composition_percent": 18,
  "face_quality_percent": 24
}
```

**Calcul du score :**
- Avec visage (ratio visage ≥ 20 %) : les quatre métriques contribuent
- Sans visage : `face_quality_percent` redistribué vers `aesthetic` et `composition`

## Considérations selon le profil VRAM

Les poids par défaut sont optimisés pour **TOPIQ** (SRCC 0,93), le modèle esthétique de tous les profils.

| Profil | Modèle esthétique | Embeddings | Tagger | Recommandations |
|---------|-----------------|-----------|--------|-----------------|
| `24gb` | TOPIQ (SRCC 0,93) | SigLIP 2 NaFlex SO400M | Qwen3.5-4B | Meilleure précision, poids par défaut |
| `16gb` | TOPIQ (SRCC 0,93) | SigLIP 2 NaFlex SO400M | Qwen3.5-2B | Poids par défaut |
| `8gb` | CLIP+MLP (SRCC 0,76) | CLIP ViT-L-14 | Similarité CLIP | Les poids par défaut fonctionnent bien |
| `legacy` | CLIP+MLP sur CPU | CLIP ViT-L-14 | Similarité CLIP | Poids par défaut, plus lent |

Tous les profils exécutent en plus les modèles PyIQA supplémentaires (TOPIQ IAA, TOPIQ NR-Face, LIQE) et, en option, BiRefNet_dynamic pour la saillance du sujet.

Exécutez `--compute-recommendations` après avoir changé de profil pour analyser les distributions de scores.

## Flux de réglage des poids

### Option A : via la galerie (recommandée)

1. Ouvrez `/stats` → onglet **Catégories** → sous-onglet **Poids**
2. Déverrouillez le mode édition
3. Sélectionnez une catégorie dans le menu déroulant de l'éditeur
4. Ajustez les curseurs — l'**aperçu de la distribution des scores** en direct montre l'impact estimé
5. Cliquez sur **Enregistrer** puis **Recalculer les scores** pour appliquer

La galerie exécute `--recompute-category` en coulisse, ne mettant à jour que les photos de cette catégorie.

### Option B : via la CLI

#### 1. Analyser les scores actuels

```bash
python facet.py --compute-recommendations
```

Affiche :
- Les distributions de scores par catégorie
- L'analyse de corrélation des poids
- Les ajustements suggérés

#### 2. Ajuster les poids

Modifiez les poids de catégorie dans `scoring_config.json`. Assurez-vous qu'ils totalisent 100.

#### 3. Recalculer les scores

```bash
python facet.py --recompute-average               # Toutes les catégories
python facet.py --recompute-category portrait      # Une seule catégorie (plus rapide)
```

Utilise les embeddings stockés — aucun GPU nécessaire.

#### 4. Valider les changements

```bash
python facet.py --compute-recommendations
```

Comparez les distributions avant/après.

## Mode de comparaison par paires

Entraînez les poids en comparant des paires de photos :

### Mise en place

1. Définissez un `edition_password` non vide dans la config : `"viewer": { "edition_password": "your-password" }`
2. Démarrez la galerie : `python viewer.py`
3. Cliquez sur le bouton « Comparer »

### Interface de comparaison

- Photos côte à côte
- Clavier : A (gauche gagne), B (droite gagne), T (égalité), S (passer)
- La barre de progression montre les comparaisons par rapport au minimum de 50

### Sources de comparaison

Les comparaisons portent un marqueur `source` afin que l'optimiseur puisse les pondérer selon leur fiabilité :

- `vote` — votes A/B explicites depuis l'interface de comparaison
- `culling` — dérivés automatiquement des décisions de tri de rafales/photos similaires : chaque
  photo rejetée est appariée à un maximum de deux photos conservées du même groupe
  (plafonné à 12 paires par groupe). Les photos conservées gagnent. Les votes explicites sur la même
  paire ne sont jamais écrasés.
- `rating` — paires synthétiques générées à partir des notes en étoiles et des favoris

Examiner les groupes de rafales dans la galerie enrichit donc le jeu d'entraînement pour
l'optimisation des poids sans aucun effort supplémentaire.

### Optimisation des poids

```bash
# Vérifier les statistiques de comparaison
python facet.py --comparison-stats

# Optimiser les poids à partir des comparaisons (appliqués seulement si cela généralise)
python facet.py --optimize-weights --optimize-category portrait

# Restreindre les données d'entraînement à des sources spécifiques
python facet.py --optimize-weights --optimize-category portrait --optimize-sources vote,culling

# Appliquer même si le seuil de validation tenue à l'écart n'est pas atteint
python facet.py --optimize-weights --optimize-category portrait --optimize-force

# Appliquer à toutes les photos
python facet.py --recompute-average
```

### Pipeline étiquette-vers-poids

Au-delà des votes A/B explicites, deux autres flux d'étiquettes alimentent l'optimiseur :

1. **Les décisions de tri** sont capturées automatiquement à chaque
   confirmation de rafale/similaires (`source='culling'`).
2. **Les notes en étoiles, favoris et rejets** sont matérialisés en paires
   synthétiques avec `python facet.py --sync-label-comparisons` (`source='rating'`).
   Une nouvelle exécution resynchronise à partir des étiquettes actuelles, de sorte que les notes retirées disparaissent.

L'optimiseur cherche les poids qui prédisent le mieux quelle photo l'a emporté dans chaque comparaison (un modèle de Bradley-Terry), en comptant chaque source selon sa fiabilité : vote 1.0, rating 0.7, culling 0.5. Une **égalité** compte comme une demi-victoire plus une demi-défaite : un indice que les deux photos devraient obtenir un score proche. L'ajustement est **ancré sur vos poids actuels** : avec peu de comparaisons, la suggestion reste proche d'eux et s'en éloigne à mesure que les votes s'accumulent. Il s'entraîne sur chaque photo telle que le scoreur la voit : les mêmes métriques 0-10 que lit l'agrégat (y compris `liqe`, `aesthetic_iaa`, `face_quality_iqa` et les métriques de saillance du sujet), avec netteté des yeux, netteté technique et couleur normalisées par percentiles, le score d'exposition reconstruit à partir de l'histogramme stocké et le drapeau de portrait de groupe. Les percentiles sont ceux que le dernier `--recompute-average` a enregistrés ; avant le premier `--recompute-average`, les colonnes stockées sont utilisées telles quelles. Il part des poids que le scoreur utilise réellement, et écrit ces mêmes poids : `quality_percent` est replié dans `aesthetic` comme le fait le scoreur, et l'application écrit `quality_percent: 0` dans les catégories qui l'ont.

Les poids **ne sont appliqués que s'ils se généralisent**. Les poids finaux sont ajustés sur toutes les comparaisons, mais la décision de les écrire se prend sur des comparaisons que l'ajustement n'a pas vues : validation croisée à 5 plis avec des **plis groupés par photo** (aucune photo n'est à la fois dans les paires d'entraînement et dans les paires réservées d'un pli). Les prédictions réservées, agrégées sur les plis, sont comparées à celles des poids actuels sur les mêmes paires par un **test exact de McNemar unilatéral**, qui demande si les nouveaux poids gagnent significativement plus des paires sur lesquelles les deux jeux de poids divergent. Les poids ne sont appliqués que si `p < 0.05` **et** que le gain réservé est d'au moins 2 points de pourcentage ; sinon l'exécution consigne les chiffres et n'écrit rien. Passez `--optimize-force` pour appliquer quand même. Si la barrière ne peut pas s'exécuter (trop peu de groupes de photos indépendants), rien n'est appliqué sauf si vous forcez. L'optimisation est par catégorie et exige au moins 30 comparaisons **décisives** (une photo l'a emporté) **dans cette catégorie** ; les égalités et les comparaisons des autres catégories ne comptent pas. Une catégorie sans votes ne peut pas être ajustée à partir des données. Le journal affiche par exemple :

```
Held-out accuracy:      69.2% +/- 4.0 (5-fold, photo-grouped) - the honest estimate
Held-out gain:          +8.5 pp over the current weights on the same 389 pairs
McNemar (one-sided):    p = 0.0006 (only the fit right: 66, only the current weights right: 33)
NOT applying: the held-out gate needs p < 0.05 and a gain of at least 2.0 pp. ...   (logged when the gate fails)
```

Appliquer un résultat avec `--optimize-weights` change la façon dont les photos de la catégorie se classent entre elles, pas la place de la catégorie face aux autres. Dans la même écriture de configuration que les poids, `modifiers.bonus` est décalé de l'opposé du décalage moyen projeté du score des photos de la catégorie, si bien que l'agrégat moyen de la catégorie reste où il était. Le bonus reste dans [-5, 5], et le journal indique quand cette limite écourte la correction. Une catégorie sans entrée dans `categories` (`others`) n'a pas de bonus et n'est pas recentrée. L'instantané de poids pris avant l'écriture inclut le bonus, et restaurer un instantané restaure poids **et** bonus (les instantanés enregistrés avant ce changement, par `calibrate.py` ou pour `others` n'ont pas de bonus et le laissent intact). Limite connue : le bouton **Appliquer** de la visionneuse et une modification manuelle dans l'onglet Poids écrivent les poids via `POST /api/config/update_weights`, qui ne recentre pas le bonus ; seul `--optimize-weights` le fait.

**Après une mise à jour, ré-entraînez le classeur personnel et la tête keeper (`--train-ranker`, `--train-keeper`), puis lancez `--recompute-average`.** L'optimiseur, le classeur et la tête keeper lisent les mêmes métriques par photo, qui incluent désormais la netteté des yeux, la netteté technique et la couleur normalisées par percentiles ; un modèle enregistré par une version précédente ne peut pas détecter le changement.

Cadence recommandée :

```bash
python facet.py --mine-insights          # quel signal existe, dérive, santé
python facet.py --sync-label-comparisons # rafraîchir les paires dérivées des notes
python facet.py --optimize-weights       # apprendre les poids de toutes les sources
python facet.py --recompute-average      # appliquer + persister l'instantané de percentiles
```

### Réglage des poids dans l'interface

1. Ouvrez le panneau d'aperçu des poids pendant la comparaison
2. Ajustez les curseurs pour voir les changements de score en temps réel
3. Cliquez sur « Suggérer les poids » pour les valeurs optimisées
4. Mettez à jour la config manuellement

**Suggérer des poids** répond aussi à une question plus précise que la barrière réservée décrite plus haut : dans quelle mesure les poids *actuellement utilisés* de cette catégorie concordent-ils déjà avec vos propres comparaisons ? Il renvoie `accuracy_before` — le pourcentage des paires étiquetées de cette catégorie (votes A/B, rejets et paires dérivées des notes) dont les poids en vigueur prédisent correctement le gagnant — à côté de `accuracy_after`, le même chiffre pour les poids suggérés. Les deux sont mesurés sur les mêmes comparaisons que celles qui ont servi à ajuster les poids : `accuracy_after` est donc optimiste. Ils s'affichent côte à côte dans l'onglet Suggestions de poids et dans la barre latérale de l'onglet Comparaison A/B à chaque exécution (`GET /api/comparison/learned_weights`). L'offre d'**Appliquer** suit la même barrière réservée de McNemar que la CLI, pas ce gain optimiste : `suggest_changes` n'est vrai que si la barrière est franchie. La requête n'exécute la validation croisée que lorsque le gain optimiste atteint déjà 2 points de pourcentage (`?use_cv=true` l'exécute dans tous les cas). La visionneuse exige `min_comparisons_for_optimization` (50 par défaut) comparaisons décisives dans cette catégorie, là où la CLI en exige 30 ; en dessous, le bouton signale le manque (« Need at least N decisive comparisons (have K) ») au lieu d'un nombre.

## Ajouter des catégories personnalisées

```json
{
  "name": "underwater",
  "priority": 62,
  "filters": {
    "required_tags": ["underwater"],
    "tag_match_mode": "any"
  },
  "weights": {
    "aesthetic_percent": 40,
    "color_percent": 25,
    "composition_percent": 20,
    "exposure_percent": 15
  },
  "modifiers": {
    "noise_tolerance_multiplier": 0.3,
    "bonus": 0.5
  },
  "tags": {
    "underwater": ["underwater", "scuba", "diving", "ocean"],
    "fish": ["fish", "coral", "reef"]
  }
}
```

Ajoutez-la au tableau `categories` dans `scoring_config.json`, puis exécutez `--recompute-average` (ou `--recompute-category underwater` pour la nouvelle catégorie uniquement).

## Exemples de flux de travail

### Régler la catégorie Concert

```bash
# Modifier scoring_config.json :
# Trouver la catégorie "concert", ajuster :
#   "noise_tolerance_multiplier": 0.05
#   "exposure_percent": 5

python facet.py --recompute-category concert
```

Ou utilisez l'éditeur de poids de la galerie sous `/stats` → Catégories → Poids pour un aperçu en direct et un recalcul en un clic.

### Passer au profil 8gb

```bash
# Modifier : "vram_profile": "8gb"
python facet.py --compute-recommendations  # Analyser
# Réduire aesthetic_percent dans les catégories si nécessaire
python facet.py --recompute-average
```

### Ajouter la catégorie Underwater

1. Ajoutez la définition de catégorie (voir ci-dessus)
2. Exécutez `python facet.py --validate-categories`
3. Exécutez `python facet.py --recompute-average`
