# Reconnaissance faciale

> 🌐 [English](../FACE_RECOGNITION.md) · **Français** · [Deutsch](../de/FACE_RECOGNITION.md) · [Italiano](../it/FACE_RECOGNITION.md) · [Español](../es/FACE_RECOGNITION.md) · [Português](../pt/FACE_RECOGNITION.md) · [简体中文](../zh/FACE_RECOGNITION.md)

Facet utilise InsightFace pour la détection des visages et HDBSCAN pour regrouper les visages en personnes.

## Vue d'ensemble

1. **Détection** - le modèle InsightFace buffalo_l détecte les visages et extrait des embeddings de 512 dimensions
2. **Regroupement** - HDBSCAN regroupe les embeddings similaires en clusters de personnes
3. **Gestion** - galerie web pour fusionner, renommer et organiser les personnes

## Workflow complet

### Étape 1 : Extraire les visages

Pendant l'analyse des photos, les visages sont extraits automatiquement :

```bash
python facet.py /path/to/photos
```

Pour les photos existantes sans visages :

```bash
python facet.py --extract-faces-gpu-incremental  # Nouvelles photos uniquement
python facet.py --extract-faces-gpu-force        # Toutes les photos (supprime l'existant)
```

### Étape 2 : Regrouper les visages

Regrouper les visages similaires en personnes :

```bash
python facet.py --cluster-faces-incremental  # Préserve les personnes existantes
```

**Modes de regroupement :**

| Commande | Comportement |
|---------|----------|
| `--cluster-faces-incremental` | Préserve toutes les personnes, rattache les nouveaux visages aux existants |
| `--cluster-faces-incremental-named` | Préserve uniquement les personnes nommées |
| `--cluster-faces-force` | Supprime toutes les personnes, re-regroupement complet |

### Étape 3 : Examiner et fusionner

Trouver les clusters de personnes en doublon :

```bash
python facet.py --suggest-person-merges
python facet.py --suggest-person-merges --merge-threshold 0.7  # Plus strict
```

Ouvre le navigateur sur la page des suggestions de fusion.

### Étape 4 : Gestion dans la visionneuse

Le travail restant se déroule dans la visionneuse web, en suivant le pipeline **Extraire → Regrouper → Fusionner → Gérer** :

- **Fusionnez** les clusters en double sur la page Suggestions de fusion.
- **Gérez** les personnes (fusionner, fusion par lot, scinder, masquer, renommer, supprimer) sur la page Gérer les personnes.

Voir [Intégration à la galerie](#intégration-à-la-galerie) pour la référence complète de l'interface.

## Configuration

### Détection des visages

```json
{
  "face_detection": {
    "min_confidence_percent": 65,
    "min_face_size": 20,
    "blink_ear_threshold": 0.28
  }
}
```

| Paramètre | Défaut | Description |
|---------|---------|-------------|
| `min_confidence_percent` | `65` | Confiance de détection minimale |
| `min_face_size` | `20` | Taille de visage minimale en pixels — InsightFace uniquement : le détecteur YuNet du mode commercial utilise des valeurs fixes mesurées (confiance 0,7, 30 px) et ignore `min_confidence_percent` ainsi que ce réglage. |
| `blink_ear_threshold` | `0.28` | Eye Aspect Ratio pour la détection des clignements |

### Regroupement des visages

```json
{
  "face_clustering": {
    "enabled": true,
    "min_faces_per_person": 2,
    "min_samples": 2,
    "clustering_algorithm": "best",
    "leaf_size": 40,
    "use_gpu": "auto",
    "thresholds_by_model": {
      "arcface_buffalo_l": { "auto_merge_distance_percent": 15, "merge_threshold": 0.6 },
      "sface": { "auto_merge_distance_percent": 27.5, "merge_threshold": 0.65 }
    }
  }
}
```

| Paramètre | Défaut | Description |
|---------|---------|-------------|
| `min_faces_per_person` | `2` | Nombre minimal de photos pour créer une personne |
| `min_samples` | `2` | Paramètre min_samples de HDBSCAN |
| `thresholds_by_model` | voir JSON | Par modèle de reconnaissance : `auto_merge_distance_percent` (distance de fusion automatique) et `merge_threshold` (similarité des centroïdes). Valeurs par défaut : ArcFace `15` / `0.6`, SFace `27.5` / `0.65` |
| `use_gpu` | `"auto"` | Mode GPU : `auto`, `always`, `never` |

### Traitement des visages

```json
{
  "face_processing": {
    "crop_padding": 0.3,
    "use_db_thumbnails": true,
    "face_thumbnail_size": 640,
    "face_thumbnail_quality": 90,
    "extract_workers": 2,
    "extract_batch_size": 16,
    "refill_workers": 4,
    "refill_batch_size": 100
  }
}
```

## Algorithmes de regroupement

Pour le regroupement sur CPU, choisissez l'algorithme en fonction de la taille du jeu de données :

| Algorithme | Complexité | Recommandé pour |
|-----------|------------|----------|
| `boruvka_balltree` | O(n log n) | Haute dimension (recommandé pour 50K+ visages) |
| `boruvka_kdtree` | O(n log n) | Données en basse dimension |
| `prims_balltree` | O(n²) | Petits jeux de données, mémoire limitée |
| `prims_kdtree` | O(n²) | Petits jeux de données |
| `best` | Auto | Laisser HDBSCAN décider |

**Note de performance :** pour les grands jeux de données, utilisez `boruvka_balltree`. Avec 80K visages, il s'achève en 2 à 5 minutes, là où les algorithmes exacts peuvent se bloquer.

## Regroupement GPU (cuML)

Pour les grands jeux de données (80K+ visages), le regroupement GPU via RAPIDS cuML est plus rapide que sur CPU.

### Installation

```bash
# Conda
conda install -c rapidsai -c conda-forge -c nvidia cuml cuda-version=12.0

# Pip
pip install --extra-index-url https://pypi.nvidia.com/ "cuml-cu12"
```

### Configuration

```json
{
  "face_clustering": {
    "use_gpu": "auto"
  }
}
```

| Mode | Comportement |
|------|----------|
| `"auto"` | Utiliser le GPU si cuML est disponible, repli sur CPU |
| `"always"` | Essayer le GPU, avertir et se replier s'il est indisponible |
| `"never"` | Toujours utiliser le CPU |

**Note :** cuML utilise sa propre implémentation de HDBSCAN. Les paramètres `algorithm` et `leaf_size` ne s'appliquent qu'au regroupement sur CPU.

### Le GPU et le CPU ne produisent pas des groupes identiques

Mesuré sur une bibliothèque de 145 677 visages, avec les mêmes embeddings et les mêmes paramètres :

| Visages | Groupes cuML | Groupes CPU | Rapport |
|--------:|-------------:|------------:|--------:|
| 5 000 | 202 | 181 | 1,12× |
| 20 000 | 1 285 | 1 039 | 1,24× |

cuML est légèrement plus granulaire, avec un taux de bruit comparable et une taille médiane de groupe identique (4). L'écart reste trop faible pour changer qui se retrouve groupé avec qui, mais le *nombre* de groupes ne correspondra pas entre une exécution GPU et une exécution CPU.

**cuML n'est pas non plus déterministe.** Deux exécutions GPU consécutives sur les mêmes 145 677 visages, à paramètres identiques, ne s'accordaient que sur **41 % des étiquettes**, tout en produisant un nombre de groupes quasi identique (14 809 contre 14 807) et des résultats finaux identiques. La structure est stable, l'étiquetage ne l'est pas. Conséquences à connaître :

- **Les identifiants de personnes ne sont pas reproductibles d'une exécution à l'autre.** Ne supposez pas qu'un nouveau regroupement conserve la numérotation des personnes auto-groupées — nommez celles qui comptent, c'est ce qui les rend stables.
- **Ne comparez pas deux regroupements par étiquette.** Comparez plutôt l'appartenance des visages.

Le regroupement CPU avec un `algorithm` fixe est déterministe pour une même entrée.

### Regroupement et personnes déjà nommées

Tous les modes sauf `--cluster-faces-force` conservent les enregistrements de personnes, mais tous effacent les affectations visage→personne et les reconstruisent. Un groupe reconstruit est rendu à une personne lorsque :

1. son embedding moyen se situe à moins de `merge_threshold` (0,6 cosinus par défaut pour ArcFace, 0,65 pour SFace) du centroïde stocké de cette personne, **ou**
2. au moins la moitié de ses visages appartenaient à cette personne avant l'exécution.

La seconde règle compte plus qu'il n'y paraît. Le centroïde d'une personne est un unique vecteur moyen : quelqu'un photographié pendant des années couvre une région large, et HDBSCAN le découpe correctement en sous-groupes serrés qui se retrouvent loin de cette moyenne et échouent à la règle 1. Sur la bibliothèque ci-dessus, la règle 1 seule rendait 38 % des visages rattachés aux personnes nommées ; avec la règle 2, on atteint 96 %.

Comme la règle 2 s'en remet à l'affectation précédente, une erreur ancienne est reconduite plutôt que corrigée. C'est le contrat attendu d'un mode dont le but est de préserver les personnes existantes — utilisez `--cluster-faces-force` pour tout recalculer de zéro.

## Détection des clignements

Utilise l'Eye Aspect Ratio (EAR) calculé à partir des 106 points de repère d'InsightFace.

### Fonctionnement

L'EAR mesure le rapport entre la hauteur et la largeur de l'œil. Lorsque les yeux se ferment, l'EAR descend sous le seuil.

### Configuration

```json
{
  "face_detection": {
    "blink_ear_threshold": 0.28
  }
}
```

Un seuil plus bas = détection plus stricte (plus de photos marquées comme clignements).

### Recalculer après un changement de seuil

```bash
python facet.py --recompute-blinks
```

Ne traite que les photos comportant des visages, aucun GPU nécessaire.

## Qualité de visage OFIQ optionnelle (mode commercial)

En mode commercial, le modèle restreint de qualité de visage est indisponible. Comme remplacement optionnel, installez `python-ofiq` (liaisons UNICEF sur OFIQ du BSI, ISO/IEC 29794-5 ; MIT ; compatible avec numpy 2 et OpenCV 4.8+) avec `pip install python-ofiq`. Il ne fournit des wheels que pour Linux x86-64 et macOS arm64 (sous Windows : WSL2 ou Docker). Les modèles OFIQ (environ 400 Mo, MIT/Apache-2.0) ne sont jamais fournis : au premier usage, Facet les télécharge, SHA-256 épinglé, dans `pretrained_models/ofiq/`. Le pipeline facial commercial divise par 10 le score unifié OFIQ (0-100) et le stocke dans `face_quality_iqa` ; la valeur d'une photo est 0,7 x son pire visage + 0,3 x la moyenne. Sans `python-ofiq`, la colonne reste NULL et son poids est renormalisé.

## Signaux d'expression par visage (yeux ouverts + sourire)

Chaque ligne de visage stocke deux signaux continus sur une échelle de 0 à 10, utilisés par le
panneau de visages du tri et les agrégats au niveau de la photo : `eyes_open_score` (10 = grand
ouvert, 0 = complètement fermé) et `smile_score` (5 = neutre, 10 = large sourire, 0 = moue).

Deux moteurs les produisent, sur la même échelle 0-10 :

1. **Géométrie (toujours disponible).** Dérivé des 106 points de repère InsightFace stockés :
   l'Eye Aspect Ratio pour les yeux ouverts, le soulèvement des coins de la bouche pour le sourire.
   Pure géométrie, donc `--recompute-face-signals` peut les recalculer à partir des points de
   repère stockés sans aucun pixel et sans GPU.
2. **Blendshapes MediaPipe (optionnel, basé sur l'apparence).** Pendant l'analyse / l'extraction
   des visages, un recadrage généreux de chaque visage passe par le MediaPipe Face Landmarker,
   dont les blendshapes de style ARKit (`eyeBlink*`, `mouthSmile*`, `mouthFrown*`) correspondent
   aux mêmes échelles. L'apparence surpasse la géométrie des points de repère sur les yeux fermés,
   les sourires discrets et les têtes de profil, donc quand un visage est noté via MediaPipe, cela
   **remplace** la valeur géométrique. Si MediaPipe ou son modèle est absent, ou si le recadrage du
   visage est trop petit / non détecté, la valeur géométrique est conservée — le comportement est
   identique à une installation géométrie seule.

### Installer MediaPipe

MediaPipe est optionnel et **doit** être installé sans son `opencv-contrib-python` embarqué, qui
installerait un second espace de noms `cv2` à côté de celui de Facet (`opencv-python`) :

```bash
pip install mediapipe==0.10.35 --no-deps
pip install absl-py flatbuffers
```

Ne lancez jamais un simple `pip install mediapipe`.

### Modèle

Le modèle `face_landmarker.task` (~3,6 Mio, Apache-2.0) est téléchargé automatiquement à la
première utilisation vers `pretrained_models/face_landmarker.task`. Si la machine est hors ligne,
téléchargez-le manuellement depuis
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
et placez-le à cet emplacement. Un téléchargement échoué journalise un avertissement une seule fois
et revient aux scores géométriques.

### Configuration

```json
{
  "face_detection": {
    "blendshapes": {
      "enabled": true,
      "min_crop_size": 192
    }
  }
}
```

- `enabled` (défaut `true`) : utiliser les scores blendshapes dès que MediaPipe et le modèle
  sont disponibles ; sinon le repli géométrique s'exécute automatiquement. Mettre à `false`
  pour forcer la géométrie seule.
- `min_crop_size` (défaut `192`) : les visages dont le recadrage rembourré est plus petit que
  cette valeur (px, côté le plus court) retombent sur la géométrie plutôt que d'agrandir un visage
  minuscule.

### Recalcul

`--recompute-face-signals` recalcule les signaux par visage uniquement à partir des points de
repère stockés — il est **géométrie seule** et n'exécute pas MediaPipe (aucun pixel n'est lu). Pour
rafraîchir les scores basés sur l'apparence, ré-extrayez les visages (`--extract-faces-gpu-force`)
afin que les recadrages en pleine résolution soient réanalysés.

## Miniatures de visages

Les miniatures sont stockées dans la base de données pour un affichage rapide.

### Stockage

- Générées pendant l'analyse à partir des images en pleine résolution
- Stockées dans la colonne `faces.face_thumbnail` sous forme de BLOBs JPEG (~5 à 10 Ko chacune)
- Utilisées par le regroupement et la galerie au lieu d'être régénérées

### Régénération

```bash
# Générer les miniatures manquantes
python facet.py --refill-face-thumbnails-incremental

# Régénérer TOUTES les miniatures
python facet.py --refill-face-thumbnails-force
```

Les deux commandes utilisent le traitement parallèle pour plus de rapidité.

## Schéma de base de données

### Table faces

| Colonne | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Clé primaire |
| `photo_path` | TEXT | Clé étrangère vers photos |
| `face_index` | INTEGER | Index au sein de la photo |
| `embedding` | BLOB | Embedding de visage de 512 dimensions |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | INTEGER | Coins de la boîte englobante |
| `confidence` | REAL | Confiance de détection |
| `person_id` | INTEGER | Clé étrangère vers persons |
| `face_thumbnail` | BLOB | Miniature JPEG |
| `landmark_2d_106` | BLOB | 106 points de repère (détection des clignements) |
| `embedding_model` | TEXT | Étiquette du modèle de reconnaissance (par défaut `arcface_buffalo_l`) |

### Table persons

| Colonne | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Clé primaire |
| `name` | TEXT | Nom de la personne (NULL = regroupé automatiquement) |
| `representative_face_id` | INTEGER | Meilleur visage pour l'avatar |
| `face_count` | INTEGER | Nombre de visages |
| `centroid` | BLOB | Embedding du centroïde du cluster |
| `auto_clustered` | INTEGER | 1 si généré automatiquement |
| `face_thumbnail` | BLOB | Miniature d'avatar de la personne |
| `is_hidden` | INTEGER | 1 = exclu des filtres/suggestions |

## Modes incrémental et forcé

### Regroupement incrémental

- Préserve toutes les personnes existantes (nommées et regroupées automatiquement)
- Ne regroupe que les nouveaux visages non assignés
- Rattache les nouveaux clusters aux personnes existantes via la similarité de centroïde
- Met à jour les centroïdes après fusion

**À utiliser quand :** vous ajoutez de nouvelles photos à une collection existante

### Regroupement forcé

- Supprime TOUTES les personnes, y compris celles nommées
- Re-regroupement complet à partir de zéro

**À utiliser quand :** vous repartez de zéro ou changez l'algorithme en profondeur

### Regroupement incrémental nommé

- Préserve uniquement les personnes nommées
- Supprime les personnes regroupées automatiquement
- Re-regroupe tous les visages non nommés

**À utiliser quand :** vous conservez des noms organisés tout en rafraîchissant les clusters détectés automatiquement

## Intégration à la galerie

### Filtre par personne

- Le menu déroulant affiche les personnes avec leurs miniatures de visages
- Filtrer la galerie par personne

### Galerie d'une personne

- Cliquez sur une personne dans le menu déroulant pour voir toutes ses photos
- URL : `/person/<id>`

### Page Gérer les personnes

Accès via le bouton d'en-tête ou `/persons` :

- **Vue en grille** - toutes les personnes reconnues
- **Fusionner** - sélectionnez la source, cliquez sur la cible, confirmez
- **Fusion par lot** - sélectionnez plusieurs personnes et fusionnez-les dans une seule cible
- **Scinder** - déplacez les visages sélectionnés vers une nouvelle personne
- **Masquer** - exclure un cluster de la liste, des filtres et des suggestions de fusion
- **Supprimer** - retirez le cluster de personne
- **Renommer** - cliquez sur le nom pour le modifier en ligne

### Créer une personne

Les personnes ne proviennent plus uniquement du regroupement automatique — vous pouvez nommer un
visage que le clusterer a manqué directement depuis la galerie :

1. Sur une carte photo, ouvrez les actions de personne et choisissez un visage non assigné.
2. Dans le sélecteur de personne, choisissez **Créer une nouvelle personne** et saisissez un nom.
3. Le visage est rattaché à la nouvelle personne (créée manuellement, `auto_clustered = 0`) en un
   seul appel.

Endpoint : `POST /api/persons` (réservé à l'édition), corps
`{ "name": "<nom>", "face_ids": [<id>, ...] }`. Le nom est requis (non vide après suppression des
espaces). Les visages déjà rattachés à une autre personne sont réassignés, et toute ancienne
personne qui se retrouve sans visage est supprimée — les mêmes règles que l'assignation de visage.
En mode multi-utilisateur, l'appelant ne peut rattacher que des visages issus de photos de ses
propres répertoires (ou partagés) ; un visage hors de ce périmètre est rejeté comme introuvable.

### À nommer

La page Gérer les personnes affiche les personnes regroupées automatiquement qui méritent d'être
nommées dans une section **À nommer** : les clusters non nommés (`name IS NULL`,
`auto_clustered = 1`) comportant au moins `viewer.persons.needs_naming_min_faces` visages (défaut
`5`), chacun avec un champ de nom en ligne afin que les grands clusters puissent être nommés sans
avoir à les chercher. Servi par `GET /api/persons/needs_naming?min_faces=N`.

### Page Suggestions de fusion

Accès via `/merge-suggestions` ou le bouton « Suggestions de fusion » de la page Gérer les personnes :

- Présente des paires de personnes aux embeddings de visages similaires susceptibles d'être le même individu
- **Curseur de seuil** — contrôle le seuil de similarité (plus bas = plus de suggestions)
- **Fusion en un clic** — fusionnez instantanément une paire suggérée
- **Fusion par lot** — sélectionnez plusieurs suggestions et fusionnez-les toutes en une fois

### Cartes de photos

- De petites miniatures de visages (avatars) sont affichées pour les personnes reconnues
- Configurable via `viewer.face_thumbnails.output_size_px`

## Marqueur d'espace d'embedding (sécurité du modèle de reconnaissance)

Chaque ligne de visage porte une étiquette `embedding_model` (colonne de `faces`, par défaut
`arcface_buffalo_l` — l'actuel modèle de reconnaissance InsightFace `buffalo_l` / ArcFace `w600k_r50`).
Les embeddings produits par des modèles de reconnaissance **différents** vivent dans des
espaces vectoriels **incompatibles** et ne doivent jamais être regroupés ensemble — le faire
produit silencieusement des personnes erronées, sans aucune erreur.

`FaceClusterer.load_embeddings()` ne charge donc que l'espace d'embedding **actif**
(`active_embedding_model()` dans `db/embedding_space.py` ; une étiquette `NULL` est traitée
comme l'ancien espace ArcFace) et émet un avertissement bien visible si des visages d'un autre
espace sont présents et exclus. Il s'agit d'une protection de compatibilité ascendante : elle
rend sûr par construction un futur changement de modèle de reconnaissance.

### Changer de modèle de reconnaissance (ex. AdaFace) — plan différé

Une amélioration de qualité telle qu'**AdaFace** (marge adaptative à la qualité, meilleur
regroupement des visages flous/candides) est intégrable comme backend 512-d optionnel (même
chemin de stockage, même HDBSCAN), mais n'est **pas encore implémentée** car elle ne peut être
validée sans données réelles. La réaliser correctement nécessite :

1. **Poids + backbone** — un checkpoint AdaFace (ex. `adaface_ir101_webface12m`)
   ainsi que son backbone IResNet ; un nouveau téléchargement dans le cache de modèles.
2. **Crops alignés** — calculer l'embedding à partir d'un crop aligné 112×112
   `norm_crop(img, face.kps, 112)` au moment de l'extraction (les kps existent sur l'objet
   `face` d'InsightFace mais ne sont pas persistés, donc AdaFace ne peut pas être recalculé
   hors ligne — il doit s'exécuter pendant l'extraction). Vérifier que le BGR/la normalisation
   correspondent au checkpoint.
3. **Bascule de configuration** — ajouter `face_detection.recognition_model: arcface|adaface`
   et en déduire `active_embedding_model()` ; étiqueter les nouveaux visages en conséquence.
4. **Ré-extraction + re-regroupement complets** — `--extract-faces-gpu-force` puis
   `--cluster-faces-force`, car les embeddings ArcFace et AdaFace ne sont pas
   comparables. Le marqueur d'espace d'embedding ci-dessus empêche qu'une base à moitié
   migrée regroupe silencieusement les deux espaces (elle avertit et exclut à la place).
5. **Validation de la qualité** — mesurer la qualité des clusters par rapport à des identités
   étiquetées ; « ça tourne et émet des vecteurs 512-d » ne prouve pas que le prétraitement est correct.

### Bibliothèques mixtes, ré-extraction et valeurs inconnues

- **Bibliothèques mixtes.** Une bibliothèque peut contenir à la fois des visages ArcFace (512-d) et SFace (128-d). Le clustering, les suggestions de fusion, la recherche de personne similaire et les centroïdes de personnes n'utilisent chacun qu'un modèle à la fois (`persons.centroid_model` indique lequel) ; les modes de regroupement incrémental et « personnes nommées seulement » ne touchent ni les visages ni les personnes de l'autre modèle, mais `--cluster-faces-force` efface toutes les affectations visage→personne et supprime toutes les personnes, nommées comprises, quel que soit leur modèle. Pour faire passer une bibliothèque au modèle actif, lancez `--extract-faces-gpu-force` (en mode commercial, il utilise l'extracteur permissif ; les personnes nommées sont conservées) puis `--cluster-faces-incremental`.
- **Avertissements.** Un scan journalise un avertissement quand des visages stockés appartiennent à un autre modèle que l'actif, et `--doctor` signale le même écart par rapport au modèle actif.
- **Les personnes nommées survivent à la ré-extraction.** Avant que `--extract-faces-gpu-force` ne supprime les lignes de visages, Facet retient la boîte de chaque visage d'une personne nommée ; ensuite chaque visage re-détecté est rendu à la personne dont l'ancienne boîte le recouvre avec une IoU >= 0,5 (un pour un, meilleur recouvrement d'abord), et les centroïdes sont recalculés par modèle. Un visage sans ancienne boîte correspondante reste non assigné jusqu'au prochain clustering. Les boîtes mémorisées sont stockées dans la table `face_reextract_snapshot`, dans la même transaction que la suppression : une exécution interrompue est donc reprise par l'extraction suivante. Rescanner une photo conserve ses personnes nommées de la même façon.
- **Inconnu n'est pas zéro.** Sans MediaPipe, l'analyseur permissif ne peut pas mesurer les yeux, le sourire ni les clignements ; ces valeurs sont stockées à NULL (ni 0, ni « clignement »). Le filtre et la pénalité de clignement traitent NULL comme « pas un clignement ». Les lecteurs des 106 points de repère ignorent les visages sans repères.

## Dépannage

| Problème | Solution |
|-------|----------|
| Le regroupement se bloque | Utilisez l'algorithme `boruvka_balltree` |
| Trop de petits clusters | Augmentez `min_faces_per_person` |
| Les visages ne se regroupent pas | Diminuez `merge_threshold` |
| Le regroupement GPU échoue | Vérifiez l'installation de cuML, utilisez `"never"` pour forcer le CPU |
| Miniatures manquantes | Exécutez `--refill-face-thumbnails-incremental` |
| Détection de clignement erronée | Ajustez `blink_ear_threshold`, exécutez `--recompute-blinks` |
| Avertissement « Excluded N faces from non-active embedding space » | Un changement de modèle de reconnaissance a laissé des embeddings mélangés — exécutez `--extract-faces-gpu-force` puis `--cluster-faces-incremental` |
