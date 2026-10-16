# Gesichtserkennung

> 🌐 [English](../FACE_RECOGNITION.md) · [Français](../fr/FACE_RECOGNITION.md) · **Deutsch** · [Italiano](../it/FACE_RECOGNITION.md) · [Español](../es/FACE_RECOGNITION.md) · [Português](../pt/FACE_RECOGNITION.md) · [简体中文](../zh/FACE_RECOGNITION.md)

Facet verwendet InsightFace zur Gesichtserkennung und HDBSCAN, um Gesichter zu Personen zu clustern.

## Überblick

1. **Erkennung** – Das InsightFace-Modell buffalo_l erkennt Gesichter und extrahiert 512-dimensionale Embeddings
2. **Clustering** – HDBSCAN gruppiert ähnliche Embeddings zu Personen-Clustern
3. **Verwaltung** – Web-Viewer zum Zusammenführen, Umbenennen und Organisieren von Personen

## Vollständiger Arbeitsablauf

### Schritt 1: Gesichter extrahieren

Während des Foto-Scans werden Gesichter automatisch extrahiert:

```bash
python facet.py /path/to/photos
```

Für vorhandene Fotos ohne Gesichter:

```bash
python facet.py --extract-faces-gpu-incremental  # Nur neue Fotos
python facet.py --extract-faces-gpu-force        # Alle Fotos (löscht vorhandene)
```

### Schritt 2: Gesichter clustern

Ähnliche Gesichter zu Personen gruppieren:

```bash
python facet.py --cluster-faces-incremental  # Erhält vorhandene Personen
```

**Clustering-Modi:**

| Befehl | Verhalten |
|---------|----------|
| `--cluster-faces-incremental` | Behält alle Personen bei, ordnet neue den vorhandenen zu |
| `--cluster-faces-incremental-named` | Behält nur benannte Personen bei |
| `--cluster-faces-force` | Löscht alle Personen, vollständiges Re-Clustering |

### Schritt 3: Überprüfen und zusammenführen

Doppelte Personen-Cluster finden:

```bash
python facet.py --suggest-person-merges
python facet.py --suggest-person-merges --merge-threshold 0.7  # Strenger
```

Öffnet den Browser mit der Seite der Zusammenführungsvorschläge.

### Schritt 4: Verwaltung im Viewer

Die verbleibende Arbeit erfolgt im Web-Viewer, entlang der Pipeline **Extrahieren → Clustern → Zusammenführen → Verwalten**:

- **Zusammenführen** doppelter Cluster auf der Seite mit Zusammenführungsvorschlägen.
- **Verwalten** von Personen (zusammenführen, Stapel-Zusammenführung, aufteilen, ausblenden, umbenennen, löschen) auf der Seite „Personen verwalten“.

Siehe [Viewer-Integration](#viewer-integration) für die vollständige Referenz der Benutzeroberfläche.

## Konfiguration

### Gesichtserkennung

```json
{
  "face_detection": {
    "min_confidence_percent": 65,
    "min_face_size": 20,
    "blink_ear_threshold": 0.28
  }
}
```

| Einstellung | Standard | Beschreibung |
|---------|---------|-------------|
| `min_confidence_percent` | `65` | Mindestkonfidenz der Erkennung |
| `min_face_size` | `20` | Mindestgröße des Gesichts in Pixeln — Nur InsightFace: der YuNet-Detektor im kommerziellen Modus verwendet feste, gemessene Werte (0,7 Konfidenz, 30 px) und ignoriert `min_confidence_percent` sowie diese Einstellung. |
| `blink_ear_threshold` | `0.28` | Eye Aspect Ratio für die Blinzelerkennung |

### Gesichts-Clustering

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

| Einstellung | Standard | Beschreibung |
|---------|---------|-------------|
| `min_faces_per_person` | `2` | Mindestanzahl Fotos, um eine Person zu erstellen |
| `min_samples` | `2` | HDBSCAN-Parameter min_samples |
| `thresholds_by_model` | siehe JSON | Pro Erkennungsmodell: `auto_merge_distance_percent` (Auto-Merge-Abstand) und `merge_threshold` (Zentroid-Ähnlichkeit). Standard: ArcFace `15` / `0.6`, SFace `27.5` / `0.65` |
| `use_gpu` | `"auto"` | GPU-Modus: `auto`, `always`, `never` |

### Gesichtsverarbeitung

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

## Clustering-Algorithmen

Wählen Sie für CPU-Clustering den Algorithmus je nach Datensatzgröße:

| Algorithmus | Komplexität | Am besten geeignet für |
|-----------|------------|----------|
| `boruvka_balltree` | O(n log n) | Hochdimensionale Daten (empfohlen für 50K+ Gesichter) |
| `boruvka_kdtree` | O(n log n) | Niedrigdimensionale Daten |
| `prims_balltree` | O(n²) | Kleine Datensätze, speicherbeschränkt |
| `prims_kdtree` | O(n²) | Kleine Datensätze |
| `best` | Auto | HDBSCAN entscheiden lassen |

**Leistungshinweis:** Verwenden Sie für große Datensätze `boruvka_balltree`. Mit 80K Gesichtern wird es in 2-5 Minuten abgeschlossen, wo exakte Algorithmen hängen bleiben können.

## GPU-Clustering (cuML)

Für große Datensätze (80K+ Gesichter) ist GPU-Clustering über RAPIDS cuML schneller als CPU.

### Installation

```bash
# Conda
conda install -c rapidsai -c conda-forge -c nvidia cuml cuda-version=12.0

# Pip
pip install --extra-index-url https://pypi.nvidia.com/ "cuml-cu12"
```

### Konfiguration

```json
{
  "face_clustering": {
    "use_gpu": "auto"
  }
}
```

| Modus | Verhalten |
|------|----------|
| `"auto"` | GPU verwenden, falls cuML verfügbar, Rückfall auf CPU |
| `"always"` | GPU versuchen, warnen und auf CPU zurückfallen, falls nicht verfügbar |
| `"never"` | Immer CPU verwenden |

**Hinweis:** cuML verwendet seine eigene HDBSCAN-Implementierung. Die Parameter `algorithm` und `leaf_size` gelten nur für CPU-Clustering.

### GPU und CPU liefern nicht dieselben Cluster

Gemessen an einer Bibliothek mit 145.677 Gesichtern, gleiche Embeddings und gleiche Parameter:

| Gesichter | cuML-Cluster | CPU-Cluster | Verhältnis |
|----------:|-------------:|------------:|-----------:|
| 5.000 | 202 | 181 | 1,12× |
| 20.000 | 1.285 | 1.039 | 1,24× |

cuML ist etwas feingliedriger, bei vergleichbarer Rauschrate und identischer medianer Clustergröße (4). Der Unterschied ist zu klein, um zu ändern, wer mit wem gruppiert wird — die *Anzahl* der Cluster stimmt zwischen einem GPU- und einem CPU-Lauf jedoch nicht überein.

**cuML ist zudem nicht deterministisch.** Zwei aufeinanderfolgende GPU-Läufe über dieselben 145.677 Gesichter mit identischen Parametern stimmten nur bei **41 % der Labels** überein, erzeugten dabei aber nahezu gleiche Clusterzahlen (14.809 gegenüber 14.807) und identische Endergebnisse. Die Struktur ist stabil, die Beschriftung nicht. Was daraus folgt:

- **Personen-IDs sind zwischen Läufen nicht reproduzierbar.** Gehen Sie nicht davon aus, dass ein erneutes Clustering die Nummerierung automatisch geclusterter Personen beibehält — benennen Sie die Personen, die Ihnen wichtig sind; das macht sie stabil.
- **Vergleichen Sie zwei Clustering-Läufe nicht über die Labels**, sondern über die Zugehörigkeit der Gesichter.

CPU-Clustering mit festem `algorithm` ist für dieselbe Eingabe deterministisch.

### Erneutes Clustern und bereits benannte Personen

Alle Modi außer `--cluster-faces-force` behalten die Personendatensätze, leiten aber sämtliche Gesicht→Person-Zuordnungen von Grund auf neu ab. Geschrieben werden nur die Gesichter, deren Person sich tatsächlich ändert, und die gesamte Aktualisierung ist eine einzige Transaktion; ein unterbrochener Lauf lässt die Datenbank daher unverändert. Ein Gesicht, das Sie im Viewer neu zuordnen, während die Datenbankphase des Laufs schreibt, behält Ihre Zuordnung (der Schreibvorgang ist ein Compare-and-Set); der Lauf protokolliert, wie viele solcher Gesichter er unangetastet gelassen hat. Eine frühere Änderung im selben Lauf, während das Clustering noch rechnet, wird wie bisher überschrieben. Ein neu gebildeter Cluster wird einer Person zurückgegeben, wenn entweder:

1. sein mittleres Embedding innerhalb von `merge_threshold` (Standard 0,6 Kosinus für ArcFace, 0,65 für SFace) zum gespeicherten Zentroid dieser Person liegt, **oder**
2. mindestens die Hälfte seiner Gesichter vor dem Lauf zu dieser Person gehörte.

Die zweite Regel wiegt schwerer, als sie klingt. Das Zentroid einer Person ist ein einzelner Mittelwertvektor; wer über Jahre fotografiert wurde, deckt einen weiten Bereich ab, und HDBSCAN teilt ihn korrekt in enge Untercluster, die weit von diesem Mittelwert entfernt liegen und an Regel 1 scheitern. In der oben genannten Bibliothek lieferte Regel 1 allein 38 % der Gesichter an benannte Personen zurück; mit Regel 2 sind es 96 %.

Da Regel 2 der früheren Zuordnung folgt, wird ein alter Fehler fortgeschrieben statt korrigiert. Das ist der Zweck eines Modus, der bestehende Personen bewahren soll — mit `--cluster-faces-force` wird alles von Grund auf neu berechnet. Verteilen sich die Gesichter eines Clusters genau gleichmäßig auf zwei Personen, entscheidet die Reihenfolge der Gesichts-IDs: Die Person, der das niedrigstnummerierte bereits zugeordnete Gesicht des Clusters gehört, bleibt erhalten. Die Embeddings werden mit `ORDER BY id` geladen; ältere Versionen ließen diese Reihenfolge unbestimmt, sodass die erhaltene Personen-ID bei einem solchen Gleichstand von einem älteren Lauf abweichen kann.

## Blinzelerkennung

Verwendet die Eye Aspect Ratio (EAR) aus den 106-Punkt-Landmarken von InsightFace.

### Funktionsweise

EAR misst das Verhältnis von Augenhöhe zu Augenbreite. Wenn sich die Augen schließen, fällt der EAR unter den Schwellenwert.

### Konfiguration

```json
{
  "face_detection": {
    "blink_ear_threshold": 0.28
  }
}
```

Niedrigerer Schwellenwert = strengere Erkennung (mehr Fotos werden als Blinzler markiert).

### Nach Schwellenwertänderung neu berechnen

```bash
python facet.py --recompute-blinks
```

Verarbeitet nur Fotos mit Gesichtern, keine GPU erforderlich.

## Optionale OFIQ-Gesichtsqualität (kommerzieller Modus)

Im kommerziellen Modus ist das eingeschränkte Gesichtsqualitätsmodell nicht verfügbar. Als optionaler Ersatz installieren Sie `python-ofiq` (UNICEF-Bindings für BSI OFIQ, ISO/IEC 29794-5; MIT; verträgt sich mit numpy 2 und OpenCV 4.8+) mit `pip install python-ofiq`. Es gibt nur Wheels für Linux x86-64 und macOS arm64 (unter Windows: WSL2 oder Docker). Die OFIQ-Modelle (ca. 400 MB, MIT/Apache-2.0) werden nie mitgeliefert: Bei der ersten Nutzung lädt Facet sie mit festgelegter SHA-256 nach `pretrained_models/ofiq/`. Die kommerzielle Gesichts-Pipeline teilt den OFIQ-Gesamtwert (0-100) durch 10 und speichert ihn in `face_quality_iqa`; der Wert eines Fotos ist 0,7 x sein schlechtestes Gesicht + 0,3 x der Mittelwert. Fehlt `python-ofiq`, bleibt die Spalte NULL und ihre Gewichtung wird herausnormiert.

## Ausdruckssignale pro Gesicht (Augen offen + Lächeln)

Jede Gesichtszeile speichert zwei kontinuierliche 0–10-Signale, die vom Gesichts-Panel der Auswahl
und den Aggregaten auf Fotoebene verwendet werden: `eyes_open_score` (10 = weit geöffnet, 0 =
vollständig geschlossen) und `smile_score` (5 = neutral, 10 = breites Lächeln, 0 = missmutiger
Ausdruck).

Zwei Backends erzeugen sie auf derselben 0–10-Skala:

1. **Geometrie (immer verfügbar).** Abgeleitet aus den gespeicherten 106-Punkt-Landmarken von
   InsightFace: Eye Aspect Ratio für Augen offen, Mundwinkel-Anhebung für Lächeln. Reine Geometrie,
   sodass `--recompute-face-signals` sie aus gespeicherten Landmarken ohne Pixel und ohne GPU
   nachträglich berechnen kann.
2. **MediaPipe-Blendshapes (optional, erscheinungsbasiert).** Beim Scannen bzw. bei der
   Gesichtsextraktion wird ein großzügiger Ausschnitt jedes Gesichts durch den MediaPipe Face
   Landmarker geführt, dessen ARKit-artige Blendshapes (`eyeBlink*`, `mouthSmile*`, `mouthFrown*`)
   auf dieselben Skalen abgebildet werden. Erscheinungsbild schlägt Landmark-Geometrie bei
   geschlossenen Augen, subtilen Lächeln und schräg gehaltenen Köpfen, daher **ersetzt** ein per
   MediaPipe bewertetes Gesicht den geometrischen Wert. Fehlt MediaPipe oder sein Modellbündel,
   oder ist der Gesichtsausschnitt zu klein / nicht erkannt, bleibt der geometrische Wert erhalten
   — das Verhalten entspricht dann einer reinen Geometrie-Installation.

### MediaPipe installieren

MediaPipe ist optional und **muss** ohne sein gebündeltes `opencv-contrib-python` installiert
werden, das sonst einen zweiten `cv2`-Namensraum neben Facets `opencv-python` installieren würde:

```bash
pip install mediapipe==0.10.35 --no-deps
pip install absl-py flatbuffers
```

Führen Sie niemals ein einfaches `pip install mediapipe` aus.

### Modellbündel

Das `face_landmarker.task`-Bündel (~3,6 MiB, Apache-2.0) wird bei der ersten Verwendung automatisch
nach `pretrained_models/face_landmarker.task` heruntergeladen. Ist die Maschine offline, laden Sie
es manuell von
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
herunter und legen Sie es an diesem Pfad ab. Ein fehlgeschlagener Download protokolliert einmalig
eine Warnung und fällt auf die geometrischen Scores zurück.

### Konfiguration

```json
{
  "face_detection": {
    "blendshapes": {
      "enabled": true,
      "min_crop_size": 192,
      "blink_pose_gate_deg": null
    }
  }
}
```

- `enabled` (Standard `true`): Blendshape-Scores verwenden, sobald MediaPipe und das Modellbündel
  verfügbar sind; andernfalls läuft automatisch der Geometrie-Fallback. Auf `false` setzen, um
  ausschließlich Geometrie zu erzwingen.
- `min_crop_size` (Standard `192`): Gesichter, deren gepolsterter Ausschnitt kleiner als dieser Wert
  ist (px, kürzere Seite), fallen auf die Geometrie zurück, statt ein winziges Gesicht
  hochzuskalieren.
- `blink_pose_gate_deg` (Standard `null` = aus): nur der MediaPipe-Analyzer des kommerziellen Modus. Ein Gesicht mit |Yaw| oder |Pitch| (Grad, aus der Transformationsmatrix des Face Landmarkers) über diesem Wert erhält kein Blinzel-Urteil und keinen Augen-offen-/Lächeln-Score (NULL = unbekannt), da ein gedrehter Kopf wie geschlossene Augen wirkt. Standardmäßig aus: Auf einer zurückgehaltenen Hälfte einer echten Bibliothek verbesserte kein Grenzwert zwischen 20° und 45° die Blinzel-Erkennung (jeder verlor mehr echte Blinzler als falsche), daher wird keiner mitgeliefert; setze ihn nur, wenn deine Bibliothek viele stark gedrehte Gesichter enthält; gilt für neu gescannte Fotos, vorhandene `is_blink`-Werte ändern sich erst bei einem erneuten Scan. Muss eine positive Gradzahl sein: 0 oder ein negativer Wert wird beim Start abgelehnt; `null` = aus.

### Neuberechnung

`--recompute-face-signals` berechnet die Signale pro Gesicht ausschließlich aus gespeicherten
Landmarken neu — es ist **rein geometrisch** und führt MediaPipe nicht aus (es werden keine Pixel
gelesen). Um die erscheinungsbasierten Scores zu aktualisieren, extrahieren Sie die Gesichter
erneut (`--extract-faces-gpu-force`), damit die Ausschnitte in voller Auflösung neu analysiert
werden.

## Gesichts-Miniaturansichten

Miniaturansichten werden für eine schnelle Anzeige in der Datenbank gespeichert.

### Speicherung

- Während des Scans aus Bildern in voller Auflösung erzeugt
- In der Spalte `faces.face_thumbnail` als JPEG-BLOBs gespeichert (~5-10KB pro Stück)
- Von Clustering und Viewer genutzt, statt sie neu zu erzeugen

### Neuerzeugung

```bash
# Fehlende Vorschaubilder generieren
python facet.py --refill-face-thumbnails-incremental

# ALLE Vorschaubilder neu generieren
python facet.py --refill-face-thumbnails-force
```

Beide Befehle nutzen Parallelverarbeitung für mehr Geschwindigkeit.

## Datenbankschema

### Tabelle faces

| Spalte | Typ | Beschreibung |
|--------|------|-------------|
| `id` | INTEGER | Primärschlüssel |
| `photo_path` | TEXT | Fremdschlüssel zu photos |
| `face_index` | INTEGER | Index innerhalb des Fotos |
| `embedding` | BLOB | 512-dimensionales Gesichts-Embedding |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | INTEGER | Ecken des Begrenzungsrahmens |
| `confidence` | REAL | Erkennungskonfidenz |
| `person_id` | INTEGER | Fremdschlüssel zu persons |
| `face_thumbnail` | BLOB | JPEG-Miniaturansicht |
| `landmark_2d_106` | BLOB | 106-Punkt-Landmarken (Blinzelerkennung) |
| `embedding_model` | TEXT | Tag des Erkennungsmodells (Standard `arcface_buffalo_l`) |

### Tabelle persons

| Spalte | Typ | Beschreibung |
|--------|------|-------------|
| `id` | INTEGER | Primärschlüssel |
| `name` | TEXT | Personenname (NULL = automatisch geclustert) |
| `representative_face_id` | INTEGER | Bestes Gesicht für den Avatar |
| `face_count` | INTEGER | Anzahl der Gesichter |
| `centroid` | BLOB | Zentroid-Embedding des Clusters |
| `auto_clustered` | INTEGER | 1 wenn automatisch erzeugt |
| `face_thumbnail` | BLOB | Avatar-Miniaturansicht der Person |
| `is_hidden` | INTEGER | 1 = von Filtern/Vorschlägen ausgeschlossen |

## Inkrementeller vs. erzwungener Modus

### Inkrementelles Clustering

- Behält alle vorhandenen Personen bei (benannte und automatisch geclusterte)
- Clustert nur neue, nicht zugewiesene Gesichter
- Ordnet neue Cluster vorhandenen Personen über die Zentroid-Ähnlichkeit zu
- Aktualisiert die Zentroide nach dem Zusammenführen

**Verwenden, wenn:** Neue Fotos zu einer vorhandenen Sammlung hinzugefügt werden

### Erzwungenes Clustering

- Löscht ALLE Personen, einschließlich benannter
- Vollständiges Re-Clustering von Grund auf

**Verwenden, wenn:** Neu begonnen wird oder größere Algorithmusänderungen anstehen

### Inkrementell-benanntes Clustering

- Behält nur benannte Personen bei
- Löscht automatisch geclusterte Personen
- Clustert alle unbenannten Gesichter neu

**Verwenden, wenn:** Kuratierte Namen erhalten bleiben sollen, während automatisch erkannte Cluster aktualisiert werden

## Viewer-Integration

### Personenfilter

- Das Dropdown zeigt Personen mit Gesichts-Miniaturansichten
- Galerie nach Person filtern

### Personen-Galerie

- Auf eine Person im Dropdown klicken, um alle ihre Fotos zu sehen
- URL: `/person/<id>`

### Seite „Personen verwalten“

Zugriff über die Header-Schaltfläche oder `/persons`:

- **Rasteransicht** – Alle erkannten Personen
- **Zusammenführen** – Quelle auswählen, Ziel anklicken, bestätigen
- **Stapel-Zusammenführung** – Mehrere Personen auswählen und in ein Ziel zusammenführen
- **Aufteilen** – Ausgewählte Gesichter in eine neue Person verschieben
- **Ausblenden** – Einen Cluster aus der Liste, den Filtern und den Zusammenführungsvorschlägen ausschließen
- **Löschen** – Personen-Cluster entfernen
- **Umbenennen** – Auf den Namen klicken, um ihn inline zu bearbeiten

### Eine Person erstellen

Personen entstehen nicht mehr nur durch Clustering — Sie können ein vom Clusterer übersehenes
Gesicht direkt aus der Galerie heraus benennen:

1. Öffnen Sie auf einer Fotokarte die Personen-Aktionen und wählen Sie ein nicht zugewiesenes
   Gesicht.
2. Wählen Sie im Personen-Auswähler **Neue Person erstellen** und geben Sie einen Namen ein.
3. Das Gesicht wird in einem Aufruf der neuen (manuell erstellten, `auto_clustered = 0`) Person
   zugeordnet.

Endpunkt: `POST /api/persons` (nur mit Bearbeitungsrecht), Body
`{ "name": "<Name>", "face_ids": [<id>, ...] }`. Der Name ist erforderlich (nach dem Trimmen nicht
leer). Gesichter, die bereits einer anderen Person gehören, werden neu zugewiesen, und jede alte
Person, die dadurch ohne Gesichter zurückbleibt, wird gelöscht — dieselbe Logik wie bei der
Gesichtszuweisung. Im Mehrbenutzermodus darf der Aufrufer nur Gesichter aus Fotos innerhalb der
eigenen (oder geteilten) Verzeichnisse anhängen; ein Gesicht außerhalb dieses Bereichs wird als
nicht gefunden abgelehnt.

### Zu benennen

Die Seite „Personen verwalten“ zeigt automatisch geclusterte Personen, die sich zu benennen lohnen,
in einem Abschnitt **Zu benennen**: unbenannte Cluster (`name IS NULL`, `auto_clustered = 1`) mit
mindestens `viewer.persons.needs_naming_min_faces` Gesichtern (Standard `5`), jeweils mit einem
Inline-Namensfeld, damit große Cluster benannt werden können, ohne sie erst suchen zu müssen.
Bereitgestellt über `GET /api/persons/needs_naming?min_faces=N`.

### Seite mit Zusammenführungsvorschlägen

Zugriff über `/merge-suggestions` oder die Schaltfläche „Zusammenführungsvorschläge“ auf der Seite „Personen verwalten“:

- Zeigt Paare von Personen mit ähnlichen Gesichts-Embeddings, die möglicherweise dieselbe Person sind
- **Schieberegler für den Schwellenwert** – steuert die Ähnlichkeitsgrenze (niedriger = mehr Vorschläge)
- **Ein-Klick-Zusammenführung** – ein vorgeschlagenes Paar sofort zusammenführen
- **Stapel-Zusammenführung** – mehrere Vorschläge auswählen und alle auf einmal zusammenführen

### Foto-Karten

- Kleine Gesichts-Miniaturansichten (Avatare) werden für erkannte Personen angezeigt
- Konfigurierbar über `viewer.face_thumbnails.output_size_px`

## Embedding-Raum-Marker (Sicherheit des Erkennungsmodells)

Jede Gesichtszeile trägt einen `embedding_model`-Tag (Spalte in `faces`, Standard
`arcface_buffalo_l` – das aktuelle InsightFace-Erkennungsmodell `buffalo_l` / ArcFace
`w600k_r50`). Embeddings, die von **unterschiedlichen** Erkennungsmodellen erzeugt
werden, leben in **inkompatiblen Vektorräumen** und dürfen niemals gemeinsam
geclustert werden – andernfalls entstehen stillschweigend unbrauchbare Personen
ohne Fehlermeldung.

`FaceClusterer.load_embeddings()` lädt daher nur den **aktiven** Embedding-Raum
(`active_embedding_model()` in `db/embedding_space.py`; ein `NULL`-Tag wird als
veralteter ArcFace-Raum behandelt) und protokolliert eine deutliche Warnung, falls
Gesichter aus einem anderen Raum vorhanden sind und ausgeschlossen werden. Dies ist
ein Schutz für die Vorwärtskompatibilität: Er macht einen zukünftigen Wechsel des
Erkennungsmodells konstruktionsbedingt sicher.

### Wechsel des Erkennungsmodells (z. B. AdaFace) – aufgeschobener Plan

Ein Qualitäts-Upgrade wie **AdaFace** (qualitätsadaptiver Margin, besseres Clustering
von unscharfen/spontanen Gesichtern) lässt sich als optionales 512-dimensionales
Backend integrieren (gleicher Speicherpfad, gleiches HDBSCAN), ist aber **noch nicht
implementiert**, da es ohne echte Daten nicht validiert werden kann. Eine korrekte
Umsetzung erfordert:

1. **Gewichte + Backbone** – einen AdaFace-Checkpoint (z. B. `adaface_ir101_webface12m`)
   plus dessen IResNet-Backbone; ein neuer Download in den Modell-Cache.
2. **Ausgerichtete Crops** – das Embedding aus einem ausgerichteten 112×112-Crop
   `norm_crop(img, face.kps, 112)` zum Extraktionszeitpunkt berechnen (die kps existieren
   auf dem InsightFace-`face`-Objekt, werden aber nicht persistiert, sodass AdaFace nicht
   offline nachgefüllt werden kann – es muss während der Extraktion laufen). Prüfen Sie,
   ob BGR/Normalisierung zum Checkpoint passen.
3. **Konfigurationsschalter** – `face_detection.recognition_model: arcface|adaface`
   hinzufügen und `active_embedding_model()` daraus auflösen; neue Gesichter entsprechend
   taggen.
4. **Vollständige Neu-Extraktion + Re-Clustering** – `--extract-faces-gpu-force`, dann
   `--cluster-faces-force`, da ArcFace- und AdaFace-Embeddings nicht vergleichbar sind.
   Der obige Embedding-Raum-Marker verhindert, dass eine halb migrierte Datenbank die
   beiden Räume stillschweigend zusammen clustert (sie warnt und schließt stattdessen aus).
5. **Qualitätsvalidierung** – die Clusterqualität gegen gelabelte Identitäten messen;
   „läuft und gibt 512-dimensionale Vektoren aus“ beweist nicht, dass die Vorverarbeitung
   korrekt ist.

### Gemischte Bibliotheken, Neu-Extraktion und unbekannte Werte

- **Gemischte Bibliotheken.** Eine Bibliothek kann ArcFace- (512-d) und SFace-Gesichter (128-d) gleichzeitig enthalten. Clusterer, Merge-Vorschläge, die Suche nach ähnlichen Personen und Personen-Zentroide verwenden jeweils nur ein Modell (`persons.centroid_model` hält fest, welches); die inkrementellen und die Nur-benannte-Modi des Clusterings lassen Gesichter und Personen des anderen Modells unberührt, `--cluster-faces-force` dagegen löscht alle Gesicht→Person-Zuordnungen und alle Personen, auch benannte, unabhängig vom Modell. Um eine Bibliothek auf das aktive Modell umzustellen, `--extract-faces-gpu-force` ausführen (im kommerziellen Modus mit dem permissiven Extraktor; benannte Personen bleiben erhalten) und danach `--cluster-faces-incremental`.
- **Warnungen.** Ein Scan protokolliert eine Warnung, wenn gespeicherte Gesichter zu einem anderen als dem aktiven Modell gehören; `--doctor` meldet dieselbe Abweichung gegenüber dem aktiven Modell. Im kommerziellen Modus listet `--doctor` die Komponenten auf, die ein Scan tatsächlich laden würde (YuNet, SFace, Q-ReAlign sowie OFIQ, wenn `python-ofiq` installiert ist), und nicht jene, die das VRAM-Profil lediglich nennt; TOPIQ, LIQE und InsightFace erscheinen daher nicht mehr als Warnungen.
- **Benannte Personen überleben die Neu-Extraktion.** Bevor `--extract-faces-gpu-force` die Gesichtszeilen löscht, merkt sich Facet die Box jedes Gesichts einer benannten Person; danach erhält jedes neu erkannte Gesicht die Person, deren alte Box es mit IoU >= 0,5 überlappt (eins zu eins, beste Überlappung zuerst), und die Zentroide werden pro Modell neu berechnet. Ein Gesicht ohne passende alte Box bleibt für den nächsten Clustering-Lauf unzugeordnet. Die gemerkten Boxen stehen in der Tabelle `face_reextract_snapshot`, geschrieben in derselben Transaktion wie das Löschen; ein unterbrochener Lauf wird daher von der nächsten Extraktion nachgeholt. Das erneute Scannen eines Fotos erhält benannte Personen auf dieselbe Weise. Personen-Zentroide, die ein erneuter Scan veraltet zurückgelassen hat, werden einmal am Ende des Scans neu aufgebaut (auch wenn dieser unterbrochen wird), nicht einmal pro Batch; Zähler und repräsentative Gesichter werden weiterhin pro Batch aktualisiert.
- **Unbekannt ist nicht null.** Ohne MediaPipe kann der permissive Analysator Augen, Lächeln und Blinzeln nicht messen; diese Werte werden als NULL gespeichert (nicht 0 und nicht „blinzelt“). Blink-Filter und Blink-Abzug behandeln NULL als „kein Blinzeln“. Die Leser der 106-Punkte-Landmarken überspringen Gesichter ohne Landmarken. Ein Foto, dessen Gesichtsdurchlauf fehlschlägt, wird mit `face_count`, `face_ratio` und `is_group_portrait` als NULL (unbekannt) gespeichert, nicht als 0, und seine gespeicherten Gesichter bleiben erhalten; die Bewertung behandelt es als ohne Gesicht, bis es erneut gescannt wird.

## Fehlerbehebung

| Problem | Lösung |
|-------|----------|
| Clustering hängt | Algorithmus `boruvka_balltree` verwenden |
| Zu viele kleine Cluster | `min_faces_per_person` erhöhen |
| Gesichter werden nicht gruppiert | `merge_threshold` verringern |
| GPU-Clustering schlägt fehl | cuML-Installation prüfen, `"never"` verwenden, um CPU zu erzwingen |
| Miniaturansichten fehlen | `--refill-face-thumbnails-incremental` ausführen |
| Falsche Blinzelerkennung | `blink_ear_threshold` anpassen, `--recompute-blinks` ausführen |
| Warnung „Excluded N faces from non-active embedding space“ | Eine Änderung des Erkennungsmodells hat gemischte Embeddings hinterlassen – `--extract-faces-gpu-force`, dann `--cluster-faces-incremental` ausführen |
