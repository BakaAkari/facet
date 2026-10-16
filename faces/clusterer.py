"""
Face clustering for Facet.

HDBSCAN-based clustering of face embeddings into persons.
"""

import sqlite3
from collections import Counter
from db import get_connection
import numpy as np
import time

from db.embedding_space import (
    ARCFACE_MODEL, active_embedding_model, expected_dim, model_filter_sql, model_mismatch_sql, tag_matches,
)
from db.person_carry_over import (
    persist_named_snapshot, repair_preserved_representatives, replay_persisted_snapshot,
)
from db.person_centroids import recompute_person_centroid
from utils import bytes_to_embedding, load_image_for_face_crop, crop_face_with_padding
from faces.processor import FaceProcessor

import logging

logger = logging.getLogger("facet.face_cluster")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        if desc:
            logger.info("%s...", desc)
        return iterable

def _check_cuml_available():
    """Check if cuML is available for GPU clustering.

    Returns:
        True if cuML and cupy are importable, False otherwise.
    """
    try:
        from cuml.cluster import HDBSCAN as cumlHDBSCAN  # noqa: F401
        import cupy as cp
        # cuML/cupy can import on a host with no usable GPU — e.g. the CPU-only
        # "legacy" profile running the same (GPU-capable) image. Require an actual
        # CUDA device so clustering falls back to CPU HDBSCAN instead of crashing
        # at the first GPU op (the GPU path has no other guard).
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        return True
    except Exception:
        return False


class FaceClusterer:
    """HDBSCAN-based clustering of face embeddings into persons."""

    def __init__(self, db_path, min_faces=2, min_samples=None, auto_merge_distance=0.15,
                 algorithm='boruvka_balltree', leaf_size=40, use_gpu='auto',
                 merge_threshold=0.6,
                 use_db_thumbnails=True,
                 face_thumbnail_size=128, face_thumbnail_quality=85,
                 embedding_model=None):
        """
        Initialize the face clusterer.

        Args:
            db_path: Path to SQLite database
            min_faces: Minimum number of faces required to form a person cluster (HDBSCAN min_cluster_size)
            min_samples: Core point density parameter - lower values create more clusters with fewer
                        noise points. Defaults to min(min_faces, 2) if not specified.
            auto_merge_distance: Merge clusters within this cosine distance (0 = disabled)
            algorithm: HDBSCAN algorithm - 'boruvka_balltree' (recommended for 512-dim embeddings),
                      'boruvka_kdtree', 'prims_balltree', 'prims_kdtree', or 'best'
            leaf_size: Leaf size for tree algorithms (20-100 typical, default 40)
            use_gpu: GPU clustering mode - 'auto' (use if available), 'always', or 'never'
            use_db_thumbnails: If True, use cached thumbnails from database instead of loading
                              original images when generating fallback thumbnails (faster for RAW/NFS)
            face_thumbnail_size: Size for face thumbnails (default 128)
            face_thumbnail_quality: JPEG quality for face thumbnails (default 85)
            embedding_model: Recognition model whose embeddings are clustered
                            (default: the active model). Faces of any other model
                            are never loaded, compared or reassigned.
        """
        self.db_path = db_path
        self.embedding_model = embedding_model or active_embedding_model()
        self.embedding_dim = expected_dim(self.embedding_model)
        self.min_faces = min_faces
        # min_samples controls core point density; lower = more clusters, fewer noise points
        # Default to min(min_faces, 2) for less conservative clustering
        self.min_samples = min_samples if min_samples is not None else min(min_faces, 2)
        # cluster_selection_epsilon: merge clusters within this distance
        self.cluster_selection_epsilon = auto_merge_distance if auto_merge_distance > 0 else None
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.use_gpu = use_gpu
        self.merge_threshold = merge_threshold
        self.use_db_thumbnails = use_db_thumbnails
        self.face_thumbnail_size = face_thumbnail_size
        self.face_thumbnail_quality = face_thumbnail_quality
        self._cuml_available = None  # Lazy check

    def _should_use_gpu(self):
        """Determine if GPU clustering should be used.

        Returns:
            True if GPU should be used for clustering, False otherwise.
        """
        if self.use_gpu == 'never':
            return False

        # Lazy check for cuML availability
        if self._cuml_available is None:
            self._cuml_available = _check_cuml_available()

        if self.use_gpu == 'always':
            if not self._cuml_available:
                logger.warning("Warning: use_gpu='always' but cuML not available, falling back to CPU")
            return self._cuml_available

        # 'auto' mode - use GPU if available
        return self._cuml_available

    def load_embeddings(self):
        """Load face embeddings for this clusterer's recognition space only.

        Faces are tagged with ``embedding_model``. Embeddings
        produced by a different recognition model live in an incompatible vector
        space; clustering them together silently corrupts persons. We therefore
        load only the active space (treating a NULL tag as the legacy ArcFace
        space) and loudly warn if faces from other spaces are excluded.
        """
        face_ids = []
        embeddings = []

        with get_connection(self.db_path, row_factory=False) as conn:
            match_sql, match_params = model_filter_sql(self.embedding_model)
            # ORDER BY id: cluster discovery order -- and with it the ids of newly
            # created persons and the float summation order of their centroids --
            # follows load order. The rowid scan is already in id order (no sort),
            # so this costs nothing today and pins the order against any index
            # added later that would serve the scan in another one.
            cursor = conn.execute(f"""
                SELECT id, embedding FROM faces
                WHERE embedding IS NOT NULL AND {match_sql}
                ORDER BY id
            """, match_params)

            for row in cursor.fetchall():
                face_id, embedding_bytes = row
                if embedding_bytes:
                    # Convert bytes to numpy array using shared utility
                    embedding = bytes_to_embedding(embedding_bytes, dim=self.embedding_dim)
                    if embedding is not None:
                        face_ids.append(face_id)
                        embeddings.append(embedding)

            # Guard: warn if faces from a different (incompatible) embedding
            # space exist — they were excluded to avoid mixing vector spaces.
            other_sql, other_params = model_mismatch_sql(self.embedding_model)
            other = conn.execute(f"""
                SELECT COALESCE(embedding_model, ?), COUNT(*) FROM faces
                WHERE embedding IS NOT NULL AND {other_sql}
                GROUP BY 1
            """, [ARCFACE_MODEL] + other_params).fetchall()
            if other:
                excluded = sum(count for _, count in other)
                spaces = ", ".join(f"{model}({count})" for model, count in other)
                logger.warning(
                    "Excluded %d face(s) from non-active embedding space(s) [%s]; "
                    "clustering only the active space '%s' to avoid mixing "
                    "incompatible embeddings. Re-extract all faces after a "
                    "recognition-model change.",
                    excluded, spaces, self.embedding_model,
                )

        return face_ids, np.array(embeddings) if embeddings else np.array([])

    def cluster_faces(self, force=False, preserve_named_only=False):
        """
        Run HDBSCAN clustering on face embeddings and update database.

        Args:
            force: If True, delete all persons including named ones.
                   If False (default), preserve all existing persons and match new faces to them.
            preserve_named_only: If True, only preserve named persons, delete unnamed ones.
        """
        import hdbscan  # Use standalone library, not sklearn (sklearn has epsilon bug)

        if force:
            logger.info("Force mode: will delete all persons including named ones")
        elif preserve_named_only:
            logger.info("Named-only mode: will preserve only named persons, re-cluster unnamed faces")
        else:
            logger.info("Incremental mode: will preserve all existing persons")

        logger.info("Step 1/4: Loading embeddings from database...")
        t0 = time.time()
        face_ids, embeddings = self.load_embeddings()
        t1 = time.time()
        logger.info("Step 1/4: Done (%.1fs) - loaded %s faces", t1-t0, len(embeddings))

        if len(embeddings) < self.min_faces:
            logger.info("Not enough faces to cluster (%s found, need at least %s)", len(embeddings), self.min_faces)
            return

        logger.info("Step 2/4: Normalizing embeddings...")
        # Normalize embeddings for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-10)
        t2 = time.time()
        logger.info("Step 2/4: Done (%.1fs)", t2-t1)

        # Convert cosine epsilon to euclidean for normalized vectors: d_euclidean = sqrt(2 * d_cosine)
        euclidean_epsilon = 0.0
        if self.cluster_selection_epsilon:
            euclidean_epsilon = float(np.sqrt(2 * self.cluster_selection_epsilon))

        labels = None
        if self._should_use_gpu():
            # Use cuML GPU clustering, falling back to CPU on any GPU failure so a
            # transient CUDA/cuML problem degrades gracefully instead of aborting.
            try:
                from cuml.cluster import HDBSCAN as cumlHDBSCAN
                import cupy as cp

                logger.info("Step 3/4: Computing clusters on GPU (cuML HDBSCAN, min_cluster_size=%s, min_samples=%s)...", self.min_faces, self.min_samples)
                embeddings_gpu = cp.asarray(embeddings_normalized)
                clusterer = cumlHDBSCAN(
                    min_cluster_size=self.min_faces,
                    min_samples=self.min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=euclidean_epsilon,
                )
                labels = cp.asnumpy(clusterer.fit_predict(embeddings_gpu))  # Back to numpy
            except Exception as e:
                logger.warning("GPU clustering failed (%s); falling back to CPU HDBSCAN", e)
                labels = None
        if labels is None:
            # Use CPU clustering (standalone hdbscan library)
            logger.info("Step 3/4: Computing clusters on CPU (algorithm=%s, leaf_size=%s, min_cluster_size=%s, min_samples=%s)...", self.algorithm, self.leaf_size, self.min_faces, self.min_samples)
            # Run HDBSCAN with cluster_selection_epsilon (works in standalone hdbscan library)
            # core_dist_n_jobs=-1 enables parallel distance computation using all CPU cores
            # algorithm='boruvka_balltree' is O(n log n) vs O(n²) for exact methods - critical for large datasets
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_faces,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_epsilon=euclidean_epsilon,
                core_dist_n_jobs=-1,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
            )
            labels = clusterer.fit_predict(embeddings_normalized)
        t3 = time.time()
        logger.info("Step 3/4: Done (%.1fs)", t3-t2)

        # Build face_id -> cluster_label mapping
        face_to_cluster = dict(zip(face_ids, labels))

        logger.info("Step 4/4: Updating database with cluster assignments...")
        # Update database
        self._update_database(face_to_cluster, embeddings_normalized, face_ids,
                             force=force, preserve_named_only=preserve_named_only)
        t4 = time.time()
        logger.info("Step 4/4: Done (%.1fs)", t4-t3)

        # Print summary
        unique_labels = set(labels)
        n_clusters = len([lbl for lbl in unique_labels if lbl >= 0])
        n_noise = list(labels).count(-1)
        logger.info("\nClustering complete in %.1fs total", t4-t0)
        logger.info("Found %s person clusters, %s unclustered faces", n_clusters, n_noise)

    def _generate_face_thumbnail(self, conn, face_id):
        """Generate a cropped face thumbnail for a representative face.

        Uses full RAW processing for higher quality thumbnails.
        Uses shared crop_face_with_padding() for consistency with FaceAnalyzer.
        Uses self.face_thumbnail_size and self.face_thumbnail_quality from config.
        """
        # Get face bbox and photo path
        face = conn.execute("""
            SELECT f.photo_path, f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2
            FROM faces f WHERE f.id = ?
        """, (face_id,)).fetchone()

        if not face:
            return None

        photo_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2 = face
        if bbox_x1 is None:
            return None

        try:
            img_cv = load_image_for_face_crop(photo_path)
            if img_cv is None:
                return None

            # Use shared crop_face_with_padding (same as FaceAnalyzer)
            return crop_face_with_padding(
                img_cv,
                [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                padding=0.3,
                size=self.face_thumbnail_size,
                quality=self.face_thumbnail_quality,
                use_cv2=True
            )

        except Exception as e:
            logger.error("Error generating face thumbnail for face %s: %s", face_id, e)
            return None

    def _load_existing_persons(self, conn, force, preserve_named_only):
        """Load existing persons from the database for matching during clustering.

        Args:
            conn: SQLite connection
            force: If True, returns empty dict (all persons will be deleted)
            preserve_named_only: If True, only loads named persons

        Returns:
            Dict mapping person_id to {'name': str, 'centroid': np.array | None}.
            A person with no stored centroid in this run's embedding space
            (API-created via create/split/assign, never re-clustered, or built
            from another model) is kept with ``centroid=None`` — it is
            still preserved, but only through the previous-owner history rule, not
            centroid similarity. Excluding these (the old ``centroid IS NOT NULL``
            filter) dropped them from preservation, so every incremental run cleared
            their faces' person_id and the cleanup then deleted the named person.
        """
        if force:
            return {}

        if preserve_named_only:
            cursor = conn.execute("""
                SELECT id, name, centroid, centroid_model FROM persons WHERE name IS NOT NULL
            """)
        else:
            cursor = conn.execute("SELECT id, name, centroid, centroid_model FROM persons")

        existing_persons = {}
        for row in cursor.fetchall():
            person_id, name, centroid_bytes, centroid_model = row
            centroid = None
            # A centroid from another model (or of another length) is in a space
            # this run's embeddings do not live in: the person is still kept, but
            # only through the previous-owner history rule.
            if centroid_bytes and tag_matches(centroid_model, self.embedding_model):
                centroid = np.frombuffer(centroid_bytes, dtype=np.float32)
                if len(centroid) == self.embedding_dim:
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
                else:
                    centroid = None
            existing_persons[person_id] = {'name': name, 'centroid': centroid}

        if existing_persons:
            logger.info("  Preserving %s existing person(s)", len(existing_persons))
        return existing_persons

    def _preserved_face_assignments(self, conn, existing_persons):
        """Capture the current face->person map for faces of preserved persons.

        Incremental and named-only clustering clear every face's ``person_id``
        before rebuilding clusters. Faces that HDBSCAN relabels as noise (-1)
        this run are never placed in any cluster, so without this snapshot they
        would be silently orphaned from a person that is otherwise being kept.
        """
        if not existing_persons:
            return {}
        preserved_ids = set(existing_persons.keys())
        match_sql, match_params = model_filter_sql(self.embedding_model)
        rows = conn.execute(
            f"SELECT id, person_id FROM faces WHERE person_id IS NOT NULL AND {match_sql}",
            match_params,
        ).fetchall()
        return {face_id: person_id for face_id, person_id in rows if person_id in preserved_ids}

    # Share of a cluster that must have belonged to one person before this run
    # for the cluster to be handed back to them. A simple majority: below it the
    # cluster is genuinely mixed and the centroid rule's answer is the better one.
    PRESERVED_MAJORITY = 0.5

    def _previous_owner_by_majority(self, cluster_face_ids, preserved_assignments):
        """The person most of this cluster belonged to before the run, or None.

        Only answers when a single person holds at least ``PRESERVED_MAJORITY``
        of the cluster's faces, so a cluster that genuinely spans two people is
        left to the centroid rule rather than being assigned to whichever of
        them happened to contribute one face more.
        """
        owners = Counter(
            pid for pid in (preserved_assignments.get(fid) for fid in cluster_face_ids)
            if pid is not None
        )
        if not owners:
            return None
        person_id, count = owners.most_common(1)[0]
        return person_id if count >= self.PRESERVED_MAJORITY * len(cluster_face_ids) else None

    def _restore_noise_faces(self, new_owner, face_to_cluster, preserved_assignments):
        """Re-attach noise (label -1) faces to their previously preserved person.

        Records the owners in ``new_owner`` (face id -> person id); the database
        is written once, afterwards, by ``_apply_owner_changes``.

        Returns the number of faces restored.
        """
        if not preserved_assignments:
            return 0
        restored = 0
        for face_id, label in face_to_cluster.items():
            if label < 0:
                person_id = preserved_assignments.get(face_id)
                if person_id is not None:
                    new_owner[face_id] = person_id
                    restored += 1
        return restored

    def _apply_owner_changes(self, conn, changes):
        """Write ``changes`` -- ``(new_owner, face_id, old_owner)`` -- as a compare-and-set.

        A face is updated only while it still has the owner this run READ for it
        (``person_id IS ?``, so a NULL old owner matches NULL). A viewer merge or
        assign committed since the read therefore wins: that face is left as the
        viewer set it and counted as a conflict. Nothing here holds a lock from
        before the read; the write lock is taken by this statement and released
        by the caller's commit.

        Returns the number of conflicts (changes that matched no row).
        """
        if not changes:
            return 0
        cursor = conn.executemany(
            "UPDATE faces SET person_id = ? WHERE id = ? AND person_id IS ?", changes)
        # For executemany, rowcount is the sum over the statements.
        conflicts = len(changes) - cursor.rowcount
        if conflicts:
            logger.warning(
                "  %s face owner change(s) skipped: the face's owner changed after this run "
                "read it (conflict with a concurrent edit); the newer owner was kept", conflicts)
        return conflicts

    def _update_person_centroids(self, conn, person_ids):
        """Recompute face counts and centroids for existing persons.

        Args:
            conn: SQLite connection
            person_ids: Iterable of person IDs to update
        """
        for pid in person_ids:
            conn.execute("""
                UPDATE persons SET face_count = (
                    SELECT COUNT(*) FROM faces WHERE person_id = ?
                ) WHERE id = ?
            """, (pid, pid))
            # A person with no face in this run's space keeps its older-space
            # centroid (tagged with its own model); never a mixed average.
            recompute_person_centroid(conn, pid, self.embedding_model)

    def _update_database(self, face_to_cluster, embeddings, face_ids, force=False, preserve_named_only=False):
        """
        Create person records and assign faces to persons.

        Args:
            face_to_cluster: Dict mapping face_id to cluster label
            embeddings: Normalized embeddings array
            face_ids: List of face IDs corresponding to embeddings
            force: If True, delete all persons. If False, preserve all existing persons.
            preserve_named_only: If True, only preserve named persons, delete unnamed ones.
        """
        # Threshold for matching new cluster to existing person (cosine similarity)
        merge_threshold = self.merge_threshold

        with get_connection(self.db_path, row_factory=False) as conn:
            # Ensure face_thumbnail column exists (for older databases)
            try:
                conn.execute("ALTER TABLE persons ADD COLUMN face_thumbnail BLOB")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Load existing persons based on mode
            existing_persons = self._load_existing_persons(conn, force, preserve_named_only)
            preserved_assignments = self._preserved_face_assignments(conn, existing_persons)

            # Only this run's embedding space is re-matched: a face (or a person
            # built from a face) of another model is not in `embeddings`, so
            # clearing it here would orphan it with no way to be reassigned.
            face_match_sql, face_match_params = model_filter_sql(self.embedding_model)
            person_match_sql, person_match_params = model_filter_sql(
                self.embedding_model, 'centroid_model')
            # Who owns each face of the re-matched space BEFORE this run (faces with
            # no owner are simply absent). The phase used to blanket-NULL these and
            # rewrite each one; it now builds the wanted owner map in memory
            # (`new_owner`) and writes only the difference at the end. A face absent
            # from `new_owner` is left with no owner, as the blanket reset left it.
            if force:
                old_owner = dict(conn.execute(
                    "SELECT id, person_id FROM faces WHERE person_id IS NOT NULL").fetchall())
            else:
                old_owner = dict(conn.execute(
                    f"SELECT id, person_id FROM faces WHERE person_id IS NOT NULL AND {face_match_sql}",
                    face_match_params).fetchall())
            new_owner = {}

            if force:
                # Full reset - delete everything
                conn.execute("DELETE FROM persons")
            elif preserve_named_only:
                # Delete unnamed persons; their faces are re-matched below
                deleted = conn.execute(
                    f"DELETE FROM persons WHERE name IS NULL AND {person_match_sql}",
                    person_match_params).rowcount
                logger.info("  Deleted %s unnamed auto-clustered person(s)", deleted)

            # Group faces by cluster label
            clusters = {}
            for face_id, label in face_to_cluster.items():
                if label >= 0:  # Skip noise points (label = -1)
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(face_id)

            # Track how many faces were merged into named persons
            merged_to_named = 0
            # ...and how many of those the centroid rule would have dropped
            restored_by_history = 0

            cluster_items = list(clusters.items())
            logger.info("  Processing %s clusters...", len(cluster_items))
            thumbnails_to_generate = []  # Defer thumbnail generation

            # Row position of every face id, built once. This loop visits every
            # clustered face, and `face_ids` is a list: a membership test and an
            # .index() call against it are each O(n), so resolving positions
            # inline made assignment O(n^2) -- on a 145,677-face library that is
            # ~4e10 comparisons, minutes to hours spent after the clustering
            # itself had finished, on CPU and GPU alike.
            index_of = {fid: i for i, fid in enumerate(face_ids)}

            # The preserved persons that can be matched by similarity, stacked once.
            # Centroid-less preserved persons (API-created) can only be matched by
            # the history rule below, not by similarity.
            match_pids = [pid for pid, pdata in existing_persons.items()
                          if pdata['centroid'] is not None]
            match_matrix = (np.stack([existing_persons[pid]['centroid'] for pid in match_pids])
                            if match_pids else None)

            # Create person records for each cluster
            for label, cluster_face_ids in tqdm(cluster_items, desc="  Assigning clusters"):
                # Get embeddings for this cluster
                cluster_indices = [index_of[fid] for fid in cluster_face_ids if fid in index_of]
                if not cluster_indices:
                    continue

                cluster_embeddings = embeddings[cluster_indices]

                # Compute centroid (average embedding)
                centroid = np.mean(cluster_embeddings, axis=0).astype(np.float32)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
                centroid_bytes = centroid.tobytes()

                # Check if this cluster matches an existing person
                matched_person_id = None
                if match_matrix is not None:
                    # One matrix-vector product instead of a dot per person. HEAD's
                    # scan kept the FIRST strict maximum above the threshold, which
                    # is argmax (first maximum) plus the threshold test. The test
                    # compares as a Python float: a float32 against the float64
                    # threshold would round the threshold and flip a near-tie.
                    sims = match_matrix @ centroid
                    best = int(np.argmax(sims))
                    if float(sims[best]) > merge_threshold:
                        matched_person_id = match_pids[best]

                # Centroid similarity alone loses the people it should protect
                # most. A person is represented by one averaged vector, but
                # someone with thousands of faces spans years, ages and lighting
                # -- HDBSCAN correctly splits them into many tight sub-clusters,
                # each sitting far from that global mean, so each fails the
                # threshold and becomes a new anonymous person. Measured on a
                # 145,677-face library: named people kept 48% of their faces,
                # and the largest kept 4-19% while small, visually consistent
                # ones kept over 90%.
                #
                # The answer is already in hand. `preserved_assignments` records
                # who owned every face before this run; `_restore_noise_faces`
                # consults it for faces HDBSCAN called noise, and this is the
                # same question asked of a cluster: if most of it belonged to
                # one person before, it still does, whatever the averaged
                # centroid says. Recovers 95% of what the centroid rule dropped.
                #
                # It defers to the previous assignment, so an earlier mistake is
                # carried forward rather than corrected -- which is the contract
                # of a mode whose whole promise is preserving existing persons.
                # `--cluster-faces-force` remains the way to re-derive from
                # scratch.
                if matched_person_id is None and preserved_assignments:
                    matched_person_id = self._previous_owner_by_majority(
                        cluster_face_ids, preserved_assignments)
                    if matched_person_id is not None:
                        restored_by_history += len(cluster_face_ids)

                if matched_person_id is not None:
                    # Merge faces into existing person
                    for face_id in cluster_face_ids:
                        new_owner[face_id] = matched_person_id
                    merged_to_named += len(cluster_face_ids)
                else:
                    # Create new auto-clustered person
                    # Find representative face (closest to centroid)
                    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                    best_idx = np.argmin(distances)
                    representative_face_id = cluster_face_ids[best_idx]

                    # Get stored face thumbnail from the representative face
                    thumb_row = conn.execute("""
                        SELECT face_thumbnail FROM faces WHERE id = ?
                    """, (representative_face_id,)).fetchone()
                    stored_thumbnail = thumb_row[0] if thumb_row and thumb_row[0] else None

                    if stored_thumbnail:
                        # Use pre-generated thumbnail from detection time
                        cursor = conn.execute("""
                            INSERT INTO persons (name, representative_face_id, face_count, centroid, centroid_model, auto_clustered, face_thumbnail)
                            VALUES (NULL, ?, ?, ?, ?, 1, ?)
                        """, (representative_face_id, len(cluster_face_ids), centroid_bytes,
                              self.embedding_model, stored_thumbnail))
                    else:
                        # Defer thumbnail generation for faces without stored thumbnails (legacy data)
                        thumbnails_to_generate.append(representative_face_id)
                        cursor = conn.execute("""
                            INSERT INTO persons (name, representative_face_id, face_count, centroid, centroid_model, auto_clustered, face_thumbnail)
                            VALUES (NULL, ?, ?, ?, ?, 1, NULL)
                        """, (representative_face_id, len(cluster_face_ids), centroid_bytes,
                              self.embedding_model))
                    person_id = cursor.lastrowid

                    # Faces get person_id when the difference is written below
                    for face_id in cluster_face_ids:
                        new_owner[face_id] = person_id

            restored_noise = self._restore_noise_faces(new_owner, face_to_cluster, preserved_assignments)
            if restored_noise:
                logger.info("  Restored %s face(s) reclassified as noise to their preserved person(s)", restored_noise)

            # Write only the faces whose owner changed, compare-and-set. This must
            # land before the centroid/count update and the representative repair,
            # which read `faces.person_id`.
            all_ids = old_owner.keys() | new_owner.keys()
            changes = [(new_owner.get(fid), fid, old_owner.get(fid))
                       for fid in sorted(all_ids) if old_owner.get(fid) != new_owner.get(fid)]
            self._apply_owner_changes(conn, changes)
            logger.info("  Face assignment: %s of %s faces changed owner", len(changes), len(all_ids))

            # Update face counts and centroids for existing persons
            if existing_persons:
                self._update_person_centroids(conn, existing_persons.keys())
                # Clustering reassigned person_id on every face, so a preserved
                # person's stored representative may now belong to someone else.
                repair_preserved_representatives(conn, existing_persons.keys())
                # An incremental run can leave a preserved *unnamed* auto-clustered
                # person with no faces (all folded into other clusters). These husks
                # accumulate run over run, so drop them -- but keep named persons and
                # API-created (auto_clustered = 0) ones even when momentarily empty.
                deleted_empty = conn.execute(
                    "DELETE FROM persons WHERE name IS NULL AND auto_clustered = 1 "
                    "AND (face_count = 0 OR face_count IS NULL)"
                ).rowcount
                if deleted_empty:
                    logger.info("  Removed %s empty auto-clustered person(s)", deleted_empty)

            # ONE commit for the whole phase: the person inserts, the owner write,
            # the centroid/count update and the husk delete land together. The
            # connection closes without committing on an exception, so an interrupt
            # anywhere above changes nothing.
            conn.commit()

            if merged_to_named > 0:
                logger.info("  Merged %s face(s) into existing persons", merged_to_named)
            if restored_by_history > 0:
                logger.info("  %s of those were handed back on their previous assignment, "
                            "which the centroid rule alone would have orphaned",
                            restored_by_history)

            # Legacy faces without a stored thumbnail: decode the ORIGINAL photo
            # (RAW included, read-only) with no transaction open, then write each
            # batch in a short one. An interrupt here loses only thumbnails, which
            # the next run regenerates.
            if thumbnails_to_generate:
                self._write_legacy_thumbnails(conn, thumbnails_to_generate)

    def _write_legacy_thumbnails(self, conn, representative_face_ids):
        """Generate and store thumbnails for new persons whose representative face has none.

        ``conn`` must have no open transaction. Each 10% batch is decoded first,
        with the connection idle, and only then written and committed, so the write
        lock is never held across the read of an original file.
        """
        logger.info("  Generating %s thumbnails (legacy faces without stored thumbnails)...",
                    len(representative_face_ids))
        batch_size = max(1, len(representative_face_ids) // 10)
        decoded = []
        for j, face_id in enumerate(tqdm(representative_face_ids, desc="  Thumbnails")):
            thumbnail = self._generate_face_thumbnail(conn, face_id)
            if thumbnail:
                decoded.append((thumbnail, face_id))
            if (j + 1) % batch_size == 0 or j + 1 == len(representative_face_ids):
                if decoded:
                    conn.executemany(
                        "UPDATE persons SET face_thumbnail = ? WHERE representative_face_id = ?",
                        decoded)
                    conn.commit()
                    decoded = []

    def match_face_to_person(self, embedding_bytes, threshold=None):
        """
        Match a new face embedding against existing person centroids.

        Args:
            embedding_bytes: face embedding (this clusterer's model) as bytes
            threshold: Minimum cosine similarity to match

        Returns:
            person_id if match found, None otherwise
        """
        if threshold is None:
            threshold = self.merge_threshold
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        if len(embedding) != self.embedding_dim:
            return None

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        best_person_id = None
        best_similarity = threshold

        with get_connection(self.db_path, row_factory=False) as conn:
            person_match_sql, person_match_params = model_filter_sql(
                self.embedding_model, 'centroid_model')
            cursor = conn.execute(f"""
                SELECT id, centroid FROM persons
                WHERE centroid IS NOT NULL AND {person_match_sql}
            """, person_match_params)

            for row in cursor.fetchall():
                person_id, centroid_bytes = row
                if centroid_bytes:
                    centroid = np.frombuffer(centroid_bytes, dtype=np.float32)
                    if len(centroid) != self.embedding_dim:
                        continue
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

                    # Cosine similarity
                    similarity = np.dot(embedding, centroid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_person_id = person_id

        return best_person_id


def extract_faces_from_existing(scorer, force=False):
    """
    Extract face embeddings from photos in the database.

    Args:
        scorer: Facet instance with initialized face analyzer
        force: If True, delete all existing faces first and re-extract all.
               If False (default), only extract for photos not yet in faces table.
    """
    # Get face processing settings
    face_processing = scorer.config.get_face_processing_settings()
    num_workers = face_processing.get('extract_workers', 2)
    batch_size = face_processing.get('extract_batch_size', 16)
    crop_padding = face_processing.get('crop_padding', 0.3)

    if force:
        # Before anything is deleted: an analyzer whose models cannot load would
        # otherwise leave the library with no faces at all.
        ensure_models = getattr(scorer.face_analyzer, 'ensure_models', None)
        if ensure_models is not None and not ensure_models():
            raise RuntimeError(
                "The face models could not be loaded, so nothing was deleted -- the "
                "first run downloads them into pretrained_models/, so check the network "
                "connection, or this OpenCV build lacks FaceDetectorYN/FaceRecognizerSF.")

    with get_connection(scorer.db_path, row_factory=False) as conn:
        if force:
            # Named persons outlive the delete only if we remember which boxes
            # were theirs. The memory is a table written in the SAME transaction
            # as the delete: the extraction below runs for hours, and a crash in
            # it must not orphan the named persons for good.
            remembered = persist_named_snapshot(conn)
            deleted = conn.execute("DELETE FROM faces").rowcount
            conn.commit()
            if deleted > 0:
                logger.info("Deleted %s existing face records (%d named-person faces remembered).",
                            deleted, remembered)

            # Get ALL photos
            cursor = conn.execute("SELECT path FROM photos")
        else:
            # Only photos without existing face records
            cursor = conn.execute("""
                SELECT path FROM photos
                WHERE path NOT IN (SELECT DISTINCT photo_path FROM faces)
            """)
        photos = [row[0] for row in cursor.fetchall()]

    if photos:
        logger.info("Extracting faces from %s photos...", len(photos))
        logger.info("Workers: %s, Batch size: %s", num_workers, batch_size)

        processor = FaceProcessor(
            db_path=scorer.db_path,
            config=face_processing,
            num_workers=num_workers,
            batch_size=batch_size,
            mode='extract',
            face_analyzer=scorer.face_analyzer,
            crop_padding=crop_padding
        )

        processor.run(photos, desc="Extracting faces")
    else:
        logger.info("No photos in database.")

    # Also replays what an earlier run that died after its delete left behind.
    with get_connection(scorer.db_path, row_factory=False) as conn:
        stats = replay_persisted_snapshot(conn)
    if stats:
        logger.info("Re-attached %d face(s) to named persons; %d previously named face(s) "
                    "found no matching detection (re-run clustering to place them).",
                    stats['reattached'], stats['unmatched'])


def refill_face_thumbnails(db_path, config, force=False):
    """
    Regenerate face thumbnails from original images.

    Args:
        db_path: Path to SQLite database
        config: ScoringConfig instance
        force: If True, clear all thumbnails and regenerate all.
               If False (default), only generate for faces missing thumbnails.
    """
    face_processing = config.get_face_processing_settings()
    thumbnail_size = face_processing.get('face_thumbnail_size', 128)
    thumbnail_quality = face_processing.get('face_thumbnail_quality', 85)
    crop_padding = face_processing.get('crop_padding', 0.3)
    num_workers = face_processing.get('refill_workers', 4)
    batch_size = face_processing.get('refill_batch_size', 100)

    with get_connection(db_path, row_factory=False) as conn:
        if force:
            # Clear all existing thumbnails
            cleared = conn.execute("UPDATE faces SET face_thumbnail = NULL").rowcount
            conn.commit()
            if cleared > 0:
                logger.info("Cleared %s existing face thumbnails.", cleared)

        # Get faces with bbox data (all if force, only missing thumbnails otherwise)
        # Only ArcFace rows are expected to carry landmarks; an SFace row has none
        # by design and must not be re-queued on every refill because of it.
        arcface_sql, arcface_params = model_filter_sql(ARCFACE_MODEL)
        cursor = conn.execute("""
            SELECT id, photo_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM faces
            WHERE bbox_x1 IS NOT NULL
            {}
        """.format("" if force else
                   f"AND (face_thumbnail IS NULL OR (landmark_2d_106 IS NULL AND {arcface_sql}))"),
            [] if force else arcface_params)
        faces = cursor.fetchall()

    if not faces:
        logger.info("No faces to process.")
        return

    logger.info("Regenerating thumbnails for %s faces...", len(faces))
    logger.info("Workers: %s, Batch size: %s", num_workers, batch_size)

    processor = FaceProcessor(
        db_path=db_path,
        config=face_processing,
        num_workers=num_workers,
        batch_size=batch_size,
        mode='refill',
        thumbnail_size=thumbnail_size,
        thumbnail_quality=thumbnail_quality,
        crop_padding=crop_padding
    )

    processor.run(faces, desc="Regenerating thumbnails")




def run_face_clustering(db_path, config, force=False, preserve_named_only=False):
    """
    Run face clustering using settings from config.

    Args:
        db_path: Path to SQLite database
        config: ScoringConfig instance
        force: If True, delete all persons including named ones.
               If False (default), preserve all existing persons and match new faces to them.
        preserve_named_only: If True, only preserve named persons (name IS NOT NULL),
                            delete auto-clustered unnamed persons and re-cluster.

    Returns:
        True if clustering ran, False if disabled
    """
    cluster_settings = config.get_face_clustering_settings()
    if not cluster_settings.get('enabled', True):
        return False

    # The legacy profile is CPU-only by design; never use GPU clustering for it,
    # even on a machine that has a GPU (legacy deliberately avoids GPU models).
    use_gpu = cluster_settings.get('use_gpu', 'auto')
    profile = config.get_model_config().get('vram_profile', 'legacy')
    if profile == 'legacy' and use_gpu != 'never':
        logger.info("legacy profile: forcing CPU face clustering (use_gpu=never)")
        use_gpu = 'never'

    auto_merge = cluster_settings['auto_merge_distance_percent'] / 100
    face_processing = config.get_face_processing_settings()

    clusterer = FaceClusterer(
        db_path,
        min_faces=cluster_settings.get('min_faces_per_person', 2),
        min_samples=cluster_settings.get('min_samples'),
        auto_merge_distance=auto_merge,
        algorithm=cluster_settings.get('clustering_algorithm', 'boruvka_balltree'),
        leaf_size=cluster_settings.get('leaf_size', 40),
        use_gpu=use_gpu,
        merge_threshold=cluster_settings['merge_threshold'],
        use_db_thumbnails=face_processing.get('use_db_thumbnails', True),
        face_thumbnail_size=face_processing.get('face_thumbnail_size', 128),
        face_thumbnail_quality=face_processing.get('face_thumbnail_quality', 85)
    )
    clusterer.cluster_faces(force=force, preserve_named_only=preserve_named_only)
    return True
