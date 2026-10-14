**DRAFT - NOT POSTED**

Thanks for raising this, and for being specific about what you need. Here is what I built.

**What it does.** There is now an opt-in commercial mode. In it Facet never imports, queues or downloads `pyiqa` or InsightFace, so no PyIQA code and no InsightFace weights are touched. Components with an unclear licence (SAMP-Net, U2-Net-P, the LAION CLIP-MLP head) are excluded too, unless you set `licensing.allow_unclear`. Each slot gets a permissive replacement: Q-ReAlign (loaded natively, no pyiqa) for aesthetic and quality, rule-based composition, and YuNet + SFace + MediaPipe for faces. OFIQ can optionally provide face quality. A refused component stops the scan at startup with an error naming it, rather than failing silently.

**How to enable it.**
- Config: `{ "licensing": { "mode": "commercial" } }` in your `scoring_config.json`.
- Or the environment variable `FACET_LICENSING_MODE=commercial` (it can only tighten, never relax the file).
- Docker: `docker build --build-arg FACET_LICENSING=commercial .`
- Install: `bash install.sh --commercial`, or `pip install -r requirements.txt`, `pip install -e .[commercial]`, then `pip install mediapipe==0.10.35 --no-deps`.
- `python facet.py --doctor` shows the active licence posture.

Note that the base install no longer includes insightface or pyiqa. For the old behaviour use `pip install -e .[all]`.

**What changes in commercial mode.**
- LIQE and the supplementary aesthetic (IAA) metric have no replacement, so they stay empty; their weights are renormalised away and the viewer hides them.
- Face quality exists only if you install `python-ofiq` (Linux x86-64 or macOS arm64 wheels).
- SFace is less accurate than ArcFace. On my own library, at a comparable false-merge rate, its same-person recall was about 12 points lower, so expect more manual merging.

**Licences.** Every component, with its code and weights licence, is listed in `docs/THIRD_PARTY_MODELS.md`. Please read the disclaimer there: a weights licence is not the licence of the training data (AVA, KonIQ, VGGFace2, MS1MV2 and 300-W are unverified), and this is information, not legal advice. Check the licences yourself before any commercial deployment.

**Docs.** `docs/THIRD_PARTY_MODELS.md`, `docs/CONFIGURATION.md` (Licensing), `docs/INSTALLATION.md` (commercial install), `docs/FACE_RECOGNITION.md`.

Feedback is welcome, especially on face results on your own photos.
