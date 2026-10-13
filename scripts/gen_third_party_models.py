"""Generate docs/THIRD_PARTY_MODELS.md (and the six translations) from licensing.COMPONENTS.

The prose is translated by hand below; only the table between the
BEGIN/END GENERATED markers comes from the registry.

    python scripts/gen_third_party_models.py           # write the files
    python scripts/gen_third_party_models.py --check   # exit 1 when any file differs
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import licensing  # noqa: E402

BEGIN = '<!-- BEGIN GENERATED -->'
END = '<!-- END GENERATED -->'

LANG_NAMES = {'en': 'English', 'fr': 'Français', 'de': 'Deutsch', 'it': 'Italiano',
              'es': 'Español', 'pt': 'Português', 'zh': '简体中文'}

TEXT = {
    'en': dict(
        title='Third-party models and licences',
        intro=[
            'Facet itself is MIT-licensed, but several of the models it downloads and runs carry their own '
            'licences, and some of them forbid commercial use. This page lists every component, generated '
            'from the registry in `licensing.py`.',
            '**Commercial mode** (`licensing.mode: commercial` in `scoring_config.json`, coming in this '
            'release) makes Facet load only components marked "yes" (and "unclear" ones if '
            '`licensing.allow_unclear` is set). Restricted components are never imported, queued or '
            'downloaded; each slot is routed to a permissive replacement or the metric is reported as '
            'unavailable.',
        ],
        restricted='What is restricted by default',
        bullets=[
            '**pyiqa** (TOPIQ, TOPIQ-IAA, TOPIQ-NR-Face, LIQE, HyperIQA, DBCNN, MUSIQ, CLIP-IQA+ and the '
            'current Q-ReAlign wrapper): the 0.1.16 code is PolyForm Noncommercial 1.0.0 (earlier releases '
            'CC BY-NC-SA) and the weights are CC BY-NC-SA 4.0. The "Apache" classifier on PyPI is an error '
            '(upstream issue #294).',
            '**InsightFace buffalo_l**: the package is MIT, but the pretrained weights are for '
            'non-commercial research only. Commercial licences are available from '
            'recognition-oss-pack@insightface.ai.',
        ],
        table='Components',
        disclaimer='Disclaimer',
        disclaimers=[
            'A weights licence is not the licence of the training data. The licences of the training '
            'datasets (AVA, KonIQ, VGGFace2, MS1MV2, 300-W) have not been verified.',
            'This page is information, not legal advice, and comes with no warranty. Check the licences '
            'yourself before any commercial deployment.',
            "Immich's personal-use permission for InsightFace does not extend to Immich's users or to Facet's.",
        ]),
    'fr': dict(
        title='Modèles tiers et licences',
        intro=[
            "Facet est sous licence MIT, mais plusieurs des modèles qu'il télécharge et exécute ont leur "
            "propre licence, et certains interdisent l'usage commercial. Cette page liste chaque composant "
            'et est générée à partir du registre de `licensing.py`.',
            'Le **mode commercial** (`licensing.mode: commercial` dans `scoring_config.json`, à venir dans '
            'cette version) ne charge que les composants marqués « oui » (et « incertain » si '
            '`licensing.allow_unclear` est activé). Les composants restreints ne sont jamais importés, mis '
            'en file ni téléchargés ; chaque emplacement est redirigé vers un remplaçant permissif, ou la '
            'métrique est signalée comme indisponible.',
        ],
        restricted='Ce qui est restreint par défaut',
        bullets=[
            "**pyiqa** (TOPIQ, TOPIQ-IAA, TOPIQ-NR-Face, LIQE, HyperIQA, DBCNN, MUSIQ, CLIP-IQA+ et "
            "l'actuel wrapper Q-ReAlign) : le code 0.1.16 est sous PolyForm Noncommercial 1.0.0 (les "
            'versions antérieures sous CC BY-NC-SA) et les poids sous CC BY-NC-SA 4.0. Le classificateur '
            '« Apache » sur PyPI est une erreur (ticket amont n°294).',
            '**InsightFace buffalo_l** : le paquet est MIT, mais les poids pré-entraînés sont réservés à la '
            'recherche non commerciale. Des licences commerciales sont proposées via '
            'recognition-oss-pack@insightface.ai.',
        ],
        table='Composants',
        disclaimer='Avertissement',
        disclaimers=[
            "La licence des poids n'est pas celle des données d'entraînement. Les licences des jeux de "
            "données d'entraînement (AVA, KonIQ, VGGFace2, MS1MV2, 300-W) n'ont pas été vérifiées.",
            'Cette page est informative, ne constitue pas un avis juridique et est fournie sans garantie. '
            'Vérifiez vous-même les licences avant tout déploiement commercial.',
            "L'autorisation d'usage personnel d'InsightFace accordée à Immich ne s'étend ni aux "
            'utilisateurs d\'Immich ni à ceux de Facet.',
        ]),
    'de': dict(
        title='Modelle von Drittanbietern und Lizenzen',
        intro=[
            'Facet selbst steht unter der MIT-Lizenz, doch mehrere der heruntergeladenen und ausgeführten '
            'Modelle haben eigene Lizenzen, und einige verbieten die kommerzielle Nutzung. Diese Seite '
            'listet alle Komponenten und wird aus der Registry in `licensing.py` erzeugt.',
            'Der **kommerzielle Modus** (`licensing.mode: commercial` in `scoring_config.json`, in diesem '
            'Release geplant) lädt nur Komponenten mit „ja“ (und „unklar“, wenn '
            '`licensing.allow_unclear` gesetzt ist). Eingeschränkte Komponenten werden nie importiert, '
            'eingeplant oder heruntergeladen; jeder Slot wird auf einen freizügigen Ersatz umgeleitet, '
            'oder die Metrik wird als nicht verfügbar gemeldet.',
        ],
        restricted='Was standardmäßig eingeschränkt ist',
        bullets=[
            '**pyiqa** (TOPIQ, TOPIQ-IAA, TOPIQ-NR-Face, LIQE, HyperIQA, DBCNN, MUSIQ, CLIP-IQA+ und der '
            'aktuelle Q-ReAlign-Wrapper): Der Code 0.1.16 steht unter PolyForm Noncommercial 1.0.0 (frühere '
            'Versionen unter CC BY-NC-SA), die Gewichte unter CC BY-NC-SA 4.0. Die Klassifizierung '
            '„Apache“ auf PyPI ist ein Fehler (Upstream-Issue #294).',
            '**InsightFace buffalo_l**: Das Paket ist MIT, die vortrainierten Gewichte sind jedoch nur für '
            'nichtkommerzielle Forschung freigegeben. Kommerzielle Lizenzen gibt es über '
            'recognition-oss-pack@insightface.ai.',
        ],
        table='Komponenten',
        disclaimer='Haftungsausschluss',
        disclaimers=[
            'Die Lizenz der Gewichte ist nicht die Lizenz der Trainingsdaten. Die Lizenzen der '
            'Trainingsdatensätze (AVA, KonIQ, VGGFace2, MS1MV2, 300-W) wurden nicht geprüft.',
            'Diese Seite dient der Information, ist keine Rechtsberatung und erfolgt ohne Gewähr. Prüfen '
            'Sie die Lizenzen vor jedem kommerziellen Einsatz selbst.',
            'Die private Nutzungserlaubnis für InsightFace, die Immich erhalten hat, gilt weder für die '
            'Nutzer von Immich noch für die von Facet.',
        ]),
    'es': dict(
        title='Modelos de terceros y licencias',
        intro=[
            'Facet tiene licencia MIT, pero varios de los modelos que descarga y ejecuta tienen sus propias '
            'licencias, y algunos prohíben el uso comercial. Esta página lista cada componente y se genera '
            'a partir del registro de `licensing.py`.',
            'El **modo comercial** (`licensing.mode: commercial` en `scoring_config.json`, que llega en '
            'esta versión) solo carga los componentes marcados como «sí» (y los «poco claro» si se activa '
            '`licensing.allow_unclear`). Los componentes restringidos nunca se importan, se encolan ni se '
            'descargan; cada ranura se redirige a un reemplazo permisivo o la métrica se declara no '
            'disponible.',
        ],
        restricted='Qué está restringido por defecto',
        bullets=[
            '**pyiqa** (TOPIQ, TOPIQ-IAA, TOPIQ-NR-Face, LIQE, HyperIQA, DBCNN, MUSIQ, CLIP-IQA+ y el '
            'wrapper actual de Q-ReAlign): el código 0.1.16 es PolyForm Noncommercial 1.0.0 (las versiones '
            'anteriores, CC BY-NC-SA) y los pesos son CC BY-NC-SA 4.0. El clasificador «Apache» de PyPI es '
            'un error (incidencia upstream n.º 294).',
            '**InsightFace buffalo_l**: el paquete es MIT, pero los pesos preentrenados son solo para '
            'investigación no comercial. Hay licencias comerciales en recognition-oss-pack@insightface.ai.',
        ],
        table='Componentes',
        disclaimer='Aviso legal',
        disclaimers=[
            'La licencia de los pesos no es la licencia de los datos de entrenamiento. Las licencias de los '
            'conjuntos de entrenamiento (AVA, KonIQ, VGGFace2, MS1MV2, 300-W) no se han verificado.',
            'Esta página es informativa, no constituye asesoramiento jurídico y se ofrece sin garantía. '
            'Compruebe las licencias usted mismo antes de cualquier despliegue comercial.',
            'El permiso de uso personal de InsightFace concedido a Immich no se extiende a los usuarios de '
            'Immich ni a los de Facet.',
        ]),
    'it': dict(
        title='Modelli di terze parti e licenze',
        intro=[
            "Facet è distribuito con licenza MIT, ma diversi dei modelli che scarica ed esegue hanno licenze "
            "proprie, e alcuni vietano l'uso commerciale. Questa pagina elenca ogni componente ed è "
            'generata dal registro in `licensing.py`.',
            'La **modalità commerciale** (`licensing.mode: commercial` in `scoring_config.json`, in arrivo '
            'in questa versione) carica solo i componenti contrassegnati «sì» (e «incerto» se è attivo '
            '`licensing.allow_unclear`). I componenti con restrizioni non vengono mai importati, messi in '
            'coda o scaricati; ogni slot è indirizzato a un sostituto permissivo oppure la metrica è '
            'segnalata come non disponibile.',
        ],
        restricted='Cosa è limitato per impostazione predefinita',
        bullets=[
            "**pyiqa** (TOPIQ, TOPIQ-IAA, TOPIQ-NR-Face, LIQE, HyperIQA, DBCNN, MUSIQ, CLIP-IQA+ e "
            "l'attuale wrapper di Q-ReAlign): il codice 0.1.16 è PolyForm Noncommercial 1.0.0 (le "
            'versioni precedenti CC BY-NC-SA) e i pesi sono CC BY-NC-SA 4.0. Il classificatore «Apache» su '
            'PyPI è un errore (issue upstream n. 294).',
            '**InsightFace buffalo_l**: il pacchetto è MIT, ma i pesi preaddestrati sono solo per ricerca '
            'non commerciale. Le licenze commerciali si ottengono da recognition-oss-pack@insightface.ai.',
        ],
        table='Componenti',
        disclaimer='Avvertenza',
        disclaimers=[
            'La licenza dei pesi non è la licenza dei dati di addestramento. Le licenze dei dataset di '
            'addestramento (AVA, KonIQ, VGGFace2, MS1MV2, 300-W) non sono state verificate.',
            'Questa pagina ha scopo informativo, non costituisce consulenza legale ed è fornita senza '
            'garanzie. Verificate voi stessi le licenze prima di qualsiasi distribuzione commerciale.',
            "Il permesso d'uso personale di InsightFace concesso a Immich non si estende agli utenti di "
            'Immich né a quelli di Facet.',
        ]),
    'pt': dict(
        title='Modelos de terceiros e licenças',
        intro=[
            'O Facet tem licença MIT, mas vários dos modelos que baixa e executa têm licenças próprias, e '
            'alguns proíbem o uso comercial. Esta página lista cada componente e é gerada a partir do '
            'registro em `licensing.py`.',
            'O **modo comercial** (`licensing.mode: commercial` em `scoring_config.json`, previsto para '
            'esta versão) carrega apenas componentes marcados como «sim» (e «incerto» se '
            '`licensing.allow_unclear` estiver ativo). Componentes restritos nunca são importados, '
            'enfileirados nem baixados; cada slot é redirecionado a um substituto permissivo ou a métrica '
            'é informada como indisponível.',
        ],
        restricted='O que é restrito por padrão',
        bullets=[
            '**pyiqa** (TOPIQ, TOPIQ-IAA, TOPIQ-NR-Face, LIQE, HyperIQA, DBCNN, MUSIQ, CLIP-IQA+ e o '
            'wrapper atual do Q-ReAlign): o código 0.1.16 é PolyForm Noncommercial 1.0.0 (versões '
            'anteriores CC BY-NC-SA) e os pesos são CC BY-NC-SA 4.0. O classificador «Apache» no PyPI é um '
            'erro (issue upstream n.º 294).',
            '**InsightFace buffalo_l**: o pacote é MIT, mas os pesos pré-treinados são apenas para pesquisa '
            'não comercial. Licenças comerciais estão disponíveis em recognition-oss-pack@insightface.ai.',
        ],
        table='Componentes',
        disclaimer='Aviso legal',
        disclaimers=[
            'A licença dos pesos não é a licença dos dados de treinamento. As licenças dos conjuntos de '
            'treinamento (AVA, KonIQ, VGGFace2, MS1MV2, 300-W) não foram verificadas.',
            'Esta página é informativa, não constitui aconselhamento jurídico e é fornecida sem garantia. '
            'Verifique as licenças por conta própria antes de qualquer implantação comercial.',
            'A permissão de uso pessoal do InsightFace concedida ao Immich não se estende aos usuários do '
            'Immich nem aos do Facet.',
        ]),
    'zh': dict(
        title='第三方模型与许可证',
        intro=[
            'Facet 本身采用 MIT 许可证，但它下载并运行的若干模型有各自的许可证，其中一些禁止商业使用。'
            '本页列出所有组件，内容由 `licensing.py` 中的注册表生成。',
            '**商业模式**（`scoring_config.json` 中的 `licensing.mode: commercial`，将随本版本推出）只加载'
            '标记为“是”的组件（若设置了 `licensing.allow_unclear`，则包括“不明确”的组件）。受限组件绝不会被'
            '导入、排队或下载；每个环节都会改用宽松许可的替代品，或将该指标报告为不可用。',
        ],
        restricted='默认受限的内容',
        bullets=[
            '**pyiqa**（TOPIQ、TOPIQ-IAA、TOPIQ-NR-Face、LIQE、HyperIQA、DBCNN、MUSIQ、CLIP-IQA+ 以及当前的 '
            'Q-ReAlign 封装）：0.1.16 版代码采用 PolyForm Noncommercial 1.0.0（更早版本为 CC BY-NC-SA），'
            '权重为 CC BY-NC-SA 4.0。PyPI 上的“Apache”分类标签有误（上游 issue #294）。',
            '**InsightFace buffalo_l**：软件包为 MIT，但预训练权重仅限非商业研究使用。可通过 '
            'recognition-oss-pack@insightface.ai 获取商业许可。',
        ],
        table='组件',
        disclaimer='免责声明',
        disclaimers=[
            '权重的许可证不等同于训练数据的许可证。训练数据集（AVA、KonIQ、VGGFace2、MS1MV2、300-W）的许可证'
            '尚未核实。',
            '本页仅供参考，不构成法律意见，也不提供任何担保。进行任何商业部署之前，请自行核实各项许可证。',
            'Immich 获得的 InsightFace 个人使用许可不适用于 Immich 的用户，也不适用于 Facet 的用户。',
        ]),
}


def doc_path(lang):
    base = os.path.join(REPO_ROOT, 'docs')
    if lang == 'en':
        return os.path.join(base, 'THIRD_PARTY_MODELS.md')
    return os.path.join(base, lang, 'THIRD_PARTY_MODELS.md')


def _nav(lang):
    parts = []
    for code, name in LANG_NAMES.items():
        if code == lang:
            parts.append(f'**{name}**')
            continue
        if lang == 'en':
            target = 'THIRD_PARTY_MODELS.md' if code == 'en' else f'{code}/THIRD_PARTY_MODELS.md'
        else:
            target = '../THIRD_PARTY_MODELS.md' if code == 'en' else f'../{code}/THIRD_PARTY_MODELS.md'
        parts.append(f'[{name}]({target})')
    return '> 🌐 ' + ' · '.join(parts)


def render(lang):
    t = TEXT[lang]
    out = [f'# {t["title"]}', '', _nav(lang), '']
    for paragraph in t['intro']:
        out += [paragraph, '']
    out += [f'## {t["restricted"]}', '', *[f'- {b}' for b in t['bullets']], '',
            f'## {t["table"]}', '', BEGIN, licensing.render_markdown(lang), END, '',
            f'## {t["disclaimer"]}', '', *[f'- {d}' for d in t['disclaimers']], '']
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='exit 1 if any file differs')
    args = parser.parse_args(argv)
    stale = []
    for lang in licensing.LANGUAGES:
        path = doc_path(lang)
        want = render(lang)
        have = None
        if os.path.exists(path):
            # Universal newlines: a core.autocrlf=true checkout turns these files
            # CRLF, which is not staleness.
            with open(path, encoding='utf-8') as f:
                have = f.read()
        if have == want:
            continue
        stale.append(os.path.relpath(path, REPO_ROOT))
        if not args.check:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(want)
    if args.check and stale:
        print('Out of date (run scripts/gen_third_party_models.py): ' + ', '.join(stale))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
