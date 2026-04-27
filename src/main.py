import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from sklearn.datasets import fetch_20newsgroups

SEED = 42

# -----------------------------------------------------------------------------
# Dataset 1: synthetic bug reports
# -----------------------------------------------------------------------------
SYN_CLASSES = ['hardware_bug', 'os_bug', 'platform_bug', 'ui_bug']

TEMPLATES = {
    0: [
        "System crashes when {action} the {component}. {detail} Error: {error}.",
        "BSOD triggered by {component} during {action}. {detail}",
        "Hardware failure in {component} after {action}. Kernel log: {error}. {detail}",
        "{component} not detected after {action}. BIOS shows {error}. {detail}",
        "Firmware bug in {component}: {error} on {action}. {detail}",
        "I/O error from {component} while {action}. Message: {error}. {detail}",
    ],
    1: [
        "Windows {version} throws {error} when {action} {component}. {detail}",
        "OS freeze during {action} on {component}. dmesg shows {error}. {detail}",
        "Kernel panic after {action} {component}. Trace: {error}. {detail}",
        "Service {component} aborts with {error} after {action}. {detail}",
        "systemd unit {component} crashes on {action}: {error}. {detail}",
        "Regression: {component} returns {error} after upgrading to {version}. {detail}",
    ],
    2: [
        "macOS {version} incompatibility with {component}: {error}. {detail}",
        "Application crashes on macOS when {action} {component}. {detail}",
        "Cross-platform defect in {component}: {error} on {action}. {detail}",
        "{component} not working on {version}. Error: {error}. {detail}",
        "Porting issue: {component} raises {error} on non-Linux platform. {detail}",
        "{action} {component} fails silently on macOS {version}. {detail}",
    ],
    3: [
        "UI element {component} not rendering after {action}. {detail}",
        "Window manager crashes when {action} {component}: {error}. {detail}",
        "X11 {component} error: {error} on {action}. {detail}",
        "Display glitch in {component} during {action}. {detail}",
        "Wayland compositor deadlock when {action} {component}: {error}. {detail}",
        "GTK widget {component} disappears after {action}. Error: {error}. {detail}",
    ],
}
ACTIONS = ['installing', 'updating', 'removing', 'configuring', 'restarting',
           'booting', 'compiling', 'running', 'loading', 'mounting', 'unmounting',
           'suspending', 'resuming', 'resizing', 'initialising']
COMPONENTS = ['AHCI driver', 'USB hub', 'GPU driver', 'RAM module', 'network card',
              'BIOS firmware', 'kernel module', 'PCI controller', 'NVMe SSD', 'ACPI layer',
              'systemd service', 'libc', 'OpenGL renderer', 'display server', 'audio daemon']
ERRORS = ['null pointer dereference', 'segmentation fault', 'kernel panic',
          'access violation', 'divide-by-zero exception', 'stack overflow',
          'out-of-memory error', 'connection timeout', 'deadlock detected',
          'assertion failed', 'bus error', 'illegal instruction', 'unhandled IRQ']
DETAILS = ["Reproducible every time with the steps above.",
           "Happens intermittently under heavy load.",
           "First noticed after the latest kernel update.",
           "Tested on three separate machines with identical results.",
           "Attached dmesg log shows repeated I/O errors.",
           "Stack trace is available in the linked Gist.",
           "Worked correctly in the previous stable release.",
           "No known workaround has been found yet.",
           "Bisected to commit a3f9d12 in the mainline tree.",
           "Memory sanitiser flags a use-after-free at this site."]
VERSIONS = ['10.15', '11.0', '12.1', 'Ventura 13', 'Monterey 12.6', '2.6.32', '5.15', '6.1']


def make_synthetic(n_per_class=750):
    rng = random.Random(SEED)
    X, y = [], []
    for label, tmpls in TEMPLATES.items():
        for _ in range(n_per_class):
            X.append(rng.choice(tmpls).format(
                action=rng.choice(ACTIONS),
                component=rng.choice(COMPONENTS),
                error=rng.choice(ERRORS),
                detail=rng.choice(DETAILS),
                version=rng.choice(VERSIONS)))
            y.append(label)
    pairs = list(zip(X, y))
    rng.shuffle(pairs)
    X, y = map(list, zip(*pairs))
    return X, np.array(y), SYN_CLASSES


# -----------------------------------------------------------------------------
# Dataset 2: 20 Newsgroups, four computer-related categories
# Mapping back to the bug-class labels used elsewhere:
#   comp.sys.ibm.pc.hardware  -> hardware_bug
#   comp.os.ms-windows.misc   -> os_bug
#   comp.sys.mac.hardware     -> platform_bug
#   comp.windows.x            -> ui_bug
# -----------------------------------------------------------------------------
NEWS_CATS = ['comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc',
             'comp.sys.mac.hardware', 'comp.windows.x']
NEWS_LABELS = ['hardware_bug', 'os_bug', 'platform_bug', 'ui_bug']


def make_newsgroups():
    d = fetch_20newsgroups(subset='all', categories=NEWS_CATS,
                           remove=('headers', 'footers', 'quotes'),
                           random_state=SEED)
    return list(d.data), np.array(d.target), NEWS_LABELS


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [[
            len(t), len(t.split()), t.count('\n'),
            sum(1 for c in t if c.isupper()),
            sum(1 for c in t if c.isdigit()),
            t.count('error') + t.count('Error') + t.count('ERROR'),
            t.count('bug') + t.count('Bug'),
            t.count('fix') + t.count('Fix'),
        ] for t in X]
        return csr_matrix(rows, dtype=float)


def make_baseline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 1),
                                  stop_words='english', min_df=2)),
        ('clf', MultinomialNB(alpha=1.0)),
    ])


def make_solution(use_word=True, use_char=True, use_stats=True, word_ngram=(1, 2)):
    parts = []
    if use_word:
        parts.append(('w', TfidfVectorizer(max_features=50000, ngram_range=word_ngram,
                                           sublinear_tf=True, stop_words='english',
                                           min_df=2)))
    if use_char:
        parts.append(('c', TfidfVectorizer(max_features=30000, ngram_range=(2, 4),
                                           sublinear_tf=True, analyzer='char_wb',
                                           min_df=3)))
    if use_stats:
        parts.append(('s', TextStats()))
    return Pipeline([
        ('feat', FeatureUnion(parts)),
        ('scale', MaxAbsScaler()),
        ('clf', LinearSVC(C=1.0, max_iter=50000, tol=1e-3, random_state=SEED)),
    ])


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
def score(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='macro', zero_division=0),
            matthews_corrcoef(y_true, y_pred))


def tune_solution(X, y):
    """Small grid search over the SVC C parameter on a 75/25 internal split.
    Returns the best C value found by Macro-F1."""
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    grid = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_c, best_f = grid[0], -1.0
    rows = []
    for c in grid:
        clf = make_solution()
        clf.set_params(clf__C=c)
        clf.fit(Xtr, ytr)
        f = f1_score(yva, clf.predict(Xva), average='macro', zero_division=0)
        rows.append({'C': c, 'val_macro_f1': round(f, 4)})
        if f > best_f:
            best_f, best_c = f, c
    return best_c, rows


def cv(make_clf, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    a, f, m = [], [], []
    for tr, te in skf.split(X, y):
        clf = make_clf()
        clf.fit([X[i] for i in tr], y[tr])
        ai, fi, mi = score(y[te], clf.predict([X[i] for i in te]))
        a.append(ai); f.append(fi); m.append(mi)
    return np.array(a), np.array(f), np.array(m)


def cm_plot(y_true, y_pred, classes, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_yticklabels(classes)
    thr = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thr else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def boxplot(scores, path, suptitle):
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    for i, (name, b, s) in enumerate(scores):
        bp = ax[i].boxplot([b, s], tick_labels=['Baseline', 'Solution'],
                           patch_artist=True, widths=0.45)
        bp['boxes'][0].set_facecolor('#9ec5e8')
        bp['boxes'][1].set_facecolor('#a8d5a3')
        for j, data in enumerate([b, s], start=1):
            ax[i].scatter(np.full_like(data, j) + np.random.RandomState(0).uniform(-0.05, 0.05, len(data)),
                          data, color='black', s=18, alpha=0.6, zorder=3)
        lo = min(b.min(), s.min()) - 0.01
        hi = max(b.max(), s.max()) + 0.005
        ax[i].set_title(name); ax[i].set_ylabel('Score')
        ax[i].set_ylim(max(0.0, lo), min(1.005, hi))
        ax[i].grid(True, axis='y', alpha=0.3)
    plt.suptitle(suptitle)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def run_dataset(name, X, y, classes, suptitle, with_ablation=True):
    print(f'\n=== {name.upper()} ({len(X)} reports, {len(classes)} classes) ===')

    print('Hyperparameter search for solution (LinearSVC C)')
    best_c, hpo_rows = tune_solution(X, y)
    print(f'  best C={best_c}')
    pd.DataFrame(hpo_rows).to_csv(f'results/hpo_{name}.csv', index=False)
    def tuned_solution():
        clf = make_solution()
        clf.set_params(clf__C=best_c)
        return clf

    print('Baseline (NB + TF-IDF), 10-fold CV')
    ba, bf, bm = cv(make_baseline, X, y)
    print(f'  Acc {ba.mean():.4f}+/-{ba.std():.4f}  F1 {bf.mean():.4f}+/-{bf.std():.4f}  MCC {bm.mean():.4f}+/-{bm.std():.4f}')

    print('Solution (LinearSVC + TF-IDF + char-ngram + stats), 10-fold CV')
    sa, sf, sm = cv(tuned_solution, X, y)
    print(f'  Acc {sa.mean():.4f}+/-{sa.std():.4f}  F1 {sf.mean():.4f}+/-{sf.std():.4f}  MCC {sm.mean():.4f}+/-{sm.std():.4f}')

    print('Wilcoxon signed-rank (two-tailed) + rank-biserial effect size')
    rows = []
    n = 10
    total_rank_sum = n * (n + 1) / 2  # = 55
    for metric, b, s in [('Accuracy', ba, sa), ('Macro-F1', bf, sf), ('MCC', bm, sm)]:
        if np.allclose(b, s):
            print(f'  {metric}: identical (|r|=1.0)')
            rows.append({'dataset': name, 'metric': metric, 'W': None, 'p': None,
                         'rank_biserial_r': 1.0, 'effect_size_label': 'large',
                         'significant': False})
            continue
        W, p = wilcoxon(b, s)
        # scipy returns W = min(W+, W-); rank-biserial |r| = |W+ - W-| / (W+ + W-)
        r = abs((total_rank_sum - 2 * W) / total_rank_sum)
        if r >= 0.5:
            label = 'large'
        elif r >= 0.3:
            label = 'medium'
        elif r >= 0.1:
            label = 'small'
        else:
            label = 'negligible'
        sig = p < 0.05
        print(f'  {metric}: W={W:.2f}  p={p:.4f}  |r|={r:.3f} ({label})  {"significant" if sig else "not significant"}')
        rows.append({'dataset': name, 'metric': metric, 'W': float(W), 'p': float(p),
                     'rank_biserial_r': round(float(r), 4), 'effect_size_label': label,
                     'significant': sig})

    cv_df = pd.DataFrame({
        'fold': range(1, 11),
        'baseline_accuracy': ba, 'baseline_f1': bf, 'baseline_mcc': bm,
        'solution_accuracy': sa, 'solution_f1': sf, 'solution_mcc': sm,
    })
    cv_df.to_csv(f'results/cv_{name}.csv', index=False)
    boxplot([('Accuracy', ba, sa), ('Macro-F1', bf, sf), ('MCC', bm, sm)],
            f'results/boxplot_{name}.png', suptitle)

    print('Held-out 80/20')
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    timing = []
    for label, mk in [('baseline', make_baseline), ('solution', tuned_solution)]:
        clf = mk()
        t0 = time.perf_counter(); clf.fit(Xtr, ytr); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = clf.predict(Xte); t_pred = time.perf_counter() - t0
        a, f, m = score(yte, pred)
        timing.append({'dataset': name, 'model': label,
                       'fit_seconds': round(t_fit, 3), 'predict_seconds': round(t_pred, 3),
                       'test_accuracy': round(a, 4), 'test_f1': round(f, 4), 'test_mcc': round(m, 4)})
        print(f'  [{label}] Acc={a:.4f}  F1={f:.4f}  MCC={m:.4f}  fit={t_fit:.2f}s  predict={t_pred:.2f}s')
        print(classification_report(yte, pred, target_names=classes, zero_division=0))
        cm_plot(yte, pred, classes, f'results/cm_{name}_{label}.png', f'{name}: {label}')

    abl_rows = []
    if with_ablation:
        print('Feature contribution check (small training subset)')
        configs = [('full', {}),
                   ('no_char_ngram', {'use_char': False}),
                   ('no_text_stats', {'use_stats': False}),
                   ('word_unigrams_only', {'word_ngram': (1, 1)}),
                   ('word_only_no_extras', {'use_char': False, 'use_stats': False, 'word_ngram': (1, 1)})]
        rng_a = np.random.RandomState(SEED)
        per_class = 25
        by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
        train_idx = np.concatenate([rng_a.choice(idx, per_class, replace=False) for idx in by_class.values()])
        test_idx = np.setdiff1d(np.arange(len(y)), train_idx)
        Xtr_a = [X[i] for i in train_idx]; ytr_a = y[train_idx]
        Xte_a = [X[i] for i in test_idx];  yte_a = y[test_idx]
        for cname, kw in configs:
            clf = make_solution(**kw)
            clf.fit(Xtr_a, ytr_a)
            a, f, m = score(yte_a, clf.predict(Xte_a))
            abl_rows.append({'dataset': name, 'config': cname,
                             'accuracy': round(a, 4), 'macro_f1': round(f, 4), 'mcc': round(m, 4)})
            print(f'  {cname:<22s}  Acc={a:.4f}  F1={f:.4f}  MCC={m:.4f}')

    return rows, timing, abl_rows


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    Xs, ys, syn_classes = make_synthetic()
    syn_w, syn_t, syn_a = run_dataset('synthetic', Xs, ys, syn_classes,
                                      '10-fold CV on synthetic bug-report corpus',
                                      with_ablation=True)

    Xn, yn, news_classes = make_newsgroups()
    news_w, news_t, news_a = run_dataset('20news', Xn, yn, news_classes,
                                         '10-fold CV on 20 Newsgroups (computer categories)',
                                         with_ablation=False)

    pd.DataFrame(syn_w + news_w).to_csv('results/wilcoxon.csv', index=False)
    pd.DataFrame(syn_t + news_t).to_csv('results/timing.csv', index=False)
    pd.DataFrame(syn_a).to_csv('results/ablation.csv', index=False)
    print('\nAll outputs in results/')
