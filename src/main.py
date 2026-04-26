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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

SEED = 42
CLASSES = ['hardware_bug', 'os_bug', 'platform_bug', 'ui_bug']

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


def make_dataset(n_per_class=750):
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
    return X, np.array(y)


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


def score(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='macro', zero_division=0),
            matthews_corrcoef(y_true, y_pred))


def cv(make_clf, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    a, f, m = [], [], []
    for tr, te in skf.split(X, y):
        clf = make_clf()
        clf.fit([X[i] for i in tr], y[tr])
        ai, fi, mi = score(y[te], clf.predict([X[i] for i in te]))
        a.append(ai); f.append(fi); m.append(mi)
    return np.array(a), np.array(f), np.array(m)


def cm_plot(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right')
    ax.set_yticklabels(CLASSES)
    thr = cm.max() / 2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thr else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


def boxplot(scores, path):
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
    plt.suptitle('10-fold CV: baseline vs solution (per-fold scores)')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    X, y = make_dataset()
    print(f'Dataset: {len(X)} reports, {len(set(y))} classes')

    print('\nBaseline (NB + TF-IDF), 10-fold CV')
    ba, bf, bm = cv(make_baseline, X, y)
    print(f'  Acc {ba.mean():.4f}+/-{ba.std():.4f}  F1 {bf.mean():.4f}+/-{bf.std():.4f}  MCC {bm.mean():.4f}+/-{bm.std():.4f}')

    print('\nSolution (LinearSVC + TF-IDF + char-ngram + stats), 10-fold CV')
    sa, sf, sm = cv(make_solution, X, y)
    print(f'  Acc {sa.mean():.4f}+/-{sa.std():.4f}  F1 {sf.mean():.4f}+/-{sf.std():.4f}  MCC {sm.mean():.4f}+/-{sm.std():.4f}')

    print('\nWilcoxon signed-rank (two-tailed)')
    rows = []
    for name, b, s in [('Accuracy', ba, sa), ('Macro-F1', bf, sf), ('MCC', bm, sm)]:
        if np.allclose(b, s):
            print(f'  {name}: identical')
            rows.append({'metric': name, 'W': None, 'p': None, 'significant': False})
            continue
        W, p = wilcoxon(b, s)
        print(f'  {name}: W={W:.2f}  p={p:.4f}  {"significant" if p < 0.05 else "not significant"}')
        rows.append({'metric': name, 'W': W, 'p': p, 'significant': p < 0.05})

    pd.DataFrame({
        'fold': range(1, 11),
        'baseline_accuracy': ba, 'baseline_f1': bf, 'baseline_mcc': bm,
        'solution_accuracy': sa, 'solution_f1': sf, 'solution_mcc': sm,
    }).to_csv('results/cv_results.csv', index=False)
    pd.DataFrame(rows).to_csv('results/wilcoxon_results.csv', index=False)
    boxplot([('Accuracy', ba, sa), ('Macro-F1', bf, sf), ('MCC', bm, sm)],
            'results/comparison_boxplot.png')

    print('\nHeld-out 80/20 evaluation')
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    timing = []
    for name, mk in [('baseline', make_baseline), ('solution', make_solution)]:
        clf = mk()
        t0 = time.perf_counter(); clf.fit(Xtr, ytr); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = clf.predict(Xte); t_pred = time.perf_counter() - t0
        a, f, m = score(yte, pred)
        timing.append({'model': name, 'fit_seconds': round(t_fit, 3), 'predict_seconds': round(t_pred, 3)})
        print(f'  [{name}] Acc={a:.4f}  F1={f:.4f}  MCC={m:.4f}  fit={t_fit:.2f}s  predict={t_pred:.2f}s')
        print(classification_report(yte, pred, target_names=CLASSES, zero_division=0))
        cm_plot(yte, pred, f'results/cm_{name}.png', name.title())
    pd.DataFrame(timing).to_csv('results/timing.csv', index=False)

    print('\nLow-resource ablation (25 reports per class for training)')
    configs = [
        ('full',                     {}),
        ('no_char_ngram',            {'use_char': False}),
        ('no_text_stats',            {'use_stats': False}),
        ('word_unigrams_only',       {'word_ngram': (1, 1)}),
        ('word_only_no_extras',      {'use_char': False, 'use_stats': False, 'word_ngram': (1, 1)}),
    ]
    abl_rng = np.random.RandomState(SEED)
    by_class = {c: np.where(y == c)[0] for c in np.unique(y)}
    abl_train = np.concatenate([abl_rng.choice(idx, 25, replace=False) for idx in by_class.values()])
    abl_test = np.setdiff1d(np.arange(len(y)), abl_train)
    Xtr_abl = [X[i] for i in abl_train]; ytr_abl = y[abl_train]
    Xte_abl = [X[i] for i in abl_test];  yte_abl = y[abl_test]
    abl_rows = []
    for name, kw in configs:
        clf = make_solution(**kw)
        clf.fit(Xtr_abl, ytr_abl)
        a, f, m = score(yte_abl, clf.predict(Xte_abl))
        abl_rows.append({'config': name,
                         'accuracy': round(a, 4),
                         'macro_f1': round(f, 4),
                         'mcc':      round(m, 4)})
        print(f'  {name:<22s}  Acc={a:.4f}  F1={f:.4f}  MCC={m:.4f}')
    pd.DataFrame(abl_rows).to_csv('results/ablation.csv', index=False)
