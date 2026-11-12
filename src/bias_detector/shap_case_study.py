# @title
# Módulo shap_case_study.py para automatizar este proceso

"""Estudio exploratorio de explicabilidad textual de 12 detecciones MLP-MIL v3.

No reentrena ni recalibra el clasificador. La concordancia con turnos anotados
NO constituye verificacion causal ni representatividad estadistica del corpus.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

QUOTAS = {
    ('confirmation_bias', 'TP'): 4,
    ('availability_bias', 'TP'): 3,
    ('confirmation_bias', 'FP'): 3,
    ('availability_bias', 'FP'): 2,
}


def _turn_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
    elif value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        items = []
    elif isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {'nan', 'none', 'null'}:
            items = []
        else:
            try:
                items = ast.literal_eval(s)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f'Formato de turnos no reconocido: {value!r}') from exc
            if isinstance(items, (int, str)):
                items = [items]
    else:
        raise TypeError(f'Tipo de anotacion de turnos inesperado: {type(value)}')
    return sorted(set(int(x) for x in items))


def _count_turns(conversation_text):
    return len(re.findall(r'\[TURN\s+\d+\s*\|\s*(?:USER|LLM)\]', str(conversation_text), re.I))


def select_cases(positive_windows, annotations, corpus, quotas=None):
    """Seleccion de casos reproducible sin usar SHAP ni modificar umbrales.

    Intenta priorizar conversaciones con >2 turnos, y al menos un ejemplo de
    multiples ventanas cuando existe. No presupone que todos los estratos existan.
    """
    quotas = QUOTAS if quotas is None else dict(quotas)
    p = positive_windows.copy()
    a = annotations.copy()
    c = corpus.copy()
    for df in (p, a, c):
        df['conversation_id'] = df['conversation_id'].astype(str)
    required_a = {'conversation_id', 'bias', 'final_label', 'final_trigger_turns', 'final_evidence_turns'}
    required_p = {'conversation_id', 'bias', 'score', 'threshold', 'window_id', 'n_windows', 'turn_ids'}
    required_c = {'conversation_id', 'conversation_text'}
    for name, df, req in [('annotations', a, required_a), ('positive_windows', p, required_p), ('corpus', c, required_c)]:
        if req - set(df.columns):
            raise ValueError(f'{name} sin columnas: {sorted(req - set(df.columns))}')
    if a.duplicated(['conversation_id', 'bias']).any():
        raise ValueError('Anotaciones duplicadas por conversacion/sesgo')
    if c['conversation_id'].duplicated().any():
        raise ValueError('Corpus con conversation_id duplicados')
    if p.duplicated(['conversation_id', 'bias']).any():
        raise ValueError('Detecciones positivas duplicadas')
    a = a[list(required_a)].copy()
    p = p.merge(a, on=['conversation_id', 'bias'], how='left', validate='one_to_one')
    if p['final_label'].isna().any():
        raise ValueError('Faltan anotaciones de alguna deteccion')
    labels = p['final_label'].astype(str).str.upper()
    if not labels.isin(['PRESENT', 'ABSENT']).all():
        raise ValueError('Etiqueta no reconocida (PRESENT/ABSENT)')
    p['case_type'] = np.where(labels.eq('PRESENT'), 'TP', 'FP')
    c['n_turns'] = c['conversation_text'].map(_count_turns)
    p = p.merge(c[['conversation_id', 'n_turns']], on='conversation_id', validate='many_to_one')
    p['n_windows'] = p['n_windows'].astype(int)
    p['score_margin'] = p['score'].astype(float) - p['threshold'].astype(float)
    # Seleccion razonada: diversidad de longitud, sin optimizar por concordancia.
    # n_turns y n_windows pueden usarse sin consultar los turnos de referencia.
    p = p.sort_values(['n_turns', 'n_windows', 'score_margin', 'conversation_id'],
                      ascending=[False, False, False, True], kind='mergesort')
    selected = []
    for (bias, label), n in quotas.items():
        pool = p[(p['bias'] == bias) & (p['case_type'] == label)].copy()
        if len(pool) < n:
            raise ValueError(f'No hay {n} casos en el estrato {bias}/{label}: {len(pool)}')
        # Garantizar al menos una conversacion multiventana por estrato si existe.
        multi = pool[pool['n_windows'] > 1]
        first = multi.head(1) if not multi.empty else pool.head(1)
        remaining = pool[~pool.index.isin(first.index)].head(n - len(first))
        selected.append(pd.concat([first, remaining]))
    out = pd.concat(selected, ignore_index=True)
    if len(out) != sum(quotas.values()):
        raise AssertionError('Numero de casos seleccionado incorrecto')
    out['case_id'] = [f'case_{i+1:02d}' for i in range(len(out))]
    out['final_trigger_turns'] = out['final_trigger_turns'].map(_turn_list)
    out['final_evidence_turns'] = out['final_evidence_turns'].map(_turn_list)
    out['turn_ids'] = out['turn_ids'].map(_turn_list)
    return out


def _turn_importances(shap_rows):
    """Importancia positiva maxima de cada grupo asignada a cada turno.

    Si un grupo atraviesa varios turnos, se comparte su importancia (no atribucion
    exclusiva); no se debe interpretar como una descomposicion SHAP por turno.
    """
    values = {}
    for _, row in shap_rows.iterrows():
        val = max(0.0, float(row['shap_value']))
        for tid in _turn_list(row['turn_ids']):
            values[tid] = max(values.get(tid, 0.0), val)
    return sorted(values.items(), key=lambda x: (-x[1], x[0]))


def _localization_row(case, shap_rows, metadata):
    ranks = _turn_importances(shap_rows)
    top1 = [t for t, _ in ranks[:1] if _ > 0]
    top2 = [t for t, _ in ranks[:2] if _ > 0]
    trigger = set(_turn_list(case['final_trigger_turns']))
    evidence = set(_turn_list(case['final_evidence_turns']))
    union = trigger | evidence
    result = {
        'case_id': case['case_id'], 'conversation_id': str(case['conversation_id']),
        'bias': case['bias'], 'case_type': case['case_type'],
        'n_turns': int(case['n_turns']), 'n_windows': int(case['n_windows']),
        'window_id': int(case['window_id']), 'score': float(case['score']),
        'n_groups': int(metadata['n_groups']),
        'original_score_recomputed': float(metadata['original_score_recomputed']),
        'additivity_error': float(metadata['additivity_error']),
        'top1_turns': top1, 'top2_turns': top2,
        'trigger_turns': sorted(trigger), 'evidence_turns': sorted(evidence),
        'reference_turns_union': sorted(union),
        'shap_turns_in_window': [t for t, _ in ranks],
        'top1_positive_score': ranks[0][1] if ranks else 0.,
    }
    # La concordancia solo tiene sentido en TP y cuando hay referencia.
    for k, picked in [(1, set(top1)), (2, set(top2))]:
        result[f'top{k}_n_selected'] = len(picked)
        result[f'top{k}_fraction_conversation'] = (len(picked) / result['n_turns']
                                                  if result['n_turns'] else np.nan)
        for name, ref in [('trigger', trigger), ('evidence', evidence), ('union', union)]:
            valid = case['case_type'] == 'TP' and bool(ref)
            result[f'{name}_precision_at_{k}'] = (len(picked & ref) / len(picked)
                                                     if valid and picked else np.nan)
            result[f'{name}_recall_at_{k}'] = (len(picked & ref) / len(ref)
                                                  if valid and picked else np.nan)
    if case['case_type'] == 'TP':
        result['trigger_coverage_window'] = (len(trigger & set(_turn_list(case['turn_ids']))) / len(trigger)
                                             if trigger else np.nan)
        result['evidence_coverage_window'] = (len(evidence & set(_turn_list(case['turn_ids']))) / len(evidence)
                                              if evidence else np.nan)
    else:
        result['trigger_coverage_window'] = np.nan
        result['evidence_coverage_window'] = np.nan
    return result


def run_case_study(cases, corpus, explainer, biases, output_dir, *, nsamples=64,
                   score_tolerance=0.02, additivity_tolerance=0.02):
    """Ejecuta y almacena cada explicacion; reanuda a partir de JSON/CSV ya guardados.

    Guarda los archivos de una explicacion solo tras validar puntuacion/aditividad.
    El resumen solo se produce cuando todos los casos terminan correctamente.
    """
    # from text_shap import save_shap_explanation
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    c = corpus.copy()
    c['conversation_id'] = c['conversation_id'].astype(str)
    text_lookup = c.set_index('conversation_id')['conversation_text']
    results = []
    for index, (_, case) in enumerate(cases.iterrows(), 1):
        cid, bias, wid = str(case['conversation_id']), str(case['bias']), int(case['window_id'])
        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', cid)
        base = f'{safe_id}__{bias}__w{wid}'
        csv_path, json_path = directory / (base + '.csv'), directory / (base + '.json')
        if csv_path.exists() != json_path.exists():
            raise RuntimeError(f'Explicacion guardada parcialmente: {base}')
        if csv_path.exists():
            shap_rows = pd.read_csv(csv_path, dtype={'conversation_id': str})
            with json_path.open(encoding='utf-8') as file:
                metadata = json.load(file)
            print(f'[{index}/{len(cases)}] Reutilizado: {cid} / {bias}')
        else:
            if cid not in text_lookup.index:
                raise KeyError(f'Conversacion ausente de df_corpus: {cid}')
            shap_rows, metadata = explainer.explain(
                conversation_id=cid, bias=bias, bias_names=biases,
                original_text=text_lookup.loc[cid], winning_window_id=wid,
                nsamples=nsamples
            )
            if abs(float(case['score']) - float(metadata['original_score_recomputed'])) > score_tolerance:
                raise ValueError(f'Desajuste de puntuacion en {cid}/{bias}')
            if float(metadata['additivity_error']) > additivity_tolerance:
                raise ValueError(f'Error de aditividad excesivo en {cid}/{bias}')
            save_shap_explanation(shap_rows, metadata, directory)
            print(f'[{index}/{len(cases)}] Explicado: {cid} / {bias}')
        if str(metadata['conversation_id']) != cid or str(metadata['bias']) != bias or int(metadata['window_id']) != wid:
            raise ValueError('Metadatos SHAP no corresponden al caso esperado')
        if abs(float(case['score']) - float(metadata['original_score_recomputed'])) > score_tolerance:
            raise ValueError('Puntuacion incoherente en explicacion recuperada')
        if float(metadata['additivity_error']) > additivity_tolerance:
            raise ValueError('Error de aditividad en explicacion recuperada')
        results.append(_localization_row(case, shap_rows, metadata))
    summary = pd.DataFrame(results)
    summary_path = directory / 'case_study_summary.csv'
    if summary_path.exists():
        raise FileExistsError(f'Ya existe el resumen: {summary_path}; eliminalo solo si vas a regenerarlo')
    # Listas en CSV se serializan como strings; al releer usar ast.literal_eval.
    summary.to_csv(summary_path, index=False)
    return summary


def aggregate_case_study(summary):
    """Descriptivos de los TP (sin inferencia estadistica)."""
    tp = summary.loc[summary['case_type'] == 'TP']
    fields = [f'{name}_{metric}_at_{k}' for name in ('trigger', 'evidence', 'union')
              for metric in ('precision', 'recall') for k in (1, 2)]
    fields += ['trigger_coverage_window', 'evidence_coverage_window', 'top1_fraction_conversation']
    grouped = tp.groupby('bias')[fields].mean(numeric_only=True).T
    grouped['TP_total'] = tp[fields].mean(numeric_only=True)
    return grouped


def build_qualitative_review(cases, shap_dir, *, top_n=3):
    """Crea tabla de revision humana sin inventar juicios semanticos."""
    folder = Path(shap_dir)
    rows = []
    for _, case in cases.iterrows():
        cid = str(case['conversation_id'])
        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', cid)
        filename = f"{safe_id}__{case['bias']}__w{int(case['window_id'])}.csv"
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        attrs = pd.read_csv(path)
        positive = attrs.loc[attrs['shap_value'] > 0].nlargest(top_n, 'shap_value')
        excerpts = ' | '.join(
            f"grupo {int(r['group_id'])}: {str(r['text']).strip()[:240]}"
            for _, r in positive.iterrows()
        )
        rows.append({
            'case_id': case['case_id'],
            'conversation_id': cid,
            'bias': case['bias'],
            'case_type': case['case_type'],
            'n_turns': int(case['n_turns']),
            'top_fragments': excerpts,
            # Campos para revisión humana
            "interpretacion_textual": "",
            "coherencia_con_el_sesgo": "",
            "observaciones": ""
        })

    df_review = pd.DataFrame(rows)

    return df_review