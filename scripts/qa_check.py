"""Verify the rebuilt report, deck and their evidence before packaging."""
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
if __package__ is None:
    sys.path.insert(0, str(CODE))
from src.evidence import evidence_grid, grid_errors

REPORT = CODE.parent / 'report'
MTP = CODE.parents[1] / 'mtp'
REPOSITORY = 'https://github.com/TikaNice/CISC-473-Group-23-Aspect-Based-Sentiment-Analysis-with-Graph-Augmented-Transformers'
MEMBERS = {'Harsh Shrivastav', 'Ibrahim Kettaneh', 'Jinpeng Deng'}
PLACEHOLDERS = ('write your', 'todo', 'guidance.', 'xx.x', '[project id and title]', '[describe',
                'runs pending', 'runs still in progress', 'examples pending',
                'regenerated from saved predictions at build time', 'need three seeds')
CONTRIBUTION_ROW = re.compile(r'^([A-Za-z]+ [A-Za-z]+)\s*&[^&]*&\s*(\d{1,3})\s*\\\\', re.M)


def read(path, failures):
    try:
        return path.read_text(encoding='utf-8')
    except OSError as exc:
        failures.append(str(exc))
        return ''


def pdf_tool(command, failures):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, 'stderr', None) or str(exc)
        failures.append(f'{command[0]}: {detail.strip()}')
        return ''


def check_pdf(tex, files, failures):
    pdf = tex.with_suffix('.pdf')
    if not pdf.is_file():
        failures.append(f'missing {pdf.name}; run bash scripts/reproduce.sh --pdf')
        return ''
    mtime = pdf.stat().st_mtime_ns
    for path in files:
        if path.is_file() and path.stat().st_mtime_ns > mtime:
            failures.append(f'{pdf.name} predates {path.name}')
    body = pdf_tool(['pdftotext', '-layout', str(pdf), '-'], failures)
    links = pdf_tool(['pdfinfo', '-url', str(pdf)], failures)
    if links and REPOSITORY not in links:
        failures.append(f'{pdf.name} is missing the project repository hyperlink')
    return body


def check_placeholders(name, text, failures):
    for token in PLACEHOLDERS:
        if token in text.lower():
            failures.append(f'{name}: placeholder {token!r}')


def report_layout(report, failures):
    pages = report.rstrip('\f').split('\f')
    heading = re.compile(r'^\s*References\s*$', re.M)
    count = None
    for i, page in enumerate(pages):
        marker = heading.search(page)
        if marker:
            above = [line for line in page[:marker.start()].splitlines()
                     if line.strip() and 'Final Report' not in line]
            count = i + bool(above)
            break
    if count is None:
        failures.append('report PDF has no References heading')
        return
    if not 6 <= count <= 8:
        failures.append(f'report main body spans {count} pages; required 6–8')
    bibliography = heading.split(report, maxsplit=1)[1].split('Implementation Environment', 1)[0]
    keys = set(re.findall(r'^\s*\[(\d+)\]\s', bibliography, re.M))
    if len(keys) < 7:
        failures.append(f'report PDF has {len(keys)} distinct bibliography entries; required at least 7')


def check_contributions(sources, pdfs, failures):
    for name, source, pdf in zip(('report', 'deck'), sources, pdfs):
        tables = re.findall(r'\\begin\{tabular\}(.*?)\\end\{tabular\}', source, re.S)
        tables = [table for table in tables if r'\textbf{Contributions}' in table]
        rows = CONTRIBUTION_ROW.findall(tables[0]) if len(tables) == 1 else []
        percentages = {member: int(pct) for member, pct in rows}
        awaiting = re.search(r'awaiting (?:group )?confirmation', source, re.I)
        if awaiting or len(rows) != len(MEMBERS) or percentages.keys() != MEMBERS or sum(percentages.values()) != 100:
            failures.append(f'{name}: confirm the percentages in one contribution table with three unique members totaling 100%')
            continue
        if pdf and any(not re.search(rf'{member}[^\n]*\s{pct}\b', pdf)
                       for member, pct in percentages.items()):
            failures.append(f'{name} PDF is missing a declared contribution or percentage')


def check_index(failures):
    root = CODE / 'results'
    index = read(root / 'INDEX.md', failures)
    for link in re.findall(r'\]\(([^)]+)\)', index):
        if not (root / link).is_file():
            failures.append(f'results/INDEX.md references missing file {link}')
    path = root / 'reports/experiment_status.csv'
    if path.is_file():
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        expected_count = sum(1 for _ in evidence_grid())
        if len(rows) != expected_count or any(
            row['status'] != 'complete' or not (CODE / row['file']).is_file() for row in rows
        ):
            failures.append(f'experiment_status.csv must list {expected_count} complete runs with existing source records')
    else:
        failures.append(f'missing {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-only', action='store_true',
                        help='Verify PDFs and evidence before team contributions are finalized')
    args = parser.parse_args()
    os.chdir(CODE)
    failures = grid_errors()
    report = REPORT / 'Group23_Final_Report.tex'
    deck = MTP / 'Group23_Slides.tex'
    report_assets = [
        REPORT / 'tables' / name for name in ('numbers.tex', 'main_results.tex', 'transfer.tex', 'failures.tex')
    ] + [REPORT / 'figures' / name for name in ('data_efficiency.png', 'graph_control.png', 'confusion_transfer.png')]
    deck_assets = [
        MTP / name for name in ('numbers.tex', 'ablation_slide.tex', 'budget_slide.tex', 'failures_slide.tex')
    ] + [MTP / 'figures' / name for name in ('data_efficiency.png', 'confusion_transfer.png')]
    sources = [read(path, failures) for path in (report, deck)]
    script = read(MTP / 'VIDEO_SCRIPT.md', failures)
    for path in [report, deck, *report_assets, *deck_assets]:
        if path.suffix == '.tex':
            text = read(path, failures)
            check_placeholders(path.name, text, failures)
            if re.search(r'&\s*--\s*(?:&|\\\\)', text):
                failures.append(f'{path.name}: empty data cell')
        elif not path.is_file():
            failures.append(f'missing {path}')
    report_macros = read(report_assets[0], failures)
    deck_macros = read(deck_assets[0], failures)
    if report_macros != deck_macros:
        failures.append('report and deck numbers.tex disagree')
    values = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', report_macros))
    combined = '\n'.join(sources)
    used = set(re.findall(r'\\([A-Z]\w+)', combined))
    for name in sorted(used & values.keys()):
        if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?', values[name]):
            failures.append(f'\\{name} has no numeric value')
    for path in (REPORT / 'tables/failures.tex', MTP / 'failures_slide.tex'):
        text = read(path, failures)
        if len(re.findall(r'^\d+\s*&', text, re.M)) < 3:
            failures.append(f'{path.name}: requires three indexed failure examples')
    for name in re.findall(r'\\([A-Z]\w+)\b', script):
        failures.append(f'video script contains unresolved LaTeX macro \\{name}')

    pdfs = [check_pdf(report, [report, *report_assets], failures),
            check_pdf(deck, [deck, *deck_assets], failures)]
    for name, pdf in zip(('report PDF', 'deck PDF'), pdfs):
        if pdf:
            check_placeholders(name, pdf, failures)
            for metric in ('BertRestF', 'GcnLapF', 'TrBertF', 'TrGcnF'):
                value = values.get(metric, '')
                if value and value != '--' and value not in pdf:
                    failures.append(f'{name} does not reproduce {metric} = {value}')
    if pdfs[0]:
        report_layout(pdfs[0], failures)
    if pdfs[1] and pdfs[1].count('\f') != sources[1].count(r'\begin{frame}'):
        failures.append('deck PDF page count differs from the slide source frame count')
    if not args.build_only:
        check_contributions(sources, pdfs, failures)
    check_index(failures)
    if failures:
        for message in failures:
            print(f'FAIL {message}')
        print(f'\n{len(failures)} check(s) failed.')
        return 1
    print('Evidence and PDF build checks passed.' if args.build_only else 'All automated submission checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
