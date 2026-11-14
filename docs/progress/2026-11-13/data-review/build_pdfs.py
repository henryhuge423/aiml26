"""Build report PDFs while retaining the exact LaTeX sources beside Markdown."""
from pathlib import Path
import subprocess
import shutil
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BUILD=ROOT/'.build/progress-20261113'

def main():
    BUILD.mkdir(parents=True,exist_ok=True)
    results={}
    for lang in ['zh','en']:
        stem=f'data-and-degradation-review.{lang}'
        md=HERE/f'{stem}.md'
        tex=HERE/f'{stem}.tex'
        result=subprocess.run(['pandoc',str(md),'-s','-t','latex','-o',str(tex),
            '--pdf-engine=xelatex','-V','mainfont=Arial','-V','CJKmainfont=Microsoft YaHei',
            '-V','monofont=Consolas','-V','fontsize=10pt','-V','geometry:margin=0.72in',
            '-V','colorlinks=true','-V','urlcolor=blue','-V','linkcolor=blue'],cwd=HERE,
            capture_output=True,text=True,encoding='utf-8',errors='replace')
        (BUILD/f'{stem}.pandoc.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        result.check_returncode()
        # Let long paths wrap in a narrow table and keep readable prose spacing.
        text=tex.read_text(encoding='utf-8').replace('\\begin{figure}', '\\begin{figure}[H]')
        text=text.replace('\\begin{longtable}', '\\Needspace{10\\baselineskip}\n\\begin{longtable}')
        prefix='\\usepackage{float}\n\\usepackage{needspace}\n\\Urlmuskip=0mu plus 2mu\\relax\n'
        if lang=='zh':prefix+='\\renewcommand{\\figurename}{图}\n'
        text=text.replace('\\begin{document}',prefix+'\\begin{document}')
        tex.write_text(text,encoding='utf-8')
        for run in range(2):
            result=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',
                f'-output-directory={BUILD}',str(tex)],cwd=HERE,capture_output=True,
                text=True,encoding='utf-8',errors='replace')
            (BUILD/f'{stem}.xelatex-{run}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
            result.check_returncode()
        shutil.copyfile(BUILD/f'{stem}.pdf',HERE/f'{stem}.pdf')
        result=subprocess.run(['pdftotext','-layout',str(HERE/f'{stem}.pdf'),str(BUILD/f'{stem}.txt')],capture_output=True)
        result.check_returncode()
        pages=(BUILD/f'{stem}.txt').read_text(encoding='utf-8').count('\f')
        results[lang]=dict(pages=pages,pdf_bytes=(HERE/f'{stem}.pdf').stat().st_size,
            latex_retained=True)
    (HERE/'pdf_build.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
