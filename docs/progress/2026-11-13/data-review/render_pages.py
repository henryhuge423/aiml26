"""Render all report pages into numbered contact sheets for delivery review."""
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageOps, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def main():
    stage=sys.argv[1] if len(sys.argv)>1 else 'final'
    assert stage in {'preliminary','final'}
    build=ROOT/'.build/progress-20261113'/stage
    build.mkdir(parents=True,exist_ok=True)
    for lang in ['zh','en']:
        # Remove only this renderer's prior page/contact images so a shorter
        # rebuilt PDF cannot leave stale pages in the delivery review.
        for pattern in [f'page-{lang}-*.png',f'contact-{lang}-*.jpg']:
            for path in build.glob(pattern):
                assert path.resolve().parent==build.resolve()
                path.unlink()
        prefix=build/f'page-{lang}'
        subprocess.run(['pdftoppm','-scale-to','1500','-png',str(HERE/f'data-and-degradation-review.{lang}.pdf'),str(prefix)],check=True,capture_output=True)
        pages=sorted(build.glob(f'page-{lang}-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
        for start in range(0,len(pages),4):
            selected=pages[start:start+4]
            sheet=Image.new('RGB',(1700,2280),'#cccccc');draw=ImageDraw.Draw(sheet)
            for j,path in enumerate(selected):
                with Image.open(path) as im:
                    im=ImageOps.contain(im.convert('RGB'),(830,1100))
                    x=(j%2)*850+(850-im.width)//2;y=(j//2)*1140+30
                    sheet.paste(im,(x,y))
                    draw.text((x,y-22),f'{lang.upper()} page {start+j+1}',fill='black')
            out=build/f'contact-{lang}-{start//4+1}.jpg'
            sheet.save(out,quality=92)
            print(str(out))

if __name__=='__main__':main()
