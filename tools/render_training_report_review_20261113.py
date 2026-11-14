"""Make numbered review contact sheets from every rendered report page."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'.build/progress-20261113-training'


def main():
    for lang in ('zh','en'):
        pages=sorted(BUILD.glob(f'page-{lang}-*.png'),key=lambda path:int(path.stem.rsplit('-',1)[1]))
        if not pages:raise ValueError(f'no rendered pages for {lang}')
        for start in range(0,len(pages),4):
            sheet=Image.new('RGB',(1800,2420),'#dddddd');draw=ImageDraw.Draw(sheet)
            for index,path in enumerate(pages[start:start+4]):
                with Image.open(path) as original:
                    page=ImageOps.contain(original.convert('RGB'),(880,1165))
                    x=(index%2)*900+(900-page.width)//2;y=(index//2)*1210+32
                    sheet.paste(page,(x,y));draw.text((x,y-24),f'{lang.upper()} page {start+index+1}',fill='black')
            out=BUILD/f'contact-{lang}-{start//4+1}.jpg';sheet.save(out,quality=95);print(out)


if __name__=='__main__':main()
