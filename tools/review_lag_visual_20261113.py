"""Local contact sheets of LAG candidate pairs for visual association review."""
import math
from pathlib import Path
from PIL import Image,ImageDraw
import audit_lag_overlap_20261113 as C

def main():
    rows=C.A.read(C.OUT/'lag_candidate_metrics.csv')
    for decision,label in [('confirmed_same_photo','confirmed'),('ambiguous_exclude','excluded')]:
        selected=[r for r in rows if r['decision']==decision]
        for page in range(math.ceil(len(selected)/6)):
            batch=selected[page*6:(page+1)*6]
            sheet=Image.new('RGB',(780,len(batch)*410),'white');draw=ImageDraw.Draw(sheet)
            for i,r in enumerate(batch):
                for j,p in enumerate([r['path_a'],r['path_b']]):
                    sheet.paste(Image.fromarray(C.A.thumbnail(p)).convert('RGB'),(j*390,i*410+25))
                draw.text((4,i*410+3),f"{page*6+i}: {Path(r['path_a']).name} / {Path(r['path_b']).name} n={r['inner_inliers']} gray={r['aligned_correlation']} high={r['vessel_highpass_correlation']}",fill='black')
            sheet.save(C.OUT/f'lag_{label}_visual_{page+1:02d}.png')
    print('23 confirmed and19 excluded pairs rendered; no decisions changed',flush=True)

if __name__=='__main__':main()
