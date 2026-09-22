"""Check public sample images, rendered equations, and desktop/mobile layout."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--browser',default='/bin/google-chrome')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    url='https://huggingface.co/ntc-ai/anima-concept-sliders'
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=args.browser,headless=True,args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(viewport={'width':1440,'height':1050})
        assert page.goto(url,wait_until='domcontentloaded',timeout=60000).status==200
        page.wait_for_selector('img[alt^="Bad Intent: Particle first"]',timeout=30000)
        lead=page.locator('img[alt$="Off third"]')
        assert lead.count()==3
        assert lead.nth(0).get_attribute('alt').startswith('Bad Intent:')
        assert 'assets/featured-v2-bad-intent-balcony.jpg' in lead.nth(0).get_attribute('src')
        assert lead.nth(1).get_attribute('alt').startswith('Moonlit:')
        assert lead.nth(2).get_attribute('alt').startswith('Candlelit:')
        assert lead.evaluate_all('(els)=>els.every(e=>!e.closest("details"))')
        # Open folded extra prompts and scroll through the card to trigger lazy images.
        page.locator('details').evaluate_all('(els)=>els.forEach(e=>e.open=true)')
        selector='img[alt*="matched strength-one samples"], img[alt$="Off third"]'
        images=page.locator(selector)
        assert images.count()==12,images.count()
        for i in range(images.count()):
            im=images.nth(i);im.scroll_into_view_if_needed()
            im.evaluate('(e)=>e.loading="eager"')
        page.wait_for_function('(selector)=>Array.from(document.querySelectorAll(selector)).every(i=>i.complete&&i.naturalWidth>0)',arg=selector,timeout=60000)
        assert page.locator('.katex-error').count()==0
        equations=page.locator('.katex-display').count()
        assert equations>=7,equations
        headings=page.locator('h2').all_text_contents()
        def pos(part):return next(i for i,v in enumerate(headings) if part in v)
        assert pos('Samples')<pos('Get the adapters')<pos('How the sliders learn')
        assert page.locator('a[href="https://github.com/mikkel/sliders-conceptmod/tree/main/packages/concept-slider-core"]').count() > 0
        assert page.locator('a[href="https://github.com/HyperGAN/anima-particle-sliders#comfyui"]').count() > 0
        for name in ('bad-intent','candlelit','moonlit'):
            assert page.locator(f'a[href*="distilled/comfyui/{name}-unit-alpha.safetensors"]').count()>0
        for name in ('candlelit','moonlit'):
            assert page.locator(f'a[href*="weights/{name}-unit-alpha.safetensors"]').count()>0
        assert page.locator('a[href*="weights/bad-intent-balanced-alpha.safetensors"]').count()>0
        first=images.first;first.scroll_into_view_if_needed()
        page.screenshot(path=str(args.output/'desktop.png'))
        records={}
        for name,width in [('desktop',1440),('mobile',390)]:
            page.set_viewport_size({'width':width,'height':900})
            first.scroll_into_view_if_needed()
            sizes=images.evaluate_all('(els)=>els.map(e=>({width:e.getBoundingClientRect().width,naturalWidth:e.naturalWidth}))')
            assert all(s['width']>200 and s['width']<=width and s['naturalWidth']==1152 for s in sizes),sizes
            records[name]=sizes
            page.screenshot(path=str(args.output/(name+'.png')))
        result=dict(passed=True,url=url,sample_grids=images.count(),loaded_images=True,equations=equations,
                    lead_slider='Bad Intent',comparison_order=['Particle','Distill','Off'],comparison_strengths=[1,1,0],
                    featured_case='balcony',featured_seed=29027,
                    samples_before_downloads_and_formulation=True,shared_core_link=True,comfyui_plugin_link=True,
                    layout=records,math_errors=0)
        (args.output/'browser.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        browser.close()

if __name__=='__main__':main()
