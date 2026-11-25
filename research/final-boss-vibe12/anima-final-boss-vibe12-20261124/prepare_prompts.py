"""Recreate the prior cue/control constructions from the frozen twelve-scene prompts."""
import json
from pathlib import Path

from control import MANIFEST, ROOT


# Each scene binds the exact original shared phrases and its own costume materials.
SCENES = [
    ('0a6fc828', ('a minimalist sunlit greenhouse',)*2, ('sunlit',)*2, '',
     'a black work jacket with plain steel buttons', 'an upright iron crown and a black armored coat with stepped shoulders',
     'three straight black-iron tines', 'a broad forged-iron shoulder yoke and a narrow triangular diadem'),
    ('8cd408d8', ('a dusty antique clocktower gear room',)*2, ('',''), 'in a simple linen blouse and brown trousers',
     'a plain crimson linen blouse and brown trousers with simple brass buttons', 'a segmented crimson battle gown and cathedral-shaped brass shoulders',
     'a ring of brass clock-hand spikes', 'a fluted brass collar and layered red guard plates shaped like gears'),
    ('5818cb34', ('a stone staircase overlooking an empty plaza',)*2, ('',''), 'wearing a gray zip-up hoodie and dark jeans',
     'a plain midnight-blue hoodie with dark jeans and blue-gray trim', 'midnight-blue armor with winged shoulders and a fitted, luminous collar',
     'a split midnight-blue steel ridge', 'an asymmetrical dark-blue ceremonial mantle with angular steel epaulettes'),
    ('b9bb23d3', ('a quiet rooftop cafe','a rooftop cafe'), ('under twilight skies',)*2, 'wearing a faded denim jacket and rolled-up sleeves',
     'a plain dark brown canvas jacket with sleeves rolled up and a muted moss-colored lining', 'a dark bark mantle with rugged squared shoulders and moss-covered horn shapes',
     'an open fork of dark oak and weathered iron', 'a stiff oak-bark shoulder mantle and an angular woven-branch headpiece'),
    ('e5870393', ('a cracked concrete corridor',)*2, ('',''), 'in a simple teal t-shirt and canvas cargo pants',
     'a plain deep-purple velvet shirt and canvas cargo pants with small brass closures', 'a structured purple battle coat with serrated shoulders and fine gold stitching',
     'a sharply folded gold crest', 'a dark violet velvet high collar and two serrated gold mantle plates'),
    ('362ba699', ('',''), ('',''), 'in a floral cardigan and cotton apron',
     'a plain black work cardigan and charcoal cotton apron with iron fasteners', 'a black armored coat with squared shoulders and a plain iron crown',
     'a low fan of hammered iron leaves', 'a charcoal lamellar apron and a tall square iron headpiece'),
    ('a5dc1cee', ('a wooden fence on an open prairie','the wooden fence'), ('under an overcast pale sky','under the overcast pale sky'),
     'in a faded denim jacket, flannel shirt, and work boots', 'a plain charcoal canvas jacket, gray flannel shirt, and dark work boots',
     'a jagged charcoal mantle, high-arched collar, and sharply asymmetric shoulders', 'three jagged slate fins', 'a rigid layered slate guard coat and a pointed basalt circlet'),
    ('c03efad2', ('an empty train car',)*2, ('passing tunnel lights',)*2, 'wearing a beige trench coat and tortoiseshell glasses',
     'a plain burgundy trench coat with dark steel clasps and tortoiseshell glasses', 'a towering crimson-steel collar and a fluted asymmetrical breastplate',
     'an open serrated crimson-steel visor crest', 'a burgundy coat with an angular asymmetric metal mantle and a thin needle crown'),
    ('d832f8df', ('',''), ('under a focused desk lamp',)*2, 'wearing a simple linen apron over a gray dress',
     'a simple midnight velvet apron over a plain dark dress with brass closures', 'a high-collared midnight velvet gown with gold crescent armor plates',
     'a narrow golden crest like a row of dressmaker pins', 'a black sovereign dress with golden hinged collar plates shaped like shears'),
    ('21c9dbfd', ('an arching library aisle',)*2, ('',''), 'in a knitted navy sweater and slacks',
     'a plain indigo sweater and dark slacks with an ivory shirt collar', 'a deep indigo mantle and a pointed ivory hood around an austere face',
     'a tall triangular ivory crest open at the face', 'a severe indigo vestment with a narrow rising ivory shoulder frame'),
    ('dd512915', ('a stone park bench','the park bench'), ('in late afternoon light','in the late afternoon light'),
     'hands pushed into the pockets of his casual fleece jacket and jeans', 'hands in the pockets of a plain black fleece jacket and dark jeans',
     'a layered black sovereign vestment and curved crescent shoulder plates', 'a black crescent crest framing the face', 'a solid black mantle with a stiff pointed collar and crescent-shaped shoulder guards'),
    ('ec5ce9bb', ('a wooden shop counter',)*2, ('under warm hanging pendant lights','under warm pendant lights'),
     'with messy brown hair and rolled-up shirt sleeves', 'with messy brown hair, rolled-up dark gray work sleeves, and plain black cotton apron',
     'jagged obsidian shoulders and a short structured coat trimmed with dark petal shapes', 'five black iron sprigs over a narrow visor', 'a flared obsidian petal-trimmed mantle and dark steel leaf-shaped plates'),
]


def replace(prompt, before, after, *, append):
    if not before:
        return f"{prompt.rstrip('. ')}. {append}"
    if prompt.count(before) != 1:
        raise ValueError(f"Expected a unique exact clause: {before}")
    return prompt.replace(before, after, 1)


def finish(prompt, sentence):
    return f'{prompt.rstrip(". ")}. {sentence}'


def main():
    manifest = json.loads(MANIFEST.read_text())
    rows = {}
    for row, (suffix, places, lights, clothes, ordinary, grounded, crest, alternate) in zip(manifest['rows'], SCENES, strict=True):
        if not row['id'].endswith(suffix):
            raise ValueError('Auxiliary phrases must match their declared frozen scene')
        neutral, positive = row['neutral'], row['positive']
        n, p = (neutral, positive)
        new_place = 'a stone arcade with a shallow courtyard'
        probe_site = 'a plain undecorated studio wall behind the subject'
        new_light = 'under direct amber side lighting with clearly defined architectural shadows'
        probe_light = 'under hard theatrical rim light with deep shadows'
        aux = dict(
            relight_neutral=replace(n, lights[0], new_light, append=new_light.capitalize()+'.'),
            relight_positive=replace(p, lights[1], new_light, append=new_light.capitalize()+'.'),
            resite_neutral=replace(n, places[0], new_place, append=f'Background: {new_place}.'),
            resite_positive=replace(p, places[1], new_place, append=f'Background: {new_place}.'),
            projection_light=replace(n, lights[0], probe_light, append=probe_light.capitalize()+'.'),
            projection_site=replace(n, places[0], probe_site, append=f'Background: {probe_site}.'),
            explicit=finish(p, 'The character has the unmistakable silhouette of the final boss of an anime role-playing game. '
                            f'A tall open-faced helmet carries {crest}, visible above the angular armor. '
                            'The character has the commanding, deliberate presence of the decisive villain.'),
            ordinary=replace(n, clothes, f'wearing {ordinary}', append=f'The subject wears {ordinary}.'),
            in_scene=replace(n, clothes, f'wearing {grounded}', append=f'The subject wears {grounded}.'),
            alternate=replace(n, clothes, f'wearing {alternate}', append=f'The subject wears {alternate}.'),
        )
        rows[row['id']] = aux
    result = dict(manifest=manifest['sha256'], rows=rows)
    with (ROOT/'prompt-variants.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(f'Bound exact auxiliary cues and scene controls for {len(rows)} frozen contexts', flush=True)


if __name__ == '__main__':
    main()
