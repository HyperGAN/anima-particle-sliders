"""The fixed scene grid every character vibe is taught and checked on.

Nothing here depends on the vibe. A person, a place and a light are paired by position
and given one of five framings in turn, so three scenes in five are portraits. The first
twenty-four scenes teach. The next sixteen are held back to confirm the finished slider.
The vibe phrase enters only through three fixed treatment sentences.
"""

PEOPLE = [
    ("laundromat", "a young woman with short red hair and a green scarf", "a quiet laundromat with a row of white washing machines and a yellow bench", "in soft morning light"),
    ("bakery", "an elderly man with a bald head and a white mustache, wearing a tan vest", "a small bakery with wooden bread shelves and a glass counter", "in warm afternoon light"),
    ("court", "a man in his thirties with a shaved head and dark skin, wearing a plain white t-shirt", "a city basketball court with a chain-link fence and painted lines", "under bright noon sun"),
    ("bus-stop", "a young woman with twin black braids and a yellow raincoat", "a bus stop with a glass shelter and a route map", "on a gray rainy day"),
    ("classroom", "a middle-aged woman with curly gray hair and glasses, wearing a lavender blouse", "a school classroom with a green chalkboard and wooden desks", "in even daylight"),
    ("hardware", "a broad-shouldered man with a red beard, wearing a plaid shirt", "a hardware store aisle with shelves of paint cans and tools", "under bright ceiling lights"),
    ("record-shop", "a young man with silver-dyed hair and a denim vest", "a record shop with crates of vinyl and posters on the wall", "in warm evening light"),
    ("pharmacy", "a woman with a long black ponytail and tan skin, wearing a white lab coat", "a pharmacy with shelves of small boxes and a blue counter", "under cool fluorescent light"),
    ("aquarium", "a tall woman in her forties with a silver pixie cut and a navy turtleneck", "an aquarium hall with a large blue tank and a brass railing", "in dim blue light"),
    ("ramen-shop", "a stocky man with a black topknot and a green hoodie", "a ramen shop with a wooden counter and red paper lanterns", "in warm evening light"),
    ("flower-stall", "an elderly woman with a white bun and a floral shawl", "a flower market stall with buckets of tulips and a striped awning", "in bright morning light"),
    ("roof-garden", "a young man with curly brown hair and freckles, wearing an orange windbreaker", "a rooftop garden with wooden planters and a round water tank", "in clear noon light"),
    ("hotel-lobby", "a woman with long straight blonde hair and a red cardigan", "a hotel lobby with a marble desk and a brass luggage cart", "in soft lamp light"),
    ("garage", "a man with a gray crew cut and a brown leather jacket", "an auto garage with a red tool chest and a car on a lift", "under cool fluorescent light"),
    ("gymnasium", "a young man with spiky blond hair and a blue blazer", "a gymnasium with wooden bleachers and a climbing rope", "in even indoor light"),
    ("gallery", "a woman with a dark afro and gold earrings, wearing a mustard sweater", "an art gallery with white walls and framed abstract paintings", "under bright track lighting"),
    ("post-office", "a middle-aged man with thinning black hair and round glasses, wearing a gray suit vest", "a post office with wooden sorting shelves and a brass scale", "in soft window light"),
    ("arcade", "a young woman with pink bobbed hair and a black choker, wearing a white tank top", "a video arcade with glowing cabinets and a checkered floor", "in colorful neon light"),
    ("pottery", "an old man with a long gray ponytail and a denim shirt", "a pottery studio with shelves of clay bowls and a potter's wheel", "in warm afternoon light"),
    ("mountain-hut", "a woman with short black hair and brown skin, wearing a green field jacket", "a mountain hut with a stone fireplace and wooden skis on the wall", "in flickering firelight"),
    ("kitchen", "a heavyset man with a brown mustache and a white chef jacket", "a restaurant kitchen with steel counters and hanging copper pots", "under bright overhead light"),
    ("skate-park", "a young woman with long auburn hair and a gray hoodie", "a skate park with concrete ramps and a painted wall", "in golden evening light"),
    ("fruit-market", "a man with dreadlocks tied back and a beige linen shirt", "a fruit market with crates of oranges and a canvas canopy", "in bright morning light"),
    ("carpentry", "a woman with a gray braid over one shoulder and a blue work apron", "a carpentry workshop with a wooden workbench and hanging saws", "in dusty window light"),
    ("bike-shop", "a lanky man with a red bandana and a gray tank top", "a bicycle repair shop with wheels hanging from the ceiling and a blue workstand", "in bright morning light"),
    ("tea-house", "a woman in her fifties with a black bob and a beige cardigan", "a tea house with low wooden tables and paper screens", "in soft afternoon light"),
    ("observatory", "a young woman with long silver-blonde hair and a navy sweater", "an observatory room with a brass telescope and star charts", "in dim evening light"),
    ("barbershop", "a bald man with a thick black beard and a white undershirt", "a barbershop with red leather chairs and round mirrors", "under bright vanity lights"),
    ("fish-market", "an older woman with short gray curls and a yellow rain jacket", "a fish market with ice trays and hanging scales", "in cold morning light"),
    ("print-shop", "a man in his twenties with a brown ponytail and an ink-stained apron", "a print shop with a letterpress machine and drying racks", "in warm lamp light"),
    ("climbing-gym", "an athletic woman with a short red undercut and a black tank top", "a climbing gym with colorful holds and blue crash mats", "under bright ceiling lights"),
    ("cafe", "a round-faced man with wavy blond hair and a green sweater vest", "a small cafe with a chalkboard menu and a pastry case", "in soft morning light"),
    ("tram", "a tall woman with dark skin and long braids, wearing a denim jacket", "a tram car with wooden seats and brass poles", "in afternoon sun through the windows"),
    ("electronics", "a thin old man with wild white hair and a brown cardigan", "an electronics workshop with oscilloscopes and shelves of parts", "under a green desk lamp"),
    ("dance-studio", "a young man with a black fade haircut and a white t-shirt", "a dance studio with a wall of mirrors and a wooden barre", "in bright even light"),
    ("vineyard", "a woman with sun-tanned skin and a straw-blond braid, wearing a plaid shirt", "a vineyard shed with wooden barrels and crates of grapes", "in golden late light"),
    ("museum", "a man with graying temples and a navy blazer", "a museum hall with a dinosaur skeleton and marble columns", "in cool skylight"),
    ("laboratory", "a woman with short curly black hair and safety glasses, wearing a gray hoodie", "a laboratory with glass flasks and a periodic table poster", "under white fluorescent light"),
    ("greenhouse", "a broad man with a shaved head and a tattooed forearm, wearing an olive t-shirt", "a plant nursery with rows of seedlings and hanging baskets", "in bright filtered daylight"),
    ("toy-store", "a petite woman with pink-framed glasses and a brown bun, wearing a striped shirt", "a toy store with shelves of wooden trains and plush animals", "in cheerful warm light"),
    ("stationery", "a man with neat gray hair and a mustard cardigan", "a stationery shop with shelves of notebooks and jars of pens", "in soft window light"),
    ("pool", "a woman with slicked-back black hair and a teal track jacket", "an indoor swimming pool with lane ropes and white tiles", "in bright even light"),
    ("plant-van", "a young man with a buzz cut and a striped apron", "a loading bay with crates of potted plants and a white van", "in overcast daylight"),
    ("reading-room", "an older woman with a silver braid and a burgundy sweater", "a reading room with green lamps and long oak tables", "in quiet lamp light"),
]
# Three portrait framings in every five scenes, because a slider is most often switched on over a portrait.
FRAMINGS = [
    ("half-body", "A half-body portrait of {who}, standing in {where}, {light}."),
    ("waist-up", "A waist-up portrait of {who}, standing in {where}, {light}."),
    ("full-length", "A full-length shot of {who}, standing in {where}, {light}."),
    ("chest-up", "A chest-up portrait of {who}, standing in {where}, {light}."),
    ("seated", "A medium shot of {who}, sitting on a stool in {where}, {light}."),
]
# The last four are watched for a structural failure while the slider trains. They choose nothing else.
TEACHING, CONFIRMATION, WATCHED = 24, 16, 4
ADULTS = " All depicted people are adults."


def article(vibe):
    return ("an " if vibe[:1].lower() in "aeiou" else "a ") + vibe


def scenes():
    for index, (tag, who, where, light) in enumerate(PEOPLE):
        framing, sentence = FRAMINGS[index % len(FRAMINGS)]
        yield dict(id=tag, framing=framing, seed=70001 + index, setting=where.split(" with ")[0],
                   neutral="Anime illustration. " + sentence.format(who=who, where=where, light=light))


def treatments(scene, vibe):
    """Three fixed ways of asking for the vibe on the same person, place, light and framing."""
    base, a, place = scene["neutral"].rstrip("."), article(vibe), scene["setting"]
    kept = f" The setting and its lighting are unchanged, and the framing is the same."
    return [
        f"{base}, now dressed and styled as {a}, with the full costume, headwear and accessories of {a}.{kept}",
        f"{base}, transformed into {a}: an unmistakable {vibe} costume, silhouette and accessories.{kept}",
        f"{vibe.capitalize()} character design. {base}, wearing the complete outfit of {a} in {place}.{kept}",
    ]


def explicit(positive, vibe):
    return f"{positive} The character has the unmistakable silhouette, costume and presence of {article(vibe)}."


def spec(vibe):
    rows = list(scenes())
    assert len(rows) == TEACHING + CONFIRMATION + WATCHED and len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        row["neutral"] += ADULTS
    teaching = [dict(row, treatments=[dict(positive=p + ADULTS, explicit=explicit(p, vibe) + ADULTS)
                                      for p in treatments(dict(row, neutral=row["neutral"].removesuffix(ADULTS)), vibe)])
                for row in rows[:TEACHING]]
    return dict(vibe=vibe, teaching=teaching, confirmation=rows[TEACHING:TEACHING + CONFIRMATION],
                watched=rows[TEACHING + CONFIRMATION:])
