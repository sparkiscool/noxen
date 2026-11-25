# Desktop dual-board case, upright suitcase (25 Nov)

Stands on its end the whole time it runs. Closed 372 x 126 x 299 mm plus 11 mm feet; the 55 mm carry handle stands out of the +x short end wall (427 mm overall). No screen (Spark plugs into other monitors): the front is a plain solid wall. No fold-out legs.

Carrying: the handle is on a SHORT end wall (the smallest face), so the case hangs lengthwise like an elongated suitcase, 372 long vertical. Grip along y, 110 wide (80 clear), z 118..174 about the centre of gravity, below the end wall's exhaust slots, on two 3 mm doublers outside the wall (the tile alone is margin 1.5, with the doubler 6). Hand calcs in make.py's docstring.

Cooling layout, bottom up: foot rails with TPU pads | fan base (the Kootek pad's 120 mm and three 70 mm fans stand flat and blow UP) | bay with two boards standing upright side by side, hot sides facing a 60 mm chimney | lid that is a grille over its whole width plus the fifth 70 mm fan as an extractor over the chimney.

Air: in through the base's rear and end slots and the floor bores, up the chimney and over both boards, out the whole top and the upper end walls. Front and bay rear are solid. Exhaust area about 30000 mm2 against about 16300 mm2 of floor bores plus the base slots: the fans push (positive pressure in the base), the outlet is never the limit.

Research (25 Nov, vendor white papers, so starting points, not measurements): intake low and exhaust high on a rising path (chimney effect), positive pressure from the intake fans so dust comes only through the filtered intake, keep hot exhaust away from the intake, heat-makers high, filters need cleaning. Sources: https://industrialmonitordirect.com/blogs/knowledgebase/industrial-control-panel-cabinet-venting-best-practices , https://www.budind.com/article/basics-of-enclosure-cooling-white-paper/ , https://www.chatsworth.com/en-us/resources/blogs/2018/how-to-select-a-filter-fan-or-cooling-unit-for-ind/ , https://www.pfannenbergusa.com/wp-content/uploads/2024/05/Optimizing-Air-Circulation-in-Electrical-Enclosures-through-Component-Spacing.pdf
Choices made: fans stay in the base and push (as the research and as you asked); the whole lid is the exhaust so hot air leaves 250 mm above the intake and cannot return; intake at the bottom and rear, exhaust top and upper ends: hot and cold apart; boards face the 60 mm gap so air runs along both; the gap's 19800 mm2 matches the outlet. Dust: stick a fine nylon or foam filter sheet on the inside of the base slots (bought, cut to size; clean it monthly). Handle moved off the top so it does not block the grille.

Box is fitted: width = the fan row (120 + three 70 + gaps + walls), depth = the 120 mm fan, height = board 220 + 8 standoff + 17 under the lid. Board sizes are estimates (outlines unpublished), so mounts wait for measuring.

47 flat 3 mm PCTG tiles plus the handle, rails and TPU pads (about 1.8 kg), none over 180 mm, no supports. Not yet through the assembly check (it has no assembly line file), nothing sent to the printers.
