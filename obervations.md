# Observations from the test sets

M, 042 refers to image 42 from the M and G sets.

## Contrast (missing)

- M 019: road marking
- M,V 042: street sign
- M 232: access cover

## Wear (missing)

- M,G 051: road marking
- M,G 253: road marking
- M 298: road marking

## (often Small) unlabelled object (extras)

- M,V 010: road marking, street sign
- G 010: road marking*2
- M 025: street light
- M 042: street sign
- G 058: road marking
- M,V 132: barrier
- G 160: road marking
- M 177: street light
- G 232: road marking *3
- G 238: pavement
- M 250: street light
- M,G 253: road marking, street sign
- M,G 256: road marking, barrier
- M 281: street light
- M 286: road marking, street sign
- M 288: street light, street sign
- V 288: street light
- M 298: street light, barrier
- M,G 301: traffic light * 2, road marking
- M,G 304: road marking, traffic light *3
- M,V 311: traffic light
- M 314: traffic light * 2

## Small labelled object (missing)

- M,V 010: street light, road marking
- V 051: street light
- M,V 103: street light
- G 110: sidewalk*2
- G 118: sidewalk
- M 125: road marking
- G 125: road marking*2
- M,G 132: sidewalk
- M,G 160: sidewalk
- G 177: sidewalk
- M,G 190: road marking
- G203: tiny sidewalk
- M,G 206: drainage
- V 281: street light
- M,G 311: sidewalk

## Broken, together (naive segnet might duplicate)

- M 019: street lamp
- lots of road markings
- M 118: sidewalk

## Broken, separate (duplicates)

- M,G 042: road marking
- M,G 281: road marking
- M,G 286: road marking
- M,G 288: road marking * 5
- M,G 311: pavement

## Blurry (missing)

- M,G 042: drainage
- M,V 180: street light

## Incorrect (extras)

- M 128: road marking
- G 128: road marking*2
- G 180: duplicate road marking, sidewalk that actually is a car!
- V 206: street light (actually telephone pole)
- M 298: sidewalk
- M 314: street light (should be barrier)

## Chaotic (many close together)

- M,G 156: road markings
- M,G 298: road markings

## Perspective/Low Numbers/Occlusion (missing)

- V 125: barrier (slanted and blurry)
- M,G 206: drainage (perspective)
- M,G 239: access cover (occlusion)
- M,G 256: access cover (low num)
- M 304: road marking (arrow up close perspective)
- M,G,V 311: road marking, street light(front or far), sidewalk
- V 311: traffic light
- M,V 314: barrier (back)
