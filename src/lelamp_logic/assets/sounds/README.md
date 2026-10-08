# LeLamp sound effects

`make_sounds.py [out_dir]` (numpy + stdlib, CPU, a few seconds) writes one folder per emotion,
4 different sounds per emotion, 3 variations of each (120 WAVs, 22 kHz mono):
`<emotion>/<emotion>_<sound>_<1..3>.wav` (1 = a bit lower/slower, 2 = original, 3 = a bit higher/quicker).

sound_node: /lelamp/play_sound with sound="happy" plays a random happy sound; sound="happy_trill" plays a random variation of that one.

| Emotion | Lamp moment | Sounds |
|---|---|---|
| wake | engaged (wake word / looked long enough) | bweeoop, powerup, hello, sparkle |
| notice | attentive (face found), quiet | hm, blip, oh, tick |
| thinking | waiting on llm_node | burble, computing, hmmm, whirr |
| ack | command understood (obey) | blip_blip, okay, yep, roger |
| confused | parse failed / timeout | uh_oh, huh, wobble, scramble |
| happy | success, found the object | trill, giggle, yay, whistle |
| curious | look_around / new object | bip_bwip, ooh, whats_that, peek |
| sigh | bored glance | sigh, meh, deflate, hum |
| stretch | bored stretch | yawn, creak, big_stretch, pop |
| sleep | back to idle | goodnight, powerdown, snooze, lullaby |

Add a sound: add a line to that emotion's dict in SOUNDS and rerun. Add an emotion: add a new dict.

Free CC0 packs for more: Kenney Interface Sounds (https://opengameart.org/content/interface-sounds),
Kenney Sci-Fi Sounds (https://opengameart.org/content/sci-fi-sounds). Browser generator: https://sfxr.me
